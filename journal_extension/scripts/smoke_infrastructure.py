from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.checkpointing import recover_latest, save_torch_checkpoint
from cropcop_je.environment import capture_environment
from cropcop_je.g1 import validate_dependency_environment, validate_dependency_lock_object
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.publication import publish_to_github_branch
from cropcop_je.session import SessionBudget
from cropcop_je.smoke_handoff import (
    RestoreSequence,
    SMOKE_A_EVIDENCE,
    SMOKE_A_MANIFEST,
    SMOKE_B_EVIDENCE,
    SMOKE_B_MANIFEST,
    build_smoke_a_manifest,
    manifest_self_hash,
    recovery_file_records,
    restore_verified_recovery_bundle,
    snapshot_files,
    validate_smoke_a_manifest,
    verify_smoke_a_export,
)
from cropcop_je.source_state import verify_clean_source

SEED = 99173
INITIAL_STEPS = 3
RESUME_STEPS = 1
DEPENDENCY_LOCK_REL = "journal_extension/locks/execution_dependency_lock.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_runtime(repo: Path) -> tuple[dict, dict, str]:
    dep_path = repo / DEPENDENCY_LOCK_REL
    dep = json.loads(dep_path.read_text(encoding="utf-8"))
    errors = validate_dependency_lock_object(dep) + validate_dependency_environment(dep)
    if errors:
        raise RuntimeError("frozen execution dependency lock failed: " + "; ".join(errors))
    env = capture_environment()
    if env.get("cuda_available") is not True:
        raise RuntimeError("real Kaggle infrastructure smoke requires CUDA")
    return dep, env, sha256_json(env)


def _prepare_synthetic_marker(root: Path, qualification_id: str) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(root / "SYNTHETIC_SMOKE.json", {
        "schema_version": "1.0",
        "qualification_id": qualification_id,
        "synthetic": True,
        "scientific": False,
        "infrastructure_only": True,
        "derived_from_cropcop_images": False,
        "restricted_cropcop_data_accessed": False,
    })


def _seed_all() -> None:
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _components():
    import torch
    device = torch.device("cuda")
    model = torch.nn.Linear(16, 4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    return device, model, optimizer, scheduler, scaler


def _advance(device, model, optimizer, scheduler, scaler, steps: int) -> None:
    import torch
    for _ in range(steps):
        x = torch.randn(16, 16, device=device)
        y = torch.randint(0, 4, (16,), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()


def _identity(source_sha: str, lane: str, qualification_id: str) -> dict:
    return {
        "experiment_id": "INFRA-SMOKE",
        "authority_id": "NON_SCIENTIFIC_INFRASTRUCTURE",
        "source_git_commit": source_sha,
        "lane_id": lane,
        "qualification_id": qualification_id,
        "synthetic_only": True,
        "scientific": False,
    }


def _payload(identity: dict, model, optimizer, scheduler, scaler, optimizer_step: int) -> dict:
    return {
        "schema_version": "smoke-2.0",
        "identity": identity,
        "identity_sha256": sha256_json(identity),
        "created_at_utc": utc_now(),
        "student": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": {"smoke_seed": SEED},
        "epoch": 0,
        "batch_in_epoch": optimizer_step,
        "optimizer_step": optimizer_step,
        "examples_seen": optimizer_step * 16,
        "data_order_state": {"synthetic": True, "next_batch_in_epoch": optimizer_step},
        "selection_state": {"history": [], "best": None},
    }


def _publish_terminal_evidence(repo: Path, source_sha: str, run_id: str, path: Path, evidence: dict) -> tuple[str, str]:
    evidence["git_publication_status"] = "PENDING"
    atomic_write_json(path, evidence)
    try:
        branch = publish_to_github_branch(
            repo_dir=repo,
            source_git_sha=source_sha,
            run_id=run_id,
            files=[path],
        )
        evidence["git_publication_status"] = "PASS"
        evidence["public_safe_evidence_branch"] = branch
        atomic_write_json(path, evidence)
        branch2 = publish_to_github_branch(
            repo_dir=repo,
            source_git_sha=source_sha,
            run_id=run_id,
            files=[path],
        )
        if branch2 != branch:
            raise RuntimeError("public evidence branch changed during terminal publication")
        return branch, sha256_file(path)
    except Exception:
        evidence["git_publication_status"] = "FAIL"
        evidence.pop("public_safe_evidence_branch", None)
        atomic_write_json(path, evidence)
        raise


def _common_preflight(args, *, mutable_roots: list[Path]) -> tuple[Path, dict, dict, str, dict, SessionBudget]:
    if not os.environ.get("CROPCOP_GITHUB_TOKEN"):
        raise RuntimeError("required Kaggle Secret missing: CROPCOP_GITHUB_TOKEN")
    repo = Path(args.repo_root).resolve()
    source = verify_clean_source(
        repo,
        authorized_source_sha=args.authorized_source_sha,
        output_roots=mutable_roots,
    )
    dep, env, env_sha = _load_runtime(repo)
    budget = SessionBudget.from_environment(require_global_clock=True)
    if not budget.can_start_phase(900, estimated_checkpoint_seconds=60, estimated_sync_seconds=0):
        raise RuntimeError("not enough notebook-global safe time remains for infrastructure smoke")
    return repo, dep, env, env_sha, source, budget


def smoke_write(args) -> int:
    export_root = Path(args.smoke_a_export_root).resolve()
    runtime_root = Path(args.runtime_output_root).resolve()
    synthetic_root = Path(args.synthetic_root).resolve()
    if export_root.exists() and any(export_root.iterdir()):
        raise RuntimeError(f"Smoke-A export root must start empty: {export_root}")
    qualification_id = args.qualification_id.strip() or (
        f"INFRA-SMOKE-{args.authorized_source_sha[:12]}-{uuid.uuid4().hex[:12]}"
    )
    repo, dep, env, env_sha, source, budget = _common_preflight(
        args, mutable_roots=[export_root, runtime_root, synthetic_root]
    )
    _prepare_synthetic_marker(synthetic_root, qualification_id)
    export_root.mkdir(parents=True, exist_ok=True)

    _seed_all()
    device, model, optimizer, scheduler, scaler = _components()
    _advance(device, model, optimizer, scheduler, scaler, INITIAL_STEPS)
    identity = _identity(args.authorized_source_sha, args.lane, qualification_id)
    payload = _payload(identity, model, optimizer, scheduler, scaler, INITIAL_STEPS)
    ref, save_seconds = save_torch_checkpoint(
        export_root,
        kind="latest",
        payload=payload,
        expected_identity=identity,
    )

    verified_path, recovered, recovery_event = recover_latest(export_root, expected_identity=identity)
    if ref.sha256 != sha256_file(verified_path):
        raise RuntimeError("Smoke-A local checkpoint verification SHA mismatch")
    if int(recovered["optimizer_step"]) != INITIAL_STEPS:
        raise RuntimeError("Smoke-A local checkpoint optimizer step verification failed")

    evidence = {
        "schema_version": "2.0",
        "status": "PASS",
        "qualification_id": qualification_id,
        "mode": "WRITE",
        "scientific": False,
        "synthetic_unprotected_data_only": True,
        "source_git_sha": args.authorized_source_sha,
        "source_tree_sha": source["tree_sha"],
        "dependency_lock_sha256": dep["dependency_lock_sha256"],
        "environment_identity_sha256": env_sha,
        "environment": env,
        "lane": args.lane,
        "checkpoint_sha256": ref.sha256,
        "checkpoint_bytes": ref.bytes,
        "checkpoint_identity": identity,
        "checkpoint_identity_sha256": ref.identity_sha256,
        "optimizer_step": ref.optimizer_step,
        "checkpoint_save_seconds": save_seconds,
        "checkpoint_local_verification": "PASS",
        "recovery_candidate": recovery_event.get("candidate"),
        "export_integrity": "PENDING",
        "notebook_session": budget.snapshot(),
        "restricted_cropcop_data_accessed": False,
        "g1_executed": False,
        "g2_executed": False,
        "r04_r05_executed": False,
        "created_at_utc": utc_now(),
    }
    evidence_path = export_root / SMOKE_A_EVIDENCE
    branch, evidence_sha = _publish_terminal_evidence(
        repo, args.authorized_source_sha, f"SMOKE-A-{qualification_id}", evidence_path, evidence
    )

    recovery_files = recovery_file_records(export_root)
    manifest = build_smoke_a_manifest(
        qualification_id=qualification_id,
        source_git_sha=args.authorized_source_sha,
        source_tree_sha=source["tree_sha"],
        dependency_lock_sha256=dep["dependency_lock_sha256"],
        environment_identity_sha256=env_sha,
        checkpoint_relative_path=ref.relative_path,
        checkpoint_sha256=ref.sha256,
        checkpoint_bytes=ref.bytes,
        checkpoint_identity_sha256=ref.identity_sha256,
        optimizer_step=ref.optimizer_step,
        recovery_files=recovery_files,
        smoke_a_evidence_sha256=evidence_sha,
        created_at_utc=utc_now(),
    )
    errors = validate_smoke_a_manifest(
        manifest,
        expected_source_sha=args.authorized_source_sha,
        expected_dependency_lock_sha256=dep["dependency_lock_sha256"],
    )
    if errors:
        raise RuntimeError("Smoke-A manifest finalization failed: " + "; ".join(errors))
    atomic_write_json(export_root / SMOKE_A_MANIFEST, manifest)
    # Re-open through the same verifier Smoke B uses.
    verified = verify_smoke_a_export(
        export_root,
        expected_source_sha=args.authorized_source_sha,
        expected_dependency_lock_sha256=dep["dependency_lock_sha256"],
    )
    evidence["export_integrity"] = "PASS"
    atomic_write_json(evidence_path, evidence)
    # Publication outcome is already PASS; republish the final export-integrity field.
    branch2 = publish_to_github_branch(
        repo_dir=repo,
        source_git_sha=args.authorized_source_sha,
        run_id=f"SMOKE-A-{qualification_id}",
        files=[evidence_path],
    )
    if branch2 != branch:
        raise RuntimeError("Smoke-A terminal evidence branch changed unexpectedly")
    final_evidence_sha = sha256_file(evidence_path)
    if final_evidence_sha != manifest["smoke_a_evidence_sha256"]:
        manifest["smoke_a_evidence_sha256"] = final_evidence_sha
        manifest["manifest_sha256"] = manifest_self_hash(manifest)
        atomic_write_json(export_root / SMOKE_A_MANIFEST, manifest)
        verify_smoke_a_export(
            export_root,
            expected_source_sha=args.authorized_source_sha,
            expected_dependency_lock_sha256=dep["dependency_lock_sha256"],
        )

    print("============================================================")
    print("CROPCOP SMOKE A COMPLETE")
    print("============================================================")
    print("STATUS=PASS")
    print(f"QUALIFICATION_ID={qualification_id}")
    print(f"SOURCE_SHA={args.authorized_source_sha}")
    print(f"DEPENDENCY_LOCK_SHA256={dep['dependency_lock_sha256']}")
    print(f"SMOKE_A_CHECKPOINT_SHA256={ref.sha256}")
    print(f"SMOKE_A_EVIDENCE_SHA256={sha256_file(evidence_path)}")
    print(f"SMOKE_A_EXPORT_ROOT={export_root}")
    print("NEXT_PHASE=smoke-restore")
    print("NEXT: Save & Run All; attach this exact Notebook Output to a fresh Saved Version.")
    print("============================================================")
    return 0


def smoke_restore(args) -> int:
    if not args.smoke_a_input_root:
        raise RuntimeError("smoke-restore requires explicit --smoke-a-input-root")
    runtime_root = Path(args.runtime_output_root).resolve()
    synthetic_root = Path(args.synthetic_root).resolve()
    b_export = Path(args.smoke_b_export_root).resolve()
    input_root = Path(args.smoke_a_input_root).resolve()
    if b_export.exists() and any(b_export.iterdir()):
        raise RuntimeError(f"Smoke-B export root must start empty: {b_export}")

    repo, dep, env, env_sha, source, budget = _common_preflight(
        args, mutable_roots=[runtime_root, synthetic_root, b_export]
    )
    sequence = RestoreSequence()
    sequence.advance("READ_A")
    attached_before = snapshot_files(input_root)
    verified = verify_smoke_a_export(
        input_root,
        expected_source_sha=args.authorized_source_sha,
        expected_dependency_lock_sha256=dep["dependency_lock_sha256"],
    )
    sequence.advance("VERIFY_A")

    manifest = verified.manifest
    evidence_a = verified.evidence
    qualification_id = manifest["qualification_id"]
    identity = evidence_a.get("checkpoint_identity")
    if not isinstance(identity, dict) or sha256_json(identity) != manifest["checkpoint_identity_sha256"]:
        raise RuntimeError("Smoke-A checkpoint identity object missing or digest mismatch")
    if identity.get("qualification_id") != qualification_id:
        raise RuntimeError("Smoke-A checkpoint identity qualification ID mismatch")

    restore_root = runtime_root / "smoke-restore-recovery" / qualification_id
    if restore_root.exists():
        shutil.rmtree(restore_root)
    restore_verified_recovery_bundle(verified, restore_root)
    sequence.advance("RESTORE_A")

    recovered_path, recovered, recovery_event = recover_latest(restore_root, expected_identity=identity)
    observed_sha = sha256_file(recovered_path)
    if observed_sha != manifest["checkpoint_sha256"]:
        raise RuntimeError("observed restored Smoke-A checkpoint SHA differs from expected manifest SHA")
    sequence.advance("RECOVER_A")

    _seed_all()
    device, model, optimizer, scheduler, scaler = _components()
    model.load_state_dict(recovered["student"], strict=True)
    optimizer.load_state_dict(recovered["optimizer"])
    scheduler.load_state_dict(recovered["scheduler"])
    scaler.load_state_dict(recovered["scaler"])
    restored_step = int(recovered["optimizer_step"])
    if restored_step != int(manifest["optimizer_step"]):
        raise RuntimeError("restored optimizer step differs from Smoke-A manifest")
    sequence.advance("LOAD_A")

    _prepare_synthetic_marker(synthetic_root, qualification_id)
    _advance(device, model, optimizer, scheduler, scaler, RESUME_STEPS)
    resumed_step = restored_step + RESUME_STEPS
    if resumed_step <= restored_step:
        raise RuntimeError("Smoke-B optimizer step did not advance")
    sequence.advance("RESUME")
    sequence.require_resume_before_checkpoint()

    # Only after verified READ→VERIFY→RESTORE→RECOVER→LOAD→RESUME may B create new recovery state.
    b_export.mkdir(parents=True, exist_ok=True)
    payload_b = _payload(identity, model, optimizer, scheduler, scaler, resumed_step)
    ref_b, save_seconds_b = save_torch_checkpoint(
        b_export,
        kind="latest",
        payload=payload_b,
        expected_identity=identity,
    )
    sequence.advance("CHECKPOINT_B")
    recovered_b_path, recovered_b, _event_b = recover_latest(b_export, expected_identity=identity)
    if sha256_file(recovered_b_path) != ref_b.sha256 or int(recovered_b["optimizer_step"]) != resumed_step:
        raise RuntimeError("Smoke-B post-resume checkpoint verification failed")

    attached_after = snapshot_files(input_root)
    if attached_after != attached_before:
        raise RuntimeError("attached Smoke-A Notebook Output changed during Smoke-B")

    evidence_b = {
        "schema_version": "2.0",
        "status": "PASS",
        "qualification_id": qualification_id,
        "mode": "RESTORE",
        "scientific": False,
        "synthetic_unprotected_data_only": True,
        "source_git_sha": args.authorized_source_sha,
        "source_tree_sha": source["tree_sha"],
        "dependency_lock_sha256": dep["dependency_lock_sha256"],
        "environment_identity_sha256": env_sha,
        "environment": env,
        "lane": args.lane,
        "smoke_a_manifest_sha256": manifest["manifest_sha256"],
        "smoke_a_evidence_sha256": manifest["smoke_a_evidence_sha256"],
        "smoke_a_expected_checkpoint_sha256": manifest["checkpoint_sha256"],
        "smoke_a_observed_restored_checkpoint_sha256": observed_sha,
        "restored_identity_sha256": manifest["checkpoint_identity_sha256"],
        "restored_optimizer_step": restored_step,
        "resumed_optimizer_step": resumed_step,
        "post_resume_checkpoint_sha256": ref_b.sha256,
        "post_resume_checkpoint_bytes": ref_b.bytes,
        "checkpoint_save_seconds": save_seconds_b,
        "restore_success": True,
        "recover_success": True,
        "resume_success": True,
        "attached_smoke_a_input_unchanged": True,
        "sequence": sequence.events,
        "recovery_candidate": recovery_event.get("candidate"),
        "notebook_session": budget.snapshot(),
        "restricted_cropcop_data_accessed": False,
        "g1_executed": False,
        "g2_executed": False,
        "r04_r05_executed": False,
        "created_at_utc": utc_now(),
    }
    evidence_b_path = b_export / SMOKE_B_EVIDENCE
    branch_b, evidence_b_sha = _publish_terminal_evidence(
        repo, args.authorized_source_sha, f"SMOKE-B-{qualification_id}", evidence_b_path, evidence_b
    )
    recovery_files_b = recovery_file_records(b_export)
    manifest_b = {
        "schema_version": "1.0",
        "qualification_id": qualification_id,
        "mode": "RESTORE",
        "scientific": False,
        "synthetic_unprotected_data_only": True,
        "source_git_sha": args.authorized_source_sha,
        "source_tree_sha": source["tree_sha"],
        "dependency_lock_sha256": dep["dependency_lock_sha256"],
        "environment_identity_sha256": env_sha,
        "smoke_a_manifest_sha256": manifest["manifest_sha256"],
        "smoke_a_evidence_sha256": manifest["smoke_a_evidence_sha256"],
        "smoke_a_expected_checkpoint_sha256": manifest["checkpoint_sha256"],
        "smoke_a_observed_restored_checkpoint_sha256": observed_sha,
        "restored_optimizer_step": restored_step,
        "resumed_optimizer_step": resumed_step,
        "post_resume_checkpoint_relative_path": ref_b.relative_path,
        "post_resume_checkpoint_sha256": ref_b.sha256,
        "post_resume_checkpoint_bytes": ref_b.bytes,
        "smoke_b_evidence_sha256": evidence_b_sha,
        "recovery_files": recovery_files_b,
        "public_safe_evidence_branch": branch_b,
        "created_at_utc": utc_now(),
    }
    manifest_b["manifest_sha256"] = manifest_self_hash(manifest_b)
    atomic_write_json(b_export / SMOKE_B_MANIFEST, manifest_b)

    print("============================================================")
    print("CROPCOP SMOKE B COMPLETE")
    print("============================================================")
    print("STATUS=PASS")
    print(f"QUALIFICATION_ID={qualification_id}")
    print(f"SOURCE_SHA={args.authorized_source_sha}")
    print(f"SMOKE_A_EXPECTED_CHECKPOINT_SHA256={manifest['checkpoint_sha256']}")
    print(f"SMOKE_A_RESTORED_CHECKPOINT_SHA256={observed_sha}")
    print("RESTORE_SUCCESS=true")
    print("RECOVER_SUCCESS=true")
    print("RESUME_SUCCESS=true")
    print(f"RESTORED_OPTIMIZER_STEP={restored_step}")
    print(f"RESUMED_OPTIMIZER_STEP={resumed_step}")
    print(f"SMOKE_B_CHECKPOINT_SHA256={ref_b.sha256}")
    print("SCIENTIFIC=false")
    print("SYNTHETIC_ONLY=true")
    print("============================================================")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["write", "restore"], required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--authorized-source-sha", required=True)
    ap.add_argument("--lane", choices=["K1", "K2", "K3"], required=True)
    ap.add_argument("--runtime-output-root", required=True)
    ap.add_argument("--synthetic-root", required=True)
    ap.add_argument("--smoke-a-export-root", required=True)
    ap.add_argument("--smoke-b-export-root", required=True)
    ap.add_argument("--smoke-a-input-root", default="")
    ap.add_argument("--qualification-id", default="")
    args = ap.parse_args()
    return smoke_write(args) if args.mode == "write" else smoke_restore(args)


if __name__ == "__main__":
    raise SystemExit(main())
