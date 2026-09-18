from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.session import SessionBudget
from cropcop_je.tracka_v12_posttraining_operator import (
    DIRECT_ALLOWED,
    XAI_ALLOWED,
    AUX_ALLOWED,
    STAGE_ALLOWED,
    cli_args,
    validate_state_operator_spec,
)

STAGE_SCRIPT = {
    "direct": "journal_extension/scripts/run_tracka_v12_direct_evidence.py",
    "xai": "journal_extension/scripts/run_tracka_v12_xai.py",
    "auxiliary": "journal_extension/scripts/run_tracka_v12_auxiliary_evidence.py",
}
STAGE_GATE = {
    "direct": Path("public_evidence/DIRECT_STATE_EVIDENCE_GATE.json"),
    "xai": Path("public_xai/XAI_EVIDENCE_GATE.json"),
    "auxiliary": Path("public_evidence/AUXILIARY_STATE_EVIDENCE_GATE.json"),
}


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def verify_self_hash(payload: dict, field: str) -> bool:
    observed = payload.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        return False
    clean = dict(payload)
    clean.pop(field, None)
    return observed == sha256_json(clean)


def terminal_stage(attempt_root: Path, stage: str, experiment_id: str, run_id: str, analysis_sha: str) -> Path | None:
    if not attempt_root.is_dir():
        return None
    for attempt in sorted((p for p in attempt_root.iterdir() if p.is_dir()), reverse=True):
        gate_path = attempt / STAGE_GATE[stage]
        if not gate_path.is_file():
            continue
        try:
            gate = load_json(gate_path)
        except Exception:
            continue
        allowed_status = {"PASS", "WARNING_NONFINITE_MAPS"} if stage == "xai" else {"PASS"}
        if (
            gate.get("status") in allowed_status
            and gate.get("experiment_id") == experiment_id
            and gate.get("run_id") == run_id
            and gate.get("analysis_source_git_commit") == analysis_sha
            and gate.get("v1_test_accessed") is False
            and gate.get("external_surface_accessed") is False
        ):
            if stage == "direct" and (
                gate.get("training_performed") is not False
                or gate.get("optimizer_state_advanced") is not False
                or gate.get("replay_gate", {}).get("status") != "PASS"
            ):
                continue
            if stage == "auxiliary" and (
                gate.get("training_performed") is not False
                or gate.get("optimizer_state_advanced") is not False
                or gate.get("replay_gate", {}).get("status") != "PASS"
            ):
                continue
            if stage == "xai" and gate.get("training_or_adaptation_performed") is not False:
                continue
            return attempt
    return None


def next_attempt(attempt_root: Path) -> Path:
    attempt_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in attempt_root.iterdir():
        if path.is_dir() and path.name.startswith("attempt-"):
            try:
                indices.append(int(path.name.split("-", 1)[1]))
            except ValueError:
                pass
    return attempt_root / f"attempt-{max(indices, default=0) + 1:03d}"


def run_stage(
    *,
    repo: Path,
    state: dict,
    stage: str,
    state_root: Path,
    analysis_sha: str,
    gpu_index: int,
    estimated_seconds: float,
    budget: SessionBudget,
    log_lock: threading.Lock,
) -> tuple[str, Path | None]:
    experiment_id = state["experiment_id"]
    run_id = state["run_id"]
    attempts = state_root / "attempts" / stage
    existing = terminal_stage(attempts, stage, experiment_id, run_id, analysis_sha)
    if existing is not None:
        return "REUSED_PASS", existing
    if not budget.can_start_phase(estimated_seconds, estimated_sync_seconds=1800, extra_reserve_seconds=600):
        return "DEFERRED_SESSION_BUDGET", None

    attempt = next_attempt(attempts)
    output = attempt
    args = state.get("executor_args", {}).get(stage)
    if not isinstance(args, dict):
        raise RuntimeError(f"missing frozen executor args for {experiment_id}:{stage}")
    command = [
        os.environ.get("PYTHON", "python"),
        str(repo / STAGE_SCRIPT[stage]),
        "--repo-root", str(repo),
        "--analysis-source-git-commit", analysis_sha,
        "--run-record", str(Path(state["run_record"]).resolve()),
        "--run-id", run_id,
        "--checkpoint-root", str(Path(state["checkpoint_root"]).resolve()),
        "--output-dir", str(output),
        "--device", "cuda",
        *cli_args(args, STAGE_ALLOWED[stage]),
    ]
    attempt.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    log_path = attempt.parent / f"{attempt.name}.log"
    started_marker = attempt.parent / f"{attempt.name}.started.json"
    atomic_write_json(started_marker, {
        "schema_version": "1.0",
        "stage": stage,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "analysis_source_git_commit": analysis_sha,
        "gpu_index": gpu_index,
        "command_without_secrets": command,
        "session": budget.snapshot(),
    })
    with log_path.open("w", encoding="utf-8") as handle:
        cp = subprocess.run(command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if cp.returncode != 0:
        atomic_write_json(attempt.parent / f"{attempt.name}.failed.json", {
            "schema_version": "1.0",
            "status": "FAIL",
            "stage": stage,
            "experiment_id": experiment_id,
            "return_code": cp.returncode,
            "log_sha256": sha256_file(log_path),
        })
        return "FAILED", None
    terminal = terminal_stage(attempts, stage, experiment_id, run_id, analysis_sha)
    if terminal != attempt:
        raise RuntimeError(f"stage returned success but terminal gate validation failed: {experiment_id}:{stage}")
    with log_lock:
        print(f"POSTTRAINING_STAGE_PASS experiment={experiment_id} stage={stage} gpu={gpu_index}")
    return "PASS", attempt


def seal_state_bundle(state_root: Path, stage_dirs: dict[str, Path], state: dict, analysis_sha: str) -> Path:
    sealed = state_root / "sealed_evidence"
    if sealed.exists():
        shutil.rmtree(sealed)
    sealed.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for stage, source in sorted(stage_dirs.items()):
        destination = sealed / stage
        shutil.copytree(source, destination)
        manifest[stage] = {
            path.relative_to(destination).as_posix(): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(p for p in destination.rglob("*") if p.is_file())
        }
    seal = {
        "schema_version": "1.0",
        "status": "PASS",
        "experiment_id": state["experiment_id"],
        "run_id": state["run_id"],
        "analysis_source_git_commit": analysis_sha,
        "stage_file_manifest": manifest,
        "training_or_adaptation_performed": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
    }
    seal["sealed_evidence_sha256"] = sha256_json(seal)
    atomic_write_json(sealed / "SEALED_STATE_EVIDENCE.json", seal)
    return sealed


def run_subprocess(command: list[str], *, cwd: Path) -> None:
    cp = subprocess.run(command, cwd=cwd, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"subprocess failed rc={cp.returncode}: {' '.join(command)}")


def restore_state_evidence(*, repo: Path, state: dict, state_root: Path, analysis_sha: str) -> dict:
    cert_dir = state_root / "certificates"
    cert_dir.mkdir(parents=True, exist_ok=True)
    restore_cert = cert_dir / "POSTTRAINING_RESTORE_CERTIFICATE.json"
    run_subprocess([
        os.environ.get("PYTHON", "python"),
        str(repo / "journal_extension/scripts/restore_tracka_v12_posttraining_evidence.py"),
        "--dataset-locator", state["evidence_dataset_locator"],
        "--run-id", state["posttraining_public_run_id"],
        "--experiment-id", state["experiment_id"],
        "--analysis-source-git-commit", analysis_sha,
        "--destination-dir", str(state_root),
        "--output", str(restore_cert),
    ], cwd=repo)
    payload = load_json(restore_cert)
    if payload.get("status") not in {"PASS", "EMPTY"}:
        raise RuntimeError(f"post-training evidence restore failed: {state['experiment_id']}")
    if payload.get("status") == "PASS" and payload.get("generation_verified") is not True:
        raise RuntimeError(f"post-training evidence restore lacked generation verification: {state['experiment_id']}")
    return payload


def sync_partial_state(
    *,
    repo: Path,
    state: dict,
    state_root: Path,
    analysis_sha: str,
    stage: str,
) -> dict:
    cert_dir = state_root / "certificates"
    cert_dir.mkdir(parents=True, exist_ok=True)
    partial_cert = cert_dir / f"POSTTRAINING_PARTIAL_SYNC_{stage.upper()}.json"
    partial_cert.unlink(missing_ok=True)
    run_subprocess([
        os.environ.get("PYTHON", "python"),
        str(repo / "journal_extension/scripts/sync_tracka_v12_posttraining_evidence.py"),
        "--source-dir", str(state_root),
        "--dataset-locator", state["evidence_dataset_locator"],
        "--run-id", state["posttraining_public_run_id"],
        "--experiment-id", state["experiment_id"],
        "--analysis-source-git-commit", analysis_sha,
        "--generation-kind", "partial",
        "--output", str(partial_cert),
    ], cwd=repo)
    payload = load_json(partial_cert)
    if (
        payload.get("status") != "PASS"
        or payload.get("generation_kind") != "partial"
        or payload.get("generation_roundtrip_verified") is not True
    ):
        raise RuntimeError(f"partial evidence durability failed: {state['experiment_id']}:{stage}")
    return payload


def finalize_state(
    *,
    repo: Path,
    state: dict,
    state_root: Path,
    stage_dirs: dict[str, Path],
    analysis_sha: str,
    publication_lock: threading.Lock,
) -> dict:
    sealed = seal_state_bundle(state_root, stage_dirs, state, analysis_sha)
    cert_dir = state_root / "certificates"
    cert_dir.mkdir(parents=True, exist_ok=True)
    sync_cert = cert_dir / "POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json"
    sync_cert.unlink(missing_ok=True)
    run_subprocess([
        os.environ.get("PYTHON", "python"),
        str(repo / "journal_extension/scripts/sync_tracka_v12_posttraining_evidence.py"),
        "--source-dir", str(state_root),
        "--dataset-locator", state["evidence_dataset_locator"],
        "--run-id", state["posttraining_public_run_id"],
        "--experiment-id", state["experiment_id"],
        "--analysis-source-git-commit", analysis_sha,
        "--generation-kind", "final",
        "--output", str(sync_cert),
    ], cwd=repo)
    sync = load_json(sync_cert)
    if (
        sync.get("status") != "PASS"
        or sync.get("generation_kind") != "final"
        or sync.get("generation_roundtrip_verified") is not True
    ):
        raise RuntimeError(f"private evidence durability did not pass: {state['experiment_id']}")

    gates = {}
    for stage, path in stage_dirs.items():
        gate = path / STAGE_GATE[stage]
        gates[stage] = {
            "basename": gate.name,
            "logical_relative_path": f"{stage}/{STAGE_GATE[stage].as_posix()}",
            "sha256": sha256_file(gate),
        }
    expected_publication_branch = f"run-evidence/{state['posttraining_public_run_id']}"
    completion = {
        "schema_version": "1.0",
        "status": "PASS",
        "completion_kind": "track_a_posttraining_state",
        "experiment_id": state["experiment_id"],
        "run_id": state["run_id"],
        "posttraining_public_run_id": state["posttraining_public_run_id"],
        "analysis_source_git_commit": analysis_sha,
        "role": state["role"],
        "stage_gates": gates,
        "private_sync_certificate_sha256": sha256_file(sync_cert),
        "private_evidence_dataset_locator": state["evidence_dataset_locator"],
        "private_generation_roundtrip_verified": True,
        "publication_branch": expected_publication_branch,
        "training_or_adaptation_performed": False,
        "optimizer_state_advanced": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
    }
    completion["completion_sha256"] = sha256_json(completion)
    completion_path = cert_dir / "POSTTRAINING_STATE_COMPLETION.json"
    atomic_write_json(completion_path, completion)

    publication_cert = cert_dir / "POSTTRAINING_PUBLICATION_CERTIFICATE.json"
    with publication_lock:
        run_subprocess([
            os.environ.get("PYTHON", "python"),
            str(repo / "journal_extension/scripts/publish_tracka_v12_posttraining_state.py"),
            "--repo-root", str(repo),
            "--state-root", str(state_root),
            "--analysis-source-git-commit", analysis_sha,
            "--run-id", state["posttraining_public_run_id"],
            "--experiment-id", state["experiment_id"],
            "--output", str(publication_cert),
        ], cwd=repo)
    publication = load_json(publication_cert)
    if publication.get("status") != "PASS":
        raise RuntimeError(f"public-safe evidence publication failed: {state['experiment_id']}")
    if publication.get("publication_branch") != expected_publication_branch:
        raise RuntimeError(f"public-safe evidence branch mismatch: {state['experiment_id']}")
    return completion


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--account-id", choices=["K1", "K2", "K3"], required=True)
    ap.add_argument("--account-inventory", required=True)
    ap.add_argument("--account-readiness", required=True)
    ap.add_argument("--global-readiness", required=True)
    ap.add_argument("--placement-freeze", required=True)
    ap.add_argument("--campaign-lock", default="journal_extension/locks/track_a_posttraining_campaign_v1.json")
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    analysis_sha = args.analysis_source_git_commit
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if observed != analysis_sha:
        raise SystemExit(f"exact analysis checkout mismatch: expected={analysis_sha}, observed={observed}")
    campaign_path = Path(args.campaign_lock)
    if not campaign_path.is_absolute():
        campaign_path = repo / campaign_path
    campaign = load_json(campaign_path)
    inventory = load_json(args.account_inventory)
    readiness = load_json(args.account_readiness)
    global_readiness = load_json(args.global_readiness)
    placement = load_json(args.placement_freeze)
    if readiness.get("status") != "PASS" or readiness.get("account_id") != args.account_id:
        raise SystemExit("account readiness is not PASS for requested account")
    if readiness.get("analysis_source_git_commit") != analysis_sha:
        raise SystemExit("account readiness analysis-source mismatch")
    if (
        global_readiness.get("status") != "PASS"
        or global_readiness.get("gate_kind") != "track_a_v12_posttraining_global_readiness"
        or global_readiness.get("analysis_source_git_commit") != analysis_sha
        or global_readiness.get("ready_for_posttraining_evidence") is not True
    ):
        raise SystemExit("global readiness must be PASS before any post-training evidence execution")
    if placement.get("status") != "PASS" or placement.get("analysis_source_git_commit") != analysis_sha:
        raise SystemExit("placement freeze is not PASS for this analysis source")
    if placement.get("placement_metric_blind") is not True:
        raise SystemExit("placement freeze is not metric-blind")
    if inventory.get("account_id") != args.account_id:
        raise SystemExit("account inventory account mismatch")

    assigned = placement["account_queues"][args.account_id]
    states = inventory.get("states") or {}
    if set(assigned) != set(states):
        raise SystemExit("account inventory must equal exact frozen placement queue for this account")
    for experiment_id, spec in states.items():
        operator_errors = validate_state_operator_spec(experiment_id, spec, check_paths=True)
        if operator_errors:
            raise SystemExit(
                f"operator contract invalid for {experiment_id}: " + "; ".join(operator_errors)
            )
    if set(readiness.get("states", {})) != set(states):
        raise SystemExit("account readiness state inventory differs from operator inventory")

    budget = SessionBudget.establish_global_clock()
    budget.install_signal_handlers()
    stage_seconds = campaign["stage_start_guards_seconds"]
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()
    publication_lock = threading.Lock()

    slot_queues = {0: [], 1: []}
    for experiment_id in assigned:
        assignment = placement["assignments"][experiment_id]
        gpu_index = 0 if assignment["slot_id"].endswith("GPU0") else 1
        row = dict(states[experiment_id])
        row["experiment_id"] = experiment_id
        row["posttraining_public_run_id"] = (
            f"TRACKA-POST-{experiment_id.lower().replace('_','-')}-{analysis_sha[:12]}"
        )
        slot_queues[gpu_index].append(row)

    results = {}
    results_lock = threading.Lock()

    def worker(gpu_index: int):
        for state in slot_queues[gpu_index]:
            experiment_id = state["experiment_id"]
            state_root = work_root / experiment_id
            state_root.mkdir(parents=True, exist_ok=True)
            completion_path = state_root / "certificates" / "POSTTRAINING_STATE_COMPLETION.json"
            if completion_path.is_file():
                existing = load_json(completion_path)
                if (
                    existing.get("status") == "PASS"
                    and existing.get("experiment_id") == experiment_id
                    and existing.get("analysis_source_git_commit") == analysis_sha
                    and existing.get("private_generation_roundtrip_verified") is True
                    and existing.get("publication_branch")
                ):
                    with results_lock:
                        results[experiment_id] = {"status": "REUSED_COMPLETE", "completion": existing}
                    continue
            try:
                restore_state_evidence(
                    repo=repo,
                    state=state,
                    state_root=state_root,
                    analysis_sha=analysis_sha,
                )
            except Exception as exc:
                with results_lock:
                    results[experiment_id] = {
                        "status": "FAIL_RESTORE",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                return
            stages = ["direct", "xai"] if state["role"] == "direct" else ["auxiliary"]
            terminal_dirs = {}
            deferred = False
            for stage in stages:
                estimate_key = {"direct": "DIRECT_EVIDENCE", "xai": "XAI_EVIDENCE", "auxiliary": "AUXILIARY_EVIDENCE"}[stage]
                status, path = run_stage(
                    repo=repo,
                    state=state,
                    stage=stage,
                    state_root=state_root,
                    analysis_sha=analysis_sha,
                    gpu_index=gpu_index,
                    estimated_seconds=float(stage_seconds[estimate_key]),
                    budget=budget,
                    log_lock=log_lock,
                )
                if status == "DEFERRED_SESSION_BUDGET":
                    deferred = True
                    break
                if status == "FAILED" or path is None:
                    with results_lock:
                        results[experiment_id] = {"status": "FAIL", "stage": stage}
                    return
                terminal_dirs[stage] = path
                if status == "PASS":
                    try:
                        sync_partial_state(
                            repo=repo,
                            state=state,
                            state_root=state_root,
                            analysis_sha=analysis_sha,
                            stage=stage,
                        )
                    except Exception as exc:
                        with results_lock:
                            results[experiment_id] = {
                                "status": "FAIL_PARTIAL_DURABILITY",
                                "stage": stage,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        return
            if deferred:
                with results_lock:
                    results[experiment_id] = {"status": "DEFERRED_SESSION_BUDGET"}
                continue
            if not budget.can_start_phase(
                float(stage_seconds["PRIVATE_DURABILITY_SYNC"]) + float(stage_seconds["PUBLICATION"]),
                extra_reserve_seconds=300,
            ):
                with results_lock:
                    results[experiment_id] = {"status": "DEFERRED_FINALIZATION_BUDGET"}
                continue
            try:
                completion = finalize_state(
                    repo=repo,
                    state=state,
                    state_root=state_root,
                    stage_dirs=terminal_dirs,
                    analysis_sha=analysis_sha,
                    publication_lock=publication_lock,
                )
            except Exception as exc:
                with results_lock:
                    results[experiment_id] = {"status": "FAIL_FINALIZATION", "error": f"{type(exc).__name__}: {exc}"}
                return
            with results_lock:
                results[experiment_id] = {"status": "PASS", "completion": completion}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu) for gpu in (0, 1)]
        for future in futures:
            future.result()

    complete_statuses = {"PASS", "REUSED_COMPLETE"}
    all_complete = set(results) == set(states) and all(row["status"] in complete_statuses for row in results.values())
    manifest = {
        "schema_version": "1.0",
        "status": "PASS" if all_complete else "INCOMPLETE",
        "manifest_kind": "track_a_posttraining_account_execution",
        "account_id": args.account_id,
        "analysis_source_git_commit": analysis_sha,
        "campaign_lock_sha256": sha256_file(campaign_path),
        "account_inventory_sha256": sha256_file(args.account_inventory),
        "account_readiness_sha256": sha256_file(args.account_readiness),
        "global_readiness_sha256": sha256_file(args.global_readiness),
        "placement_freeze_sha256": sha256_file(args.placement_freeze),
        "assigned_state_count": len(states),
        "completed_state_count": sum(row["status"] in complete_statuses for row in results.values()),
        "states": results,
        "session": budget.snapshot(),
        "cross_gpu_gradient_synchronization": False,
        "scientific_scheduling_adaptive": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    atomic_write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all_complete else 20


if __name__ == "__main__":
    raise SystemExit(main())
