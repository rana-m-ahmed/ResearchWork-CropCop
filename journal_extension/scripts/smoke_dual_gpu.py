from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.envelope import (
    AMENDMENT_ID,
    AMENDMENT_SHA256,
    EnvelopeError,
    child_environment,
    close_child_log,
    gpu_inventory,
    launch_process,
    require_parent_batch,
    terminate_process_group,
    validate_t4x2_inventory,
)
from cropcop_je.hashing import sha256_json
from cropcop_je.publication import publish_to_github_branch
from cropcop_je.science_diff import validate_science_diff
from cropcop_je.smoke_handoff import (
    validate_terminal_dual_gpu_smoke_evidence,
    validate_terminal_smoke_b_evidence,
)
from cropcop_je.source_state import verify_clean_source

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCY_LOCK = ROOT / "journal_extension/locks/execution_dependency_lock.json"
STEPS = 4
SEED = 876421


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _child(args) -> int:
    if os.environ.get("CROPCOP_DUAL_ENVELOPE") != "1":
        raise EnvelopeError("dual smoke child requires CROPCOP_DUAL_ENVELOPE=1")
    if os.environ.get("CROPCOP_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN"):
        raise EnvelopeError("dual smoke child must not receive Git credentials")

    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EnvelopeError("dual smoke child must see exactly one CUDA device")

    random.seed(SEED + int(args.slot))
    np.random.seed((SEED + int(args.slot)) % (2**32))
    torch.manual_seed(SEED + int(args.slot))
    torch.cuda.manual_seed_all(SEED + int(args.slot))

    out = Path(args.child_output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started_mono = time.monotonic()
    device = torch.device("cuda")
    model = torch.nn.Linear(32, 8).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    for _ in range(STEPS):
        x = torch.randn(16, 32, device=device)
        y = torch.randint(0, 8, (16,), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    checkpoint = out / "synthetic_state.pt"
    torch.save(
        {
            "schema_version": "1.0",
            "scientific": False,
            "synthetic": True,
            "optimizer_step": STEPS,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint,
    )
    evidence = {
        "schema_version": "1.0",
        "status": "PASS",
        "scientific": False,
        "synthetic_unprotected_data_only": True,
        "restricted_cropcop_data_accessed": False,
        "g1_executed": False,
        "g2_executed": False,
        "r04_r05_executed": False,
        "child_id": args.child_id,
        "requested_physical_slot": int(args.slot),
        "visible_cuda_device_count": torch.cuda.device_count(),
        "visible_gpu_name": torch.cuda.get_device_name(0),
        "optimizer_step": STEPS,
        "checkpoint_basename": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "notebook_started_monotonic": float(os.environ["CROPCOP_NOTEBOOK_STARTED_MONOTONIC"]),
        "started_monotonic": started_mono,
        "ended_monotonic": time.monotonic(),
        "created_at_utc": utc_now(),
    }
    atomic_write_json(out / "CHILD_EVIDENCE.json", evidence)
    return 0


def _parent(args) -> int:
    source_sha = os.environ.get("CROPCOP_SOURCE_GIT_COMMIT", "").strip()
    if len(source_sha) != 40:
        raise EnvelopeError("CROPCOP_SOURCE_GIT_COMMIT must name the frozen execution source")
    run_type = require_parent_batch("dual-gpu-smoke")
    dependency = json.loads(DEPENDENCY_LOCK.read_text(encoding="utf-8"))
    smoke_path = os.environ.get("CROPCOP_INFRA_SMOKE_EVIDENCE", "").strip()
    if not smoke_path:
        raise EnvelopeError("dual-gpu-smoke requires terminal Smoke-B evidence")
    smoke = json.loads(Path(smoke_path).read_text(encoding="utf-8"))
    smoke_errors = validate_terminal_smoke_b_evidence(
        smoke,
        expected_source_sha=source_sha,
        expected_dependency_lock_sha256=dependency["dependency_lock_sha256"],
        require_batch=True,
    )
    if smoke_errors:
        raise EnvelopeError("dual-gpu-smoke preflight Smoke-B invalid: " + "; ".join(smoke_errors))
    smoke_b_evidence_sha256 = sha256_json(smoke)

    output_root = Path(
        args.output_root
        or os.environ.get(
            "CROPCOP_DUAL_SMOKE_OUTPUT_ROOT",
            str(Path(os.environ["CROPCOP_OUTPUT_ROOT"]) / "dual-gpu-smoke"),
        )
    ).resolve()
    if output_root.exists():
        if any(output_root.iterdir()):
            raise EnvelopeError(f"dual-gpu-smoke output root must start empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    verify_clean_source(ROOT, authorized_source_sha=source_sha, output_roots=[output_root])
    science = validate_science_diff(ROOT)
    if science["status"] != "PASS":
        raise EnvelopeError("science-diff sentinel failed before dual smoke: " + "; ".join(science["errors"]))

    inventory = gpu_inventory()
    hw_errors = validate_t4x2_inventory(inventory)
    if hw_errors:
        raise EnvelopeError("dual-gpu-smoke T4X2 preflight failed: " + "; ".join(hw_errors))

    children = []
    for slot, child_id in ((0, "DUAL-SMOKE-A"), (1, "DUAL-SMOKE-B")):
        child_root = output_root / child_id
        env = child_environment(
            base_env=dict(os.environ),
            envelope_id="MGPU-DUAL-SMOKE-T4X2-V1",
            physical_slot=slot,
            output_root=child_root / "unused-output-root",
            g2_summaries_root=child_root / "unused-g2-root",
            terminal_root=child_root / "unused-terminal-root",
            workers=0,
        )
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--child-id",
            child_id,
            "--slot",
            str(slot),
            "--child-output",
            str(child_root),
        ]
        obj = launch_process(
            child_id=child_id,
            experiment_id="INFRA-DUAL-GPU-SMOKE",
            run_id=child_id,
            slot=slot,
            cmd=cmd,
            env=env,
            cwd=ROOT,
            log_path=output_root / f"{child_id}.log",
            timeout_seconds=300,
        )
        children.append(obj)

    failed = []
    deadline = time.monotonic() + 360
    try:
        while any(obj.process.poll() is None for obj in children):
            if time.monotonic() > deadline:
                for obj in children:
                    if obj.process.poll() is None:
                        terminate_process_group(obj, grace_seconds=10)
                raise EnvelopeError("dual-gpu-smoke child timeout")
            time.sleep(0.2)
    except BaseException:
        for obj in children:
            if obj.process.poll() is None:
                terminate_process_group(obj, grace_seconds=10)
        raise
    finally:
        for obj in children:
            close_child_log(obj)

    rows = []
    for obj in children:
        if obj.process.returncode != 0:
            failed.append(f"{obj.child_id}: rc={obj.process.returncode}")
            continue
        path = output_root / obj.child_id / "CHILD_EVIDENCE.json"
        if not path.is_file():
            failed.append(f"{obj.child_id}: child evidence missing")
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = output_root / obj.child_id / row["checkpoint_basename"]
        if row.get("status") != "PASS":
            failed.append(f"{obj.child_id}: status not PASS")
        if row.get("visible_cuda_device_count") != 1:
            failed.append(f"{obj.child_id}: visible CUDA count != 1")
        if row.get("visible_gpu_name") not in {"Tesla T4", "NVIDIA T4"}:
            failed.append(f"{obj.child_id}: visible GPU is not T4")
        if int(row.get("optimizer_step", 0)) <= 0:
            failed.append(f"{obj.child_id}: optimizer did not advance")
        if not checkpoint.is_file() or sha256_file(checkpoint) != row.get("checkpoint_sha256"):
            failed.append(f"{obj.child_id}: checkpoint verification failed")
        if float(row.get("notebook_started_monotonic", -1)) != float(os.environ["CROPCOP_NOTEBOOK_STARTED_MONOTONIC"]):
            failed.append(f"{obj.child_id}: common notebook clock mismatch")
        rows.append(row)

    if len(rows) == 2:
        overlap = max(
            0.0,
            min(float(rows[0]["ended_monotonic"]), float(rows[1]["ended_monotonic"]))
            - max(float(rows[0]["started_monotonic"]), float(rows[1]["started_monotonic"])),
        )
        if overlap <= 0:
            failed.append("dual smoke children did not overlap")
    else:
        overlap = 0.0

    if inventory[0]["uuid"] == inventory[1]["uuid"]:
        failed.append("physical GPU UUIDs are not distinct")

    evidence = {
        "schema_version": "1.0",
        "status": "PASS" if not failed else "FAIL",
        "qualification_id": "MGPU-DUAL-SMOKE-T4X2-V1",
        "scientific": False,
        "synthetic_unprotected_data_only": True,
        "source_git_sha": source_sha,
        "dependency_lock_sha256": dependency["dependency_lock_sha256"],
        "amendment_id": AMENDMENT_ID,
        "amendment_sha256": AMENDMENT_SHA256,
        "kaggle_run_type": run_type,
        "smoke_b_evidence_sha256": smoke_b_evidence_sha256,
        "notebook_started_monotonic": float(os.environ["CROPCOP_NOTEBOOK_STARTED_MONOTONIC"]),
        "parent_gpu_inventory": inventory,
        "children": [
            {
                "child_id": row.get("child_id"),
                "requested_physical_slot": row.get("requested_physical_slot"),
                "physical_gpu_uuid": inventory[int(row.get("requested_physical_slot", 0))]["uuid"],
                "visible_cuda_device_count": row.get("visible_cuda_device_count"),
                "visible_gpu_name": row.get("visible_gpu_name"),
                "optimizer_step": row.get("optimizer_step"),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "checkpoint_bytes": row.get("checkpoint_bytes"),
                "notebook_started_monotonic": row.get("notebook_started_monotonic"),
                "git_credentials_present_in_child": False,
            }
            for row in rows
        ],
        "overlap_duration_seconds": overlap,
        "no_output_collision": len({row.get("checkpoint_sha256") for row in rows}) == len(rows),
        "no_git_child_publication": True,
        "common_session_clock": all(
            float(row.get("notebook_started_monotonic", -1))
            == float(os.environ["CROPCOP_NOTEBOOK_STARTED_MONOTONIC"])
            for row in rows
        ),
        "parent_finalized_both": len(rows) == 2 and not failed,
        "science_diff_status": science["status"],
        "restricted_cropcop_data_accessed": False,
        "g1_executed": False,
        "g2_executed": False,
        "r04_r05_executed": False,
        "errors": failed,
        "created_at_utc": utc_now(),
    }
    evidence_path = output_root / "DUAL_GPU_SMOKE_EVIDENCE.json"
    atomic_write_json(evidence_path, evidence)
    if failed:
        raise EnvelopeError("dual-gpu-smoke failed: " + "; ".join(failed))

    branch = publish_to_github_branch(
        repo_dir=ROOT,
        source_git_sha=source_sha,
        run_id="DUAL-GPU-SMOKE",
        files=[evidence_path],
    )
    evidence["git_publication_status"] = "PASS"
    evidence["public_safe_evidence_branch"] = branch
    terminal_errors = validate_terminal_dual_gpu_smoke_evidence(
        evidence,
        expected_source_sha=source_sha,
        expected_dependency_lock_sha256=dependency["dependency_lock_sha256"],
        expected_amendment_id=AMENDMENT_ID,
        expected_amendment_sha256=AMENDMENT_SHA256,
        expected_smoke_b_evidence_sha256=smoke_b_evidence_sha256,
        require_batch=True,
    )
    if terminal_errors:
        raise EnvelopeError("dual-gpu-smoke terminal evidence invalid: " + "; ".join(terminal_errors))
    atomic_write_json(evidence_path, evidence)
    branch2 = publish_to_github_branch(
        repo_dir=ROOT,
        source_git_sha=source_sha,
        run_id="DUAL-GPU-SMOKE",
        files=[evidence_path],
    )
    if branch2 != branch:
        raise EnvelopeError("dual-gpu-smoke evidence branch changed unexpectedly")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--child-id", default="")
    ap.add_argument("--slot", type=int, default=-1)
    ap.add_argument("--child-output", default="")
    ap.add_argument("--output-root", default="")
    args = ap.parse_args()
    if args.child:
        if args.slot not in {0, 1} or not args.child_id or not args.child_output:
            raise SystemExit("dual smoke child arguments incomplete")
        return _child(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
