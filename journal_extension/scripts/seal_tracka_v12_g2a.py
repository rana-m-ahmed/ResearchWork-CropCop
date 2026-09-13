from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.checkpointing import CheckpointCorruptionError, save_torch_checkpoint, verify_selected
from cropcop_je.hashing import sha256_json
from cropcop_je.tracka_v12_g2a import (
    REQUIRED_PROFILES,
    build_g2a_barrier,
    build_scheduler_freeze,
    validate_g2a_barrier_object,
    validate_scheduler_freeze,
)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_contract_probe() -> dict:
    import torch

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        identity = {
            "experiment_id": "TRACKA-V12-G2A-CHECKPOINT-PROBE",
            "authority_id": "EAAI-JE-SDL-v2.1-QA",
            "source_git_commit": "0" * 40,
            "config_sha256": "1" * 64,
            "ctc_v2_sha256": "2" * 64,
            "manifest_sha256": "3" * 64,
            "class_map_sha256": "4" * 64,
            "seed": 21270083,
            "student_init_sha256": "5" * 64,
            "pretrained_sha256": "6" * 64,
            "teacher_sha256": None,
            "teacher_factory_sha256": None,
            "teacher_factory_bundle_sha256": None,
            "software_stack_sha256": "7" * 64,
            "dependency_lock_sha256": "8" * 64,
            "g1_seal_sha256": "9" * 64,
            "g2_barrier_sha256": None,
            "lane_id": "K1/GPU0",
        }
        payload = {
            "schema_version": "2.0",
            "identity": identity,
            "identity_sha256": sha256_json(identity),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "student": {"probe": torch.arange(8, dtype=torch.float32)},
            "projection": None,
            "optimizer": {},
            "scheduler": {},
            "scaler": {},
            "rng": {"python": None, "numpy": None, "torch": None, "cuda": None},
            "epoch": 1,
            "batch_in_epoch": 0,
            "optimizer_step": 1,
            "examples_seen": 1,
            "data_order_state": {"epoch": 1, "next_batch_in_epoch": 0},
            "selection_state": {"history": [], "best": None},
        }
        _latest, _ = save_torch_checkpoint(root, kind="latest", payload=payload, expected_identity=identity)
        selected, _ = save_torch_checkpoint(root, kind="selected", payload=payload, expected_identity=identity)
        verify_selected(root, expected_identity=identity, expected_sha256=selected.sha256)

        wrong = dict(identity)
        wrong["seed"] = identity["seed"] + 1
        mismatch_rejected = False
        try:
            verify_selected(root, expected_identity=wrong)
        except CheckpointCorruptionError:
            mismatch_rejected = True

        corrupt_root = root / "corrupt-copy"
        shutil.copytree(root, corrupt_root, ignore=shutil.ignore_patterns("corrupt-copy"))
        selected_path = corrupt_root / selected.relative_path
        data = bytearray(selected_path.read_bytes())
        if not data:
            raise RuntimeError("checkpoint probe selected file unexpectedly empty")
        data[len(data) // 2] ^= 0x01
        selected_path.write_bytes(bytes(data))
        corrupt_rejected = False
        try:
            verify_selected(corrupt_root, expected_identity=identity, expected_sha256=selected.sha256)
        except CheckpointCorruptionError:
            corrupt_rejected = True

        status = "PASS" if mismatch_rejected and corrupt_rejected else "FAIL"
        return {
            "status": status,
            "selected_checkpoint_verified": True,
            "identity_mismatch_rejected": mismatch_rejected,
            "corrupt_checkpoint_rejected": corrupt_rejected,
            "scientific_result_produced": False,
            "protected_data_accessed": False,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="append", required=True, help="Path to one G2A calibration summary; provide exactly five.")
    ap.add_argument("--barrier-out", required=True)
    ap.add_argument("--scheduler-out", required=True)
    args = ap.parse_args()

    summaries = [load_json(path) for path in args.summary]
    if len(summaries) != len(REQUIRED_PROFILES):
        raise SystemExit(f"expected exactly {len(REQUIRED_PROFILES)} G2A summaries")
    if {str(row.get("calibration_id")) for row in summaries} != set(REQUIRED_PROFILES):
        raise SystemExit("G2A summary set must contain each required calibration profile exactly once")

    probe = checkpoint_contract_probe()
    if probe["status"] != "PASS":
        raise SystemExit("checkpoint contract probe failed")
    barrier = build_g2a_barrier(summaries, checkpoint_contract_probe=probe)
    barrier_errors = validate_g2a_barrier_object(barrier)
    if barrier_errors:
        raise SystemExit("G2A barrier invalid: " + "; ".join(barrier_errors))
    scheduler = build_scheduler_freeze(barrier)
    scheduler_errors = validate_scheduler_freeze(
        scheduler,
        expected_g2a_barrier_sha256=barrier["barrier_sha256"],
    )
    if scheduler_errors:
        raise SystemExit("scheduler freeze invalid: " + "; ".join(scheduler_errors))

    atomic_write_json(args.barrier_out, barrier)
    atomic_write_json(args.scheduler_out, scheduler)
    print(json.dumps({"g2a": barrier, "scheduler": scheduler}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
