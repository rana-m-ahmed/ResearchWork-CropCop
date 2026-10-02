from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.persistence import build_store
from cropcop_je.tracka_v12_g1a import R13_PRETRAINED_SHA256, TEACHER_SHA256
from cropcop_je.tracka_v12_g2a import PROFILE_COVERAGE, validate_calibration_summary
from run_tracka_v12_training import execute, parser as training_parser

REPRESENTATIVE_EXPERIMENT = {
    "CAL-EFFB0": "R06-EFFB0-CONTEXT-S2",
    "CAL-CNXTT": "R07-CNXTT-CONTEXT-S2",
    "CAL-MNV4-LOGITS": "R12-MNV4-LOGITS-S2",
    "CAL-MNV4-FEATURE": "R12-MNV4-FEATURE-S2",
    "CAL-R13": "R13-VIT-DLITTLE-DIFF-CONTEXT-S1",
}
GIT_CREDENTIAL_ENV_NAMES = (
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "CROPCOP_GITHUB_TOKEN",
    "GIT_ASKPASS",
    "SSH_AUTH_SOCK",
)


def git_credentials_present() -> bool:
    return any(bool(os.environ.get(name)) for name in GIT_CREDENTIAL_ENV_NAMES)


def main() -> int:
    ap = training_parser()
    ap.add_argument("--calibration-id", choices=tuple(REPRESENTATIVE_EXPERIMENT), required=True)
    ap.add_argument("--initial-steps", type=int, default=12)
    ap.add_argument("--resume-steps", type=int, default=4)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    if args.mode != "calibration":
        raise SystemExit("G2A profile qualification requires --mode calibration")
    expected_experiment = REPRESENTATIVE_EXPERIMENT[args.calibration_id]
    if args.experiment_id != expected_experiment:
        raise SystemExit(
            f"{args.calibration_id} must use frozen representative {expected_experiment}, got {args.experiment_id}"
        )
    if args.initial_steps <= 0 or args.resume_steps <= 0:
        raise SystemExit("G2A initial/resume steps must both be positive")
    if not args.durable_required or not args.durable_store_kind or not args.durable_store_locator:
        raise SystemExit("G2A qualification requires a configured durable store and --durable-required")
    if git_credentials_present():
        raise SystemExit("G2A scientific child environment contains Git/publication credentials; launch with sanitized environment")

    args.max_optimizer_steps = args.initial_steps
    first = execute(args, mode="calibration", max_optimizer_steps=args.initial_steps, resume_mode="never")
    if first.get("status") != "PASS":
        raise SystemExit("initial G2A calibration segment did not PASS")
    first_result = first.get("result_summary", {})
    if first.get("validation_enabled") is not False or first.get("scientific_metric_computed") is not False:
        raise SystemExit("initial G2A segment violated non-scientific calibration boundary")

    output = Path(args.output_dir)
    checkpoints = output / "private_checkpoints"
    if not checkpoints.exists():
        raise SystemExit("initial G2A segment did not produce checkpoint state")
    store = build_store(args.durable_store_kind, args.durable_store_locator)
    shutil.rmtree(checkpoints)
    restore_started = time.perf_counter()
    restored = store.restore(checkpoints, run_id=args.run_id)
    restore_seconds = time.perf_counter() - restore_started
    if restored is not True or not (checkpoints / "checkpoint_index.json").exists():
        raise SystemExit("G2A durable restore did not reconstruct checkpoint state")

    args.max_optimizer_steps = args.resume_steps
    second = execute(args, mode="calibration", max_optimizer_steps=args.resume_steps, resume_mode="required")
    if second.get("status") != "PASS":
        raise SystemExit("resumed G2A calibration segment did not PASS")
    second_result = second.get("result_summary", {})
    if int(second_result.get("optimizer_step_total", 0)) <= int(first_result.get("optimizer_step_total", 0)):
        raise SystemExit("G2A resumed segment did not advance optimizer state")
    if float(second_result.get("checkpoint_load_seconds", 0.0) or 0.0) <= 0:
        raise SystemExit("G2A resumed segment did not demonstrate checkpoint load")

    costs = [float(first_result["sec_per_optimizer_step"]), float(second_result["sec_per_optimizer_step"])]
    throughputs = [float(first_result["examples_per_second"]), float(second_result["examples_per_second"])]
    peak = max(int(first_result["peak_gpu_memory_bytes"]), int(second_result["peak_gpu_memory_bytes"]))
    checkpoint_save = float(first_result.get("checkpoint_save_seconds", 0.0)) + float(second_result.get("checkpoint_save_seconds", 0.0))
    checkpoint_load = float(second_result.get("checkpoint_load_seconds", 0.0))
    persistence = second.get("persistence_status") or {}

    summary = {
        "schema_version": "1.0",
        "calibration_id": args.calibration_id,
        "representative_experiment_id": args.experiment_id,
        "coverage": sorted(PROFILE_COVERAGE[args.calibration_id]),
        "status": "PASS",
        "calibration_weights_scientific": False,
        "validation_enabled": False,
        "scientific_metric_computed": False,
        "resume_success": True,
        "durable_roundtrip_success": True,
        "visible_cuda_device_count": int(second.get("hardware_identity", {}).get("visible_cuda_device_count", 0)),
        "visible_gpu_name": second.get("hardware_identity", {}).get("accelerator"),
        "git_credentials_present": False,
        "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"],
        "v1_test_accessed": False,
        "external_protected_surface_accessed": False,
        "source_git_commit": second["source_git_commit"],
        "software_stack_sha256": second["software_stack_sha256"],
        "g1a_seal_sha256": second["tracka_g1a_seal_sha256"],
        "dependency_lock_sha256": second["dependency_lock_sha256"],
        "measured": {
            "sec_per_optimizer_step": statistics.median(costs),
            "examples_per_second": statistics.median(throughputs),
            "peak_gpu_memory_bytes": peak,
            "checkpoint_save_seconds": checkpoint_save,
            "checkpoint_load_seconds": checkpoint_load,
            "initial_segment_optimizer_steps": int(first_result["optimizer_steps_segment"]),
            "resumed_segment_optimizer_steps": int(second_result["optimizer_steps_segment"]),
        },
        "durability": {
            "sync_status": persistence.get("status"),
            "restore_status": "PASS",
            "sync_seconds": float(second_result.get("durable_sync_seconds", 0.0) or 0.0),
            "restore_seconds": restore_seconds,
            "backend": persistence.get("backend"),
            "locator": persistence.get("locator"),
        },
        "first_segment_result_sha256": first.get("last_segment_result_sha256"),
        "second_segment_result_sha256": second.get("last_segment_result_sha256"),
    }
    if args.calibration_id.startswith("CAL-MNV4-"):
        summary["teacher_checkpoint_sha256"] = TEACHER_SHA256
    if args.calibration_id == "CAL-R13":
        summary["r13_pretrained_sha256"] = R13_PRETRAINED_SHA256

    errors = validate_calibration_summary(summary)
    if errors:
        raise SystemExit("G2A profile self-validation failed: " + "; ".join(errors))
    atomic_write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
