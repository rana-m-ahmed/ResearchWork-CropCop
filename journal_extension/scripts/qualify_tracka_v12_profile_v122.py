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
from cropcop_je.persistence import validate_durable_access_plan
from cropcop_je.persistence_v8 import GenerationAwareKagglePrivateDatasetStore, build_store_v8
from cropcop_je.tracka_v12_g1a import R13_PRETRAINED_SHA256, TEACHER_SHA256
from cropcop_je.tracka_v12_g2a import PROFILE_COVERAGE
from cropcop_je.tracka_v12_g2a_durability import validate_calibration_durability
from cropcop_je.tracka_v12_g2a_v122 import validate_calibration_summary_v122
from run_tracka_v12_training_v121 import execute, parser as training_parser

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


def _steady_seconds_per_step(result: dict) -> float:
    steps = int(result.get("optimizer_steps_segment", 0))
    if steps <= 0:
        raise RuntimeError("calibration segment has no optimizer steps")
    wall = float(result.get("wall_seconds_segment", 0.0) or 0.0)
    save = float(result.get("checkpoint_save_seconds", 0.0) or 0.0)
    load = float(result.get("checkpoint_load_seconds", 0.0) or 0.0)
    steady_total = wall - save - load
    if steady_total <= 0:
        raise RuntimeError("calibration steady-state duration is not positive after checkpoint subtraction")
    return steady_total / steps


def _validation_rate(result: dict) -> float:
    bench = result.get("validation_forward_benchmark") or {}
    value = float(bench.get("end_to_end_examples_per_second", 0.0) or 0.0)
    if value <= 0:
        raise RuntimeError("calibration validation forward benchmark is missing/invalid")
    return value


def _generation_proof(persistence: dict, *, label: str) -> dict:
    if persistence.get("backend") != "kaggle_private_dataset":
        raise RuntimeError(f"{label} G2A persistence backend is not Kaggle private dataset")
    if persistence.get("generation_roundtrip_verified") is not True:
        raise RuntimeError(f"{label} G2A durable sync did not prove generation-aware round-trip")
    try:
        previous = int(persistence["previous_version_number"])
        confirmed = int(persistence["confirmed_version_number"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} G2A durable sync lacks generation numbers") from exc
    if confirmed <= previous:
        raise RuntimeError(
            f"{label} G2A durable sync did not advance Kaggle generation: previous={previous}, confirmed={confirmed}"
        )
    marker_sha = str(persistence.get("generation_marker_sha256", ""))
    if len(marker_sha) != 64 or any(ch not in "0123456789abcdef" for ch in marker_sha.lower()):
        raise RuntimeError(f"{label} G2A durable sync marker SHA-256 is invalid")
    return {
        "previous_version_number": previous,
        "confirmed_version_number": confirmed,
        "generation_marker_sha256": marker_sha,
        "generation_roundtrip_verified": True,
    }


def main() -> int:
    ap = training_parser()
    ap.add_argument("--calibration-id", choices=tuple(REPRESENTATIVE_EXPERIMENT), required=True)
    ap.add_argument("--initial-steps", type=int, default=24)
    ap.add_argument("--resume-steps", type=int, default=8)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    if args.mode != "calibration":
        raise SystemExit("G2A v1.2.2 profile qualification requires --mode calibration")
    expected_experiment = REPRESENTATIVE_EXPERIMENT[args.calibration_id]
    if args.experiment_id != expected_experiment:
        raise SystemExit(
            f"{args.calibration_id} must use frozen representative {expected_experiment}, got {args.experiment_id}"
        )
    if args.initial_steps <= 0 or args.resume_steps <= 0:
        raise SystemExit("G2A v1.2.2 initial/resume steps must both be positive")
    if not args.durable_required or not args.durable_store_kind or not args.durable_store_locator:
        raise SystemExit("G2A v1.2.2 qualification requires durable storage and --durable-required")
    if args.durable_store_kind != "kaggle-dataset":
        raise SystemExit("G2A v1.2.2 qualification requires --durable-store-kind kaggle-dataset")
    if git_credentials_present():
        raise SystemExit("G2A child environment contains Git/publication credentials; launch with sanitized environment")

    durable_preflight = validate_durable_access_plan(
        "kaggle-dataset",
        {args.run_id: args.durable_store_locator},
    )
    if durable_preflight.get("status") != "PASS":
        raise SystemExit(
            "G2A private Kaggle durability preflight failed: "
            + "; ".join(str(error) for error in durable_preflight.get("errors", []))
        )

    first = execute(args, mode="calibration", max_optimizer_steps=args.initial_steps, resume_mode="never")
    if first.get("status") != "PASS":
        raise SystemExit("initial G2A v1.2.2 calibration segment did not PASS")
    first_result = first.get("result_summary", {})
    if first.get("validation_enabled") is not False or first.get("scientific_metric_computed") is not False:
        raise SystemExit("initial G2A v1.2.2 segment violated non-scientific boundary")
    first_generation = _generation_proof(first.get("persistence_status") or {}, label="initial")

    output = Path(args.output_dir)
    checkpoints = output / "private_checkpoints"
    if not checkpoints.exists():
        raise SystemExit("initial G2A v1.2.2 segment did not produce checkpoint state")
    store = build_store_v8(args.durable_store_kind, args.durable_store_locator)
    if not isinstance(store, GenerationAwareKagglePrivateDatasetStore):
        raise SystemExit("G2A v1.2.2 destructive restore is not bound to generation-aware Kaggle durability")
    shutil.rmtree(checkpoints)
    restore_started = time.perf_counter()
    restored = store.restore(checkpoints, run_id=args.run_id)
    restore_seconds = time.perf_counter() - restore_started
    if restored is not True or not (checkpoints / "checkpoint_index.json").exists():
        raise SystemExit("G2A v1.2.2 durable restore did not reconstruct checkpoint state")

    second = execute(args, mode="calibration", max_optimizer_steps=args.resume_steps, resume_mode="required")
    if second.get("status") != "PASS":
        raise SystemExit("resumed G2A v1.2.2 calibration segment did not PASS")
    second_result = second.get("result_summary", {})
    if int(second_result.get("optimizer_step_total", 0)) <= int(first_result.get("optimizer_step_total", 0)):
        raise SystemExit("G2A v1.2.2 resumed segment did not advance optimizer state")
    if float(second_result.get("checkpoint_load_seconds", 0.0) or 0.0) <= 0:
        raise SystemExit("G2A v1.2.2 resumed segment did not demonstrate checkpoint load")
    persistence = second.get("persistence_status") or {}
    second_generation = _generation_proof(persistence, label="resumed")

    steady_costs = [_steady_seconds_per_step(first_result), _steady_seconds_per_step(second_result)]
    validation_rates = [_validation_rate(first_result), _validation_rate(second_result)]
    checkpoint_saves = [
        float(first_result.get("checkpoint_save_seconds", 0.0) or 0.0),
        float(second_result.get("checkpoint_save_seconds", 0.0) or 0.0),
    ]
    if any(value <= 0 for value in checkpoint_saves):
        raise SystemExit("G2A v1.2.2 checkpoint save timing missing")
    syncs = [
        float(first_result.get("durable_sync_seconds", 0.0) or 0.0),
        float(second_result.get("durable_sync_seconds", 0.0) or 0.0),
    ]
    if any(value <= 0 for value in syncs):
        raise SystemExit("G2A v1.2.2 durable sync timing missing")

    summary = {
        "schema_version": "1.2.2",
        "calibration_id": args.calibration_id,
        "representative_experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "coverage": sorted(PROFILE_COVERAGE[args.calibration_id]),
        "status": "PASS",
        "calibration_weights_scientific": False,
        "validation_enabled": False,
        "scientific_metric_computed": False,
        "resume_success": True,
        "durable_roundtrip_success": True,
        "durable_store_kind": args.durable_store_kind,
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
            "sec_per_optimizer_step": statistics.median(
                [float(first_result["sec_per_optimizer_step"]), float(second_result["sec_per_optimizer_step"])]
            ),
            "steady_sec_per_optimizer_step": statistics.median(steady_costs),
            "examples_per_second": statistics.median(
                [float(first_result["examples_per_second"]), float(second_result["examples_per_second"])]
            ),
            "validation_end_to_end_examples_per_second": statistics.median(validation_rates),
            "checkpoint_save_seconds_per_event": statistics.median(checkpoint_saves),
            "durable_sync_seconds_per_segment": statistics.median(syncs),
            "peak_gpu_memory_bytes": max(
                int(first_result["peak_gpu_memory_bytes"]), int(second_result["peak_gpu_memory_bytes"])
            ),
            "checkpoint_save_seconds": sum(checkpoint_saves),
            "checkpoint_load_seconds": float(second_result.get("checkpoint_load_seconds", 0.0) or 0.0),
            "initial_segment_optimizer_steps": int(first_result["optimizer_steps_segment"]),
            "resumed_segment_optimizer_steps": int(second_result["optimizer_steps_segment"]),
        },
        "durability": {
            "sync_status": persistence.get("status"),
            "restore_status": "PASS",
            "restore_contract": "generation_aware_kaggle_v8",
            "restore_store_class": type(store).__name__,
            "sync_seconds": float(second_result.get("durable_sync_seconds", 0.0) or 0.0),
            "restore_seconds": restore_seconds,
            "backend": persistence.get("backend"),
            "locator": persistence.get("locator"),
            "initial_sync_generation": first_generation,
            "resumed_sync_generation": second_generation,
            "preflight_private_access": durable_preflight,
        },
        "physical_slot_id": second.get("physical_slot_id"),
        "logical_lane_id": second.get("lane_id"),
        "first_segment_result_sha256": first.get("last_segment_result_sha256"),
        "second_segment_result_sha256": second.get("last_segment_result_sha256"),
    }
    if args.calibration_id.startswith("CAL-MNV4-"):
        summary["teacher_checkpoint_sha256"] = TEACHER_SHA256
    if args.calibration_id == "CAL-R13":
        summary["r13_pretrained_sha256"] = R13_PRETRAINED_SHA256

    errors = validate_calibration_summary_v122(summary) + validate_calibration_durability(summary)
    if errors:
        raise SystemExit("G2A v1.2.2 profile self-validation failed: " + "; ".join(errors))
    atomic_write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
