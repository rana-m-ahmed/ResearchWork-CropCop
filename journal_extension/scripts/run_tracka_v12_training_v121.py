from __future__ import annotations

import copy
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je import tracka_v12_g1a_v121 as g1a_v121
from cropcop_je import tracka_v12_runtime as tracka_runtime
from cropcop_je.persistence_v8 import build_store_v8
from cropcop_je.runlog import write_run_record
from cropcop_je.segments import append_segment_event, utc_now
from cropcop_je.tracka_v12_g2a import (
    validate_g2a_barrier_object as validate_g2a_v121,
    validate_scheduler_freeze as validate_scheduler_v121,
)
from cropcop_je.tracka_v12_g2a_v122 import (
    validate_g2a_v122_barrier,
    validate_scheduler_freeze_v122,
)
from cropcop_je.tracka_v12_placement import SLOT_IDS, logical_lane_id
import run_tracka_v12_training as base


# Track-A v1.2.1 uses a generation-aware Kaggle durable store. This changes only
# persistence/restore transport semantics; frozen training/scientific semantics remain in base.
base.build_store = build_store_v8

# R13 parity-contract v1.2.1 supersedes only the pre-science numerical acceptance
# bound. The runtime bundle validator and R13 initialization loader must use the
# versioned contract identity/tolerance rather than historical v1.2 defaults.
tracka_runtime.validate_g1a_seal_object = g1a_v121.validate_g1a_seal_object
tracka_runtime.load_r13_initialization = g1a_v121.load_r13_initialization


def validate_g2a_compatible(
    barrier: dict,
    *,
    expected_source_sha: str | None = None,
    expected_g1a_seal_sha256: str | None = None,
) -> list[str]:
    if str(barrier.get("schema_version")) == "1.2.2":
        return validate_g2a_v122_barrier(
            barrier,
            expected_source_sha=expected_source_sha,
            expected_g1a_seal_sha256=expected_g1a_seal_sha256,
        )
    return validate_g2a_v121(
        barrier,
        expected_source_sha=expected_source_sha,
        expected_g1a_seal_sha256=expected_g1a_seal_sha256,
    )


def validate_scheduler_compatible(
    scheduler: dict,
    *,
    expected_g2a_barrier_sha256: str | None = None,
) -> list[str]:
    if str(scheduler.get("schema_version")) == "1.2.2":
        return validate_scheduler_freeze_v122(
            scheduler,
            expected_g2a_barrier_sha256=expected_g2a_barrier_sha256,
        )
    return validate_scheduler_v121(
        scheduler,
        expected_g2a_barrier_sha256=expected_g2a_barrier_sha256,
    )


base.validate_g2a_barrier_object = validate_g2a_compatible
base.validate_scheduler_freeze = validate_scheduler_compatible


def annotate_physical_placement(
    output_dir: str | Path,
    *,
    physical_slot_id: str,
    logical_lane: str,
) -> dict | None:
    if physical_slot_id not in SLOT_IDS:
        raise ValueError(f"unqualified physical slot: {physical_slot_id}")
    root = Path(output_dir)
    record_path = root / "run_record.json"
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("lane_id") != logical_lane:
        raise RuntimeError("run record logical lane differs from placement-neutral execution identity")
    record["physical_slot_id"] = physical_slot_id
    hardware = dict(record.get("hardware_identity") or {})
    hardware["physical_slot_id"] = physical_slot_id
    hardware["logical_lane_id"] = logical_lane
    hardware.pop("slot_id", None)
    record["hardware_identity"] = hardware
    write_run_record(record_path, record)
    append_segment_event(
        root / "segments.jsonl",
        {
            "segment_id": record.get("segment_id"),
            "parent_run_id": record.get("run_id"),
            "state": "PHYSICAL_PLACEMENT",
            "timestamp_utc": utc_now(),
            "logical_lane_id": logical_lane,
            "physical_slot_id": physical_slot_id,
            "placement_is_scientific_identity": False,
        },
    )
    return record


def execute(args, *, mode: str, max_optimizer_steps: int | None = None, resume_mode: str = "auto") -> dict:
    physical_slot = str(args.slot_id)
    if physical_slot not in SLOT_IDS:
        raise ValueError(f"unqualified physical slot: {physical_slot}")
    logical_lane = logical_lane_id(str(args.experiment_id))
    child_args = copy.copy(args)
    child_args.slot_id = logical_lane
    try:
        record = base.execute(
            child_args,
            mode=mode,
            max_optimizer_steps=max_optimizer_steps,
            resume_mode=resume_mode,
        )
    except BaseException:
        annotate_physical_placement(
            args.output_dir,
            physical_slot_id=physical_slot,
            logical_lane=logical_lane,
        )
        raise
    annotated = annotate_physical_placement(
        args.output_dir,
        physical_slot_id=physical_slot,
        logical_lane=logical_lane,
    )
    return annotated or record


def parser():
    return base.parser()


def main() -> int:
    args = parser().parse_args()
    if args.mode == "scientific" and args.max_optimizer_steps:
        raise SystemExit("scientific mode must run the full frozen 30 epochs")
    if args.mode == "calibration" and args.max_optimizer_steps <= 0:
        raise SystemExit("calibration mode requires --max-optimizer-steps")
    record = execute(
        args,
        mode=args.mode,
        max_optimizer_steps=(args.max_optimizer_steps or None),
        resume_mode=args.resume_mode,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
