from __future__ import annotations

import copy
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.runlog import write_run_record
from cropcop_je.segments import append_segment_event, utc_now
from cropcop_je.tracka_v12_placement import SLOT_IDS, logical_lane_id
import run_tracka_v12_training as base


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
