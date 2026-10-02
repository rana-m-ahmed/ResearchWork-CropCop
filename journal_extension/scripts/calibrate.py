from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import run_training as rt
from cropcop_je.atomic_io import atomic_write_json

LOCKED_STEPS = {"direct": 200, "teacher": 100}
QUALIFICATION_STEPS = 10
QUALIFICATION_RESUME_STEPS = 5


def _clone_args(args):
    return argparse.Namespace(**vars(args))


def main() -> int:
    ap = rt.parser()
    ap.description = "Run scheduling-only MobileNetV4 calibration plus an independent save→resume qualification."
    ap.add_argument("--condition", choices=["direct", "teacher"], required=True)
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config["condition"] != args.condition:
        raise SystemExit(f"config condition={config['condition']} does not match --condition={args.condition}")
    calibration_id = "CAL-MNV4-DIRECT" if args.condition == "direct" else "CAL-MNV4-TEACHER"
    target_steps = LOCKED_STEPS[args.condition]
    base = Path(args.output_dir)

    q1 = _clone_args(args)
    q1.run_id = f"{calibration_id}-RESUME-QUAL"
    q1.output_dir = str(base / "resume_qualification")
    q1.resume_mode = "never"
    first = rt.execute(q1, max_optimizer_steps=QUALIFICATION_STEPS, resume_mode="never", mode="calibration")
    if first["status"] != "PASS":
        raise SystemExit("resume qualification first segment did not complete")

    q2 = _clone_args(q1)
    q2.resume_mode = "required"
    second = rt.execute(q2, max_optimizer_steps=QUALIFICATION_RESUME_STEPS, resume_mode="required", mode="calibration")
    resume_success = (
        second["status"] == "PASS"
        and second.get("result_summary", {}).get("optimizer_steps_segment") == QUALIFICATION_RESUME_STEPS
        and second.get("result_summary", {}).get("checkpoint_load_seconds", 0) > 0
    )
    if not resume_success:
        raise SystemExit("save→resume qualification did not advance the requested resumed optimizer steps")

    measured_args = _clone_args(args)
    measured_args.run_id = calibration_id
    measured_args.output_dir = str(base / "measured_calibration")
    measured_args.resume_mode = "never"
    measured = rt.execute(measured_args, max_optimizer_steps=target_steps, resume_mode="never", mode="calibration")
    if measured["status"] != "PASS":
        raise SystemExit("measured calibration did not complete")

    m = dict(measured["result_summary"])
    m["checkpoint_load_seconds"] = second["result_summary"]["checkpoint_load_seconds"]
    m["accelerator"] = measured.get("hardware_identity", {}).get("accelerator")
    m["cuda_driver_identity"] = measured.get("environment", {}).get("nvidia_smi")
    summary = {
        "schema_version": "2.0",
        "status": "PASS",
        "calibration_id": calibration_id,
        "condition": args.condition,
        "locked_optimizer_steps": target_steps,
        "resume_qualification": {
            "fresh_steps": QUALIFICATION_STEPS,
            "resumed_steps": QUALIFICATION_RESUME_STEPS,
            "resume_success": True,
            "checkpoint_load_seconds": second["result_summary"]["checkpoint_load_seconds"],
        },
        "resume_success": True,
        "calibration_weights_scientific": False,
        "source_git_commit": measured["source_git_commit"],
        "software_stack_sha256": measured["software_stack_sha256"],
        "g1_seal_sha256": measured["g1_seal_sha256"],
        "dependency_lock_sha256": measured["dependency_lock_sha256"],
        "mnv4_pretrained_sha256": measured["pretrained_sha256"],
        "teacher_checkpoint_sha256": measured.get("teacher_sha256"),
        "teacher_factory_bundle_sha256": measured.get("teacher_factory_bundle_sha256"),
        "measured": m,
        "observed_kaggle_constraints": {
            "configured_hard_session_seconds": args.session_hard_limit_seconds,
            "configured_finalization_margin_seconds": args.finalization_margin_seconds,
            "quota_visibility": "not_exposed_programmatically_to_runner",
        },
    }
    out = base / "calibration_summary.json"
    atomic_write_json(out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
