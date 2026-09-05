from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import run_training as rt

LOCKED_STEPS = {"direct": 200, "teacher": 100}


def main() -> int:
    ap = rt.parser()
    ap.description = "Run Stage-03R scheduling-only calibration with a save→resume qualification."
    ap.add_argument("--condition", choices=["direct", "teacher"], required=True)
    args = ap.parse_args()

    target_steps = LOCKED_STEPS[args.condition]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config["condition"] != args.condition:
        raise SystemExit(f"config condition={config['condition']} does not match --condition={args.condition}")

    base = Path(args.output_dir)
    qualification_dir = base / "resume_qualification"
    full_dir = base / "full_calibration"
    qualification_steps = min(20, max(5, target_steps // 10))

    args.output_dir = str(qualification_dir)
    args.run_id = f"{args.run_id}-RESUME-QUAL"
    first = rt.execute(args, max_optimizer_steps=qualification_steps, resume_path=None, mode="calibration")
    resume_ckpt = qualification_dir / "private_checkpoints/latest.ckpt"
    if not resume_ckpt.exists():
        raise SystemExit("resume qualification failed to produce a checkpoint")

    original_id = args.run_id.removesuffix("-RESUME-QUAL")
    args.run_id = original_id
    args.output_dir = str(full_dir)
    second = rt.execute(args, max_optimizer_steps=target_steps, resume_path=str(resume_ckpt), mode="calibration")

    summary = {
        "schema_version": "1.0",
        "calibration_id": "CAL-MNV4-DIRECT" if args.condition == "direct" else "CAL-MNV4-TEACHER",
        "condition": args.condition,
        "locked_optimizer_steps": target_steps,
        "resume_qualification_steps": qualification_steps,
        "resume_success": second["status"] == "PASS",
        "calibration_weights_scientific": False,
        "qualification_run": first["run_id"],
        "full_run": second["run_id"],
        "measured": second["result_summary"],
        "observed_session_quota_constraints": "record from Kaggle UI/job metadata if exposed; not inferred",
    }
    out = base / "calibration_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
