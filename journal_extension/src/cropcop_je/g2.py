from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json
from .hashing import sha256_json

REQUIRED_CALIBRATIONS = ("CAL-MNV4-DIRECT", "CAL-MNV4-TEACHER", "CAL-CNXTT")
TELEMETRY_FIELDS = (
    "sec_per_optimizer_step",
    "examples_per_second",
    "dataloader_wait_seconds",
    "dataloader_examples_per_wait_second",
    "peak_gpu_memory_bytes",
    "checkpoint_save_seconds",
    "checkpoint_load_seconds",
    "validation_forward_benchmark",
)


def validate_calibration_summary(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if summary.get("calibration_id") not in REQUIRED_CALIBRATIONS:
        errors.append(f"unexpected calibration_id={summary.get('calibration_id')}")
    if summary.get("status") != "PASS":
        errors.append("calibration status is not PASS")
    if summary.get("resume_success") is not True:
        errors.append("save→resume qualification did not pass")
    if summary.get("calibration_weights_scientific") is not False:
        errors.append("calibration weights must be explicitly non-scientific")
    measured = summary.get("measured", {})
    for field in TELEMETRY_FIELDS:
        if measured.get(field) is None:
            errors.append(f"missing telemetry: {field}")
    if not summary.get("source_git_commit"):
        errors.append("source_git_commit missing")
    if not summary.get("software_stack_sha256"):
        errors.append("software_stack_sha256 missing")
    return errors


def build_g2_barrier(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {s.get("calibration_id"): s for s in summaries}
    missing = [x for x in REQUIRED_CALIBRATIONS if x not in by_id]
    errors: list[str] = [f"missing calibration summary: {x}" for x in missing]
    for cid in REQUIRED_CALIBRATIONS:
        if cid in by_id:
            errors.extend(f"{cid}: {e}" for e in validate_calibration_summary(by_id[cid]))
    commits = {by_id[c].get("source_git_commit") for c in REQUIRED_CALIBRATIONS if c in by_id}
    stacks = {by_id[c].get("software_stack_sha256") for c in REQUIRED_CALIBRATIONS if c in by_id}
    if len(commits) > 1:
        errors.append(f"calibrations were not produced from one source Git SHA: {sorted(commits)}")
    if len(stacks) > 1:
        errors.append(f"calibrations used different software stacks: {sorted(stacks)}")

    forecast = {}
    if not errors:
        train_rows = 76376
        val_rows = 16368
        micro = 16
        accum = 4
        epochs = 30
        optimizer_steps = epochs * math.ceil(math.ceil(train_rows / micro) / accum)
        val_examples_total = epochs * val_rows
        for cid in ("CAL-MNV4-DIRECT", "CAL-MNV4-TEACHER"):
            m = by_id[cid]["measured"]
            train_seconds = optimizer_steps * float(m["sec_per_optimizer_step"])
            vbench = m["validation_forward_benchmark"]
            val_rate = float(vbench.get("end_to_end_examples_per_second") or vbench.get("forward_examples_per_second") or 0)
            validation_seconds = val_examples_total / max(val_rate, 1e-12)
            raw = train_seconds + validation_seconds
            forecast[cid] = {
                "optimizer_steps_per_30_epoch_run": optimizer_steps,
                "estimated_train_seconds": train_seconds,
                "estimated_validation_seconds": validation_seconds,
                "estimated_run_seconds": raw,
                "estimated_safe_segments_at_11h": max(1, math.ceil(raw / (11 * 3600))),
                "forecast_is_scheduling_only": True,
            }
    barrier = {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "required_calibrations": list(REQUIRED_CALIBRATIONS),
        "source_git_commit": next(iter(commits)) if len(commits) == 1 else None,
        "software_stack_sha256": next(iter(stacks)) if len(stacks) == 1 else None,
        "errors": errors,
        "forecast": forecast,
        "input_summary_sha256": {cid: sha256_json(by_id[cid]) for cid in REQUIRED_CALIBRATIONS if cid in by_id},
    }
    barrier["barrier_sha256"] = sha256_json(barrier)
    return barrier


def write_g2_barrier(path: str | Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    barrier = build_g2_barrier(summaries)
    atomic_write_json(path, barrier)
    return barrier
