from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json
from .g1 import TEACHER_SHA256
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


def barrier_hash(barrier: dict[str, Any]) -> str:
    clean = dict(barrier)
    clean.pop("barrier_sha256", None)
    return sha256_json(clean)


def validate_calibration_summary(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cid = summary.get("calibration_id")
    if cid not in REQUIRED_CALIBRATIONS:
        errors.append(f"unexpected calibration_id={cid}")
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
    for field in ("source_git_commit", "software_stack_sha256", "g1_seal_sha256",
                  "dependency_lock_sha256", "mnv4_pretrained_sha256"):
        if not summary.get(field):
            errors.append(f"{field} missing")
    if cid == "CAL-MNV4-TEACHER":
        if summary.get("teacher_checkpoint_sha256") != TEACHER_SHA256:
            errors.append("teacher calibration checkpoint SHA mismatch")
        if len(str(summary.get("teacher_factory_bundle_sha256", ""))) != 64:
            errors.append("teacher calibration factory bundle SHA missing/invalid")
    if cid == "CAL-CNXTT":
        cnxtt = str(summary.get("cnxtt_pretrained_sha256", ""))
        if len(cnxtt) != 64 or not cnxtt.startswith("983f1562"):
            errors.append("ConvNeXt-Tiny calibration pretrained SHA missing/invalid")
    return errors


def build_g2_barrier(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {s.get("calibration_id"): s for s in summaries}
    missing = [x for x in REQUIRED_CALIBRATIONS if x not in by_id]
    errors: list[str] = [f"missing calibration summary: {x}" for x in missing]
    for cid in REQUIRED_CALIBRATIONS:
        if cid in by_id:
            errors.extend(f"{cid}: {e}" for e in validate_calibration_summary(by_id[cid]))

    def values(field: str):
        return {by_id[c].get(field) for c in REQUIRED_CALIBRATIONS if c in by_id}

    commits = values("source_git_commit")
    stacks = values("software_stack_sha256")
    g1s = values("g1_seal_sha256")
    deps = values("dependency_lock_sha256")
    mnv4 = values("mnv4_pretrained_sha256")
    cnxtt_sha = by_id.get("CAL-CNXTT", {}).get("cnxtt_pretrained_sha256")
    for label, vals in (
        ("source Git SHA", commits),
        ("software stack", stacks),
        ("G1 seal", g1s),
        ("dependency lock", deps),
        ("MobileNetV4 pretrained identity", mnv4),
    ):
        if len(vals) > 1:
            errors.append(f"calibrations used different {label}: {sorted(str(x) for x in vals)}")

    teacher_summary = by_id.get("CAL-MNV4-TEACHER", {})
    teacher_sha = teacher_summary.get("teacher_checkpoint_sha256")
    factory_sha = teacher_summary.get("teacher_factory_bundle_sha256")
    if teacher_summary and teacher_sha != TEACHER_SHA256:
        errors.append("teacher calibration did not bind the frozen historical teacher")
    if teacher_summary and len(str(factory_sha or "")) != 64:
        errors.append("teacher calibration did not bind one teacher factory bundle")

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
                "estimated_checkpoint_seconds": float(m.get("checkpoint_save_seconds") or 0.0),
                "estimated_durable_sync_seconds": float(m.get("durable_sync_seconds") or 0.0),
                "estimated_safe_segments_at_11h": max(1, math.ceil(raw / (11 * 3600))),
                "forecast_is_scheduling_only": True,
            }
    barrier = {
        "schema_version": "2.0",
        "status": "PASS" if not errors else "FAIL",
        "required_calibrations": list(REQUIRED_CALIBRATIONS),
        "source_git_commit": next(iter(commits)) if len(commits) == 1 else None,
        "software_stack_sha256": next(iter(stacks)) if len(stacks) == 1 else None,
        "g1_seal_sha256": next(iter(g1s)) if len(g1s) == 1 else None,
        "dependency_lock_sha256": next(iter(deps)) if len(deps) == 1 else None,
        "mnv4_pretrained_sha256": next(iter(mnv4)) if len(mnv4) == 1 else None,
        "teacher_checkpoint_sha256": teacher_sha,
        "teacher_factory_bundle_sha256": factory_sha,
        "cnxtt_pretrained_sha256": cnxtt_sha,
        "errors": errors,
        "forecast": forecast,
        "input_summary_sha256": {cid: sha256_json(by_id[cid]) for cid in REQUIRED_CALIBRATIONS if cid in by_id},
    }
    barrier["barrier_sha256"] = barrier_hash(barrier)
    return barrier


def validate_g2_barrier_object(barrier: dict[str, Any], *, expected_source_sha: str | None = None,
                               expected_g1_seal_sha256: str | None = None) -> list[str]:
    errors: list[str] = []
    if barrier.get("schema_version") != "2.0":
        errors.append("unsupported G2 barrier schema")
    if barrier.get("status") != "PASS" or barrier.get("errors"):
        errors.append("G2 barrier is not terminal PASS")
    if barrier.get("barrier_sha256") != barrier_hash(barrier):
        errors.append("G2 barrier self-hash mismatch")
    if expected_source_sha and barrier.get("source_git_commit") != expected_source_sha:
        errors.append("G2 source Git SHA mismatch")
    if expected_g1_seal_sha256 and barrier.get("g1_seal_sha256") != expected_g1_seal_sha256:
        errors.append("G2 G1-seal binding mismatch")
    for field in ("dependency_lock_sha256", "software_stack_sha256", "mnv4_pretrained_sha256"):
        if len(str(barrier.get(field, ""))) != 64:
            errors.append(f"G2 {field} missing/invalid")
    if barrier.get("teacher_checkpoint_sha256") != TEACHER_SHA256:
        errors.append("G2 teacher checkpoint identity mismatch")
    if len(str(barrier.get("teacher_factory_bundle_sha256", ""))) != 64:
        errors.append("G2 teacher factory bundle identity missing")
    cnxtt = str(barrier.get("cnxtt_pretrained_sha256", ""))
    if len(cnxtt) != 64 or not cnxtt.startswith("983f1562"):
        errors.append("G2 ConvNeXt-Tiny pretrained identity missing/invalid")
    return errors


def write_g2_barrier(path: str | Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    barrier = build_g2_barrier(summaries)
    atomic_write_json(path, barrier)
    return barrier


def validate_principal_gate_bindings(
    g1_seal: dict[str, Any] | None,
    g2_barrier: dict[str, Any] | None,
    *,
    source_git_sha: str,
) -> list[str]:
    errors: list[str] = []
    if not g1_seal:
        return ["principal launch requires G1 seal", "principal launch requires G2 barrier"] if not g2_barrier else ["principal launch requires G1 seal"]
    from .g1 import validate_g1_seal_object
    errors.extend("G1: " + e for e in validate_g1_seal_object(g1_seal))
    if g1_seal.get("source_git_sha") != source_git_sha:
        errors.append("G1 source Git SHA mismatch")
    if not g2_barrier:
        errors.append("principal launch requires G2 barrier")
        return errors
    errors.extend("G2: " + e for e in validate_g2_barrier_object(
        g2_barrier,
        expected_source_sha=source_git_sha,
        expected_g1_seal_sha256=g1_seal.get("g1_seal_sha256"),
    ))
    if g2_barrier.get("dependency_lock_sha256") != g1_seal.get("dependency_lock_sha256"):
        errors.append("G1/G2 dependency-lock binding mismatch")
    if g2_barrier.get("mnv4_pretrained_sha256") != g1_seal.get("student", {}).get("pretrained", {}).get("sha256"):
        errors.append("G1/G2 MobileNetV4 pretrained binding mismatch")
    if g2_barrier.get("teacher_factory_bundle_sha256") != g1_seal.get("teacher", {}).get("factory_bundle_sha256"):
        errors.append("G1/G2 teacher-factory bundle binding mismatch")
    return errors
