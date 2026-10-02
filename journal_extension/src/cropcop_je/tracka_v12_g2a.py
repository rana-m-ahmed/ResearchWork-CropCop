from __future__ import annotations

from typing import Any

from .hashing import sha256_json
from .tracka_v12 import EXPERIMENT_SPECS
from .tracka_v12_g1a import R13_PRETRAINED_SHA256, TEACHER_SHA256

PROFILE_COVERAGE = {
    "CAL-EFFB0": {"R06-EFFB0-CONTEXT-S2", "R06-EFFB0-CONTEXT-S3"},
    "CAL-CNXTT": {"R07-CNXTT-CONTEXT-S2", "R07-CNXTT-CONTEXT-S3"},
    "CAL-MNV4-LOGITS": {"R12-MNV4-LOGITS-S2", "R12-MNV4-LOGITS-S3"},
    "CAL-MNV4-FEATURE": {"R12-MNV4-FEATURE-S2", "R12-MNV4-FEATURE-S3"},
    "CAL-R13": {
        "R13-VIT-DLITTLE-DIFF-CONTEXT-S1",
        "R13-VIT-DLITTLE-DIFF-CONTEXT-S2",
        "R13-VIT-DLITTLE-DIFF-CONTEXT-S3",
    },
}
REQUIRED_PROFILES = tuple(PROFILE_COVERAGE)
SLOT_ORDER = ("K1/GPU0", "K1/GPU1", "K2/GPU0", "K2/GPU1", "K3/GPU0", "K3/GPU1")
TELEMETRY_FIELDS = (
    "sec_per_optimizer_step",
    "examples_per_second",
    "peak_gpu_memory_bytes",
    "checkpoint_save_seconds",
    "checkpoint_load_seconds",
)


class TrackAV12G2AError(RuntimeError):
    pass


def g2a_barrier_hash(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("barrier_sha256", None)
    return sha256_json(clean)


def scheduler_freeze_hash(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("scheduler_freeze_sha256", None)
    return sha256_json(clean)


def validate_calibration_summary(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    calibration_id = str(summary.get("calibration_id", ""))
    if calibration_id not in PROFILE_COVERAGE:
        return [f"unexpected Track-A v1.2 calibration ID: {calibration_id}"]
    if summary.get("status") != "PASS":
        errors.append("calibration status is not PASS")
    if summary.get("calibration_weights_scientific") is not False:
        errors.append("calibration weights must be explicitly non-scientific")
    if summary.get("validation_enabled") is not False:
        errors.append("G2A calibration must not compute validation metrics")
    if summary.get("scientific_metric_computed") is not False:
        errors.append("G2A calibration unexpectedly computed a scientific metric")
    if summary.get("resume_success") is not True:
        errors.append("G2A save/resume qualification failed")
    if summary.get("durable_roundtrip_success") is not True:
        errors.append("G2A durable checkpoint roundtrip failed")
    if summary.get("visible_cuda_device_count") != 1:
        errors.append("G2A child must see exactly one CUDA device")
    if summary.get("visible_gpu_name") not in {"Tesla T4", "NVIDIA T4"}:
        errors.append("G2A child did not observe an NVIDIA T4")
    if summary.get("git_credentials_present") is not False:
        errors.append("G2A child inherited Git publication credentials")
    if summary.get("allowed_surfaces") != ["DS-V1-TRAIN", "DS-V1-VAL"]:
        errors.append("G2A allowed surface set drift")
    if summary.get("v1_test_accessed") is not False:
        errors.append("G2A accessed V1 test")
    if summary.get("external_protected_surface_accessed") is not False:
        errors.append("G2A accessed external protected data")
    if set(summary.get("coverage", [])) != PROFILE_COVERAGE[calibration_id]:
        errors.append(f"G2A coverage mismatch for {calibration_id}")
    for field, lengths in (
        ("source_git_commit", {40}),
        ("software_stack_sha256", {64}),
        ("g1a_seal_sha256", {64}),
        ("dependency_lock_sha256", {64}),
    ):
        if len(str(summary.get(field, ""))) not in lengths:
            errors.append(f"G2A identity missing/invalid: {field}")
    if calibration_id.startswith("CAL-MNV4-") and summary.get("teacher_checkpoint_sha256") != TEACHER_SHA256:
        errors.append("G2A MNV4 teacher identity mismatch")
    if calibration_id == "CAL-R13" and summary.get("r13_pretrained_sha256") != R13_PRETRAINED_SHA256:
        errors.append("G2A R13 pretrained identity mismatch")
    measured = summary.get("measured", {})
    for field in TELEMETRY_FIELDS:
        value = measured.get(field)
        if value is None or float(value) < 0:
            errors.append(f"G2A telemetry missing/invalid: {field}")
    if float(measured.get("sec_per_optimizer_step", 0.0) or 0.0) <= 0:
        errors.append("G2A sec_per_optimizer_step must be positive")
    durability = summary.get("durability", {})
    if durability.get("sync_status") != "PASS" or durability.get("restore_status") != "PASS":
        errors.append("G2A durable checkpoint roundtrip status is not PASS")
    return errors


def build_g2a_barrier(summaries: list[dict[str, Any]], *, checkpoint_contract_probe: dict[str, Any]) -> dict[str, Any]:
    by_id = {str(row.get("calibration_id", "")): row for row in summaries}
    errors: list[str] = []
    for calibration_id in REQUIRED_PROFILES:
        if calibration_id not in by_id:
            errors.append(f"missing G2A calibration: {calibration_id}")
        else:
            errors.extend(f"{calibration_id}: {error}" for error in validate_calibration_summary(by_id[calibration_id]))

    common_fields = ("source_git_commit", "software_stack_sha256", "g1a_seal_sha256", "dependency_lock_sha256")
    common_values: dict[str, Any] = {}
    for field in common_fields:
        values = {by_id[cid].get(field) for cid in REQUIRED_PROFILES if cid in by_id}
        if len(values) != 1:
            errors.append(f"G2A calibrations disagree on {field}: {sorted(str(x) for x in values)}")
            common_values[field] = None
        else:
            common_values[field] = next(iter(values))

    union = set()
    for calibration_id in REQUIRED_PROFILES:
        if calibration_id in by_id:
            union.update(by_id[calibration_id].get("coverage", []))
    if union != set(EXPERIMENT_SPECS):
        errors.append("G2A calibration coverage does not equal the exact 11-state continuation inventory")

    if checkpoint_contract_probe.get("status") != "PASS":
        errors.append("G2A checkpoint contract probe is not PASS")
    for field in ("selected_checkpoint_verified", "identity_mismatch_rejected", "corrupt_checkpoint_rejected"):
        if checkpoint_contract_probe.get(field) is not True:
            errors.append(f"G2A checkpoint contract probe missing {field}")

    profile_cost_seconds_per_step = {}
    if not errors:
        for calibration_id in REQUIRED_PROFILES:
            profile_cost_seconds_per_step[calibration_id] = float(by_id[calibration_id]["measured"]["sec_per_optimizer_step"])

    barrier = {
        "schema_version": "1.0",
        "barrier_kind": "track_a_v12_g2a",
        "status": "PASS" if not errors else "FAIL",
        "science_authorized": False,
        "required_profiles": list(REQUIRED_PROFILES),
        "covered_experiment_ids": sorted(union),
        **common_values,
        "input_summary_sha256": {cid: sha256_json(by_id[cid]) for cid in REQUIRED_PROFILES if cid in by_id},
        "profile_cost_seconds_per_optimizer_step": profile_cost_seconds_per_step,
        "checkpoint_contract_probe": checkpoint_contract_probe,
        "errors": errors,
    }
    barrier["barrier_sha256"] = g2a_barrier_hash(barrier)
    return barrier


def validate_g2a_barrier_object(
    barrier: dict[str, Any],
    *,
    expected_source_sha: str | None = None,
    expected_g1a_seal_sha256: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if barrier.get("schema_version") != "1.0" or barrier.get("barrier_kind") != "track_a_v12_g2a":
        errors.append("unsupported G2A barrier schema/kind")
    if barrier.get("status") != "PASS" or barrier.get("errors"):
        errors.append("G2A barrier is not terminal PASS")
    if barrier.get("science_authorized") is not False:
        errors.append("G2A barrier may not independently authorize science")
    if barrier.get("barrier_sha256") != g2a_barrier_hash(barrier):
        errors.append("G2A barrier self-hash mismatch")
    if tuple(barrier.get("required_profiles", [])) != REQUIRED_PROFILES:
        errors.append("G2A required profile order/set mismatch")
    if set(barrier.get("covered_experiment_ids", [])) != set(EXPERIMENT_SPECS):
        errors.append("G2A experiment coverage mismatch")
    if expected_source_sha and barrier.get("source_git_commit") != expected_source_sha:
        errors.append("G2A source SHA mismatch")
    if expected_g1a_seal_sha256 and barrier.get("g1a_seal_sha256") != expected_g1a_seal_sha256:
        errors.append("G2A G1A-seal binding mismatch")
    for field in ("software_stack_sha256", "g1a_seal_sha256", "dependency_lock_sha256"):
        if len(str(barrier.get(field, ""))) != 64:
            errors.append(f"G2A {field} missing/invalid")
    costs = barrier.get("profile_cost_seconds_per_optimizer_step", {})
    if set(costs) != set(REQUIRED_PROFILES):
        errors.append("G2A profile cost inventory mismatch")
    elif any(float(value) <= 0 for value in costs.values()):
        errors.append("G2A profile costs must all be positive")
    return errors


def profile_for_experiment(experiment_id: str) -> str:
    matches = [profile for profile, coverage in PROFILE_COVERAGE.items() if experiment_id in coverage]
    if len(matches) != 1:
        raise TrackAV12G2AError(f"experiment does not map to exactly one G2A profile: {experiment_id}")
    return matches[0]


def build_scheduler_freeze(barrier: dict[str, Any]) -> dict[str, Any]:
    errors = validate_g2a_barrier_object(barrier)
    if errors:
        raise TrackAV12G2AError("cannot freeze scheduler from invalid G2A barrier: " + "; ".join(errors))
    costs = barrier["profile_cost_seconds_per_optimizer_step"]
    rows = []
    for experiment_id in sorted(EXPERIMENT_SPECS):
        profile = profile_for_experiment(experiment_id)
        rows.append((experiment_id, profile, float(costs[profile])))
    rows.sort(key=lambda row: (-row[2], row[0]))
    priority = [row[0] for row in rows]
    initial = [
        {"slot": slot, "experiment_id": experiment_id}
        for slot, experiment_id in zip(SLOT_ORDER, priority[: len(SLOT_ORDER)])
    ]
    freeze = {
        "schema_version": "1.0",
        "freeze_kind": "track_a_v12_pre_science_scheduler",
        "status": "PASS",
        "science_authorized": False,
        "source_git_commit": barrier["source_git_commit"],
        "g1a_seal_sha256": barrier["g1a_seal_sha256"],
        "g2a_barrier_sha256": barrier["barrier_sha256"],
        "algorithm": "deterministic_longest_processing_time_first",
        "cost_source": "sealed G2A sec_per_optimizer_step only; no scientific metric",
        "slot_order": list(SLOT_ORDER),
        "priority": priority,
        "priority_rows": [
            {"experiment_id": eid, "profile": profile, "sec_per_optimizer_step": cost}
            for eid, profile, cost in rows
        ],
        "initial_dispatch": initial,
        "continuation_dispatch": "next priority item goes to first free qualified slot; simultaneous-free ties use slot_order",
    }
    freeze["scheduler_freeze_sha256"] = scheduler_freeze_hash(freeze)
    return freeze


def validate_scheduler_freeze(freeze: dict[str, Any], *, expected_g2a_barrier_sha256: str | None = None) -> list[str]:
    errors: list[str] = []
    if freeze.get("schema_version") != "1.0" or freeze.get("freeze_kind") != "track_a_v12_pre_science_scheduler":
        errors.append("unsupported scheduler freeze schema/kind")
    if freeze.get("status") != "PASS":
        errors.append("scheduler freeze is not PASS")
    if freeze.get("science_authorized") is not False:
        errors.append("scheduler freeze may not independently authorize science")
    if freeze.get("scheduler_freeze_sha256") != scheduler_freeze_hash(freeze):
        errors.append("scheduler freeze self-hash mismatch")
    if tuple(freeze.get("slot_order", [])) != SLOT_ORDER:
        errors.append("scheduler slot order drift")
    priority = freeze.get("priority", [])
    if len(priority) != len(EXPERIMENT_SPECS) or set(priority) != set(EXPERIMENT_SPECS):
        errors.append("scheduler priority is not an exact permutation of 11 frozen states")
    initial = freeze.get("initial_dispatch", [])
    if len(initial) != 6:
        errors.append("scheduler must initially fill exactly six GPU slots")
    if expected_g2a_barrier_sha256 and freeze.get("g2a_barrier_sha256") != expected_g2a_barrier_sha256:
        errors.append("scheduler/G2A binding mismatch")
    return errors
