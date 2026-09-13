from __future__ import annotations

import math
from typing import Any

from .hashing import sha256_json
from .tracka_v12 import EXPERIMENT_SPECS
from .tracka_v12_g2a import (
    PROFILE_COVERAGE,
    REQUIRED_PROFILES,
    SLOT_ORDER,
    validate_calibration_summary,
)

TRAIN_ROWS = 76376
VAL_ROWS = 16368
MICRO_BATCH = 16
GRAD_ACCUM = 4
EPOCHS = 30
CHECKPOINT_EVERY_STEPS = 250
BATCHES_PER_EPOCH = math.ceil(TRAIN_ROWS / MICRO_BATCH)
STEP_ATTEMPTS_PER_EPOCH = math.ceil(BATCHES_PER_EPOCH / GRAD_ACCUM)
STEP_ATTEMPTS_TOTAL = EPOCHS * STEP_ATTEMPTS_PER_EPOCH
VAL_EXAMPLES_TOTAL = EPOCHS * VAL_ROWS
CHECKPOINT_EVENT_UPPER_BOUND = STEP_ATTEMPTS_TOTAL // CHECKPOINT_EVERY_STEPS + EPOCHS + EPOCHS

REQUIRED_FORECAST_FIELDS = (
    "steady_sec_per_optimizer_step",
    "validation_end_to_end_examples_per_second",
    "checkpoint_save_seconds_per_event",
    "durable_sync_seconds_per_segment",
)


class TrackAV12G2AV122Error(RuntimeError):
    pass


def barrier_hash(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("barrier_sha256", None)
    return sha256_json(clean)


def scheduler_hash(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("scheduler_freeze_sha256", None)
    return sha256_json(clean)


def validate_calibration_summary_v122(summary: dict[str, Any]) -> list[str]:
    errors = list(validate_calibration_summary(summary))
    measured = summary.get("measured", {})
    for field in REQUIRED_FORECAST_FIELDS:
        value = measured.get(field)
        if value is None:
            errors.append(f"v1.2.2 scheduling telemetry missing: {field}")
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"v1.2.2 scheduling telemetry non-numeric: {field}")
            continue
        if numeric <= 0:
            errors.append(f"v1.2.2 scheduling telemetry must be positive: {field}")
    if summary.get("scientific_metric_computed") is not False or summary.get("validation_enabled") is not False:
        errors.append("v1.2.2 scheduling qualification must remain non-scientific")
    return errors


def profile_full_run_forecast(summary: dict[str, Any]) -> dict[str, float | int | str]:
    errors = validate_calibration_summary_v122(summary)
    if errors:
        raise TrackAV12G2AV122Error("invalid v1.2.2 calibration summary: " + "; ".join(errors))
    measured = summary["measured"]
    steady = float(measured["steady_sec_per_optimizer_step"])
    val_rate = float(measured["validation_end_to_end_examples_per_second"])
    save = float(measured["checkpoint_save_seconds_per_event"])
    durable = float(measured["durable_sync_seconds_per_segment"])
    train_seconds = steady * STEP_ATTEMPTS_TOTAL
    validation_seconds = VAL_EXAMPLES_TOTAL / val_rate
    checkpoint_seconds = save * CHECKPOINT_EVENT_UPPER_BOUND
    total = train_seconds + validation_seconds + checkpoint_seconds + durable
    return {
        "profile": str(summary["calibration_id"]),
        "optimizer_step_attempts": STEP_ATTEMPTS_TOTAL,
        "estimated_train_seconds": train_seconds,
        "validation_examples_total": VAL_EXAMPLES_TOTAL,
        "estimated_validation_seconds": validation_seconds,
        "checkpoint_event_upper_bound": CHECKPOINT_EVENT_UPPER_BOUND,
        "estimated_checkpoint_seconds_upper_bound": checkpoint_seconds,
        "estimated_durable_sync_seconds": durable,
        "estimated_full_run_seconds_upper_bound": total,
        "forecast_role": "pre-science scheduling only",
    }


def build_g2a_v122_barrier(
    summaries: list[dict[str, Any]],
    *,
    checkpoint_contract_probe: dict[str, Any],
) -> dict[str, Any]:
    by_id = {str(row.get("calibration_id", "")): row for row in summaries}
    errors: list[str] = []
    for calibration_id in REQUIRED_PROFILES:
        if calibration_id not in by_id:
            errors.append(f"missing v1.2.2 calibration: {calibration_id}")
        else:
            errors.extend(
                f"{calibration_id}: {error}"
                for error in validate_calibration_summary_v122(by_id[calibration_id])
            )

    common_fields = ("source_git_commit", "software_stack_sha256", "g1a_seal_sha256", "dependency_lock_sha256")
    common_values: dict[str, Any] = {}
    for field in common_fields:
        values = {by_id[cid].get(field) for cid in REQUIRED_PROFILES if cid in by_id}
        if len(values) != 1:
            errors.append(f"v1.2.2 calibrations disagree on {field}: {sorted(str(x) for x in values)}")
            common_values[field] = None
        else:
            common_values[field] = next(iter(values))

    union = set()
    for calibration_id in REQUIRED_PROFILES:
        if calibration_id in by_id:
            union.update(by_id[calibration_id].get("coverage", []))
    if union != set(EXPERIMENT_SPECS):
        errors.append("v1.2.2 calibration coverage does not equal the exact 11-state inventory")

    if checkpoint_contract_probe.get("status") != "PASS":
        errors.append("v1.2.2 checkpoint contract probe is not PASS")
    for field in ("selected_checkpoint_verified", "identity_mismatch_rejected", "corrupt_checkpoint_rejected"):
        if checkpoint_contract_probe.get(field) is not True:
            errors.append(f"v1.2.2 checkpoint contract probe missing {field}")

    forecasts = {}
    if not errors:
        forecasts = {cid: profile_full_run_forecast(by_id[cid]) for cid in REQUIRED_PROFILES}

    barrier = {
        "schema_version": "1.2.2",
        "barrier_kind": "track_a_v12_g2a_full_run_forecast",
        "status": "PASS" if not errors else "FAIL",
        "science_authorized": False,
        "required_profiles": list(REQUIRED_PROFILES),
        "covered_experiment_ids": sorted(union),
        **common_values,
        "input_summary_sha256": {cid: sha256_json(by_id[cid]) for cid in REQUIRED_PROFILES if cid in by_id},
        "checkpoint_contract_probe": checkpoint_contract_probe,
        "forecast_contract": {
            "train_rows": TRAIN_ROWS,
            "validation_rows": VAL_ROWS,
            "micro_batch_size": MICRO_BATCH,
            "gradient_accumulation": GRAD_ACCUM,
            "epochs": EPOCHS,
            "checkpoint_every_steps": CHECKPOINT_EVERY_STEPS,
            "optimizer_step_attempts_total": STEP_ATTEMPTS_TOTAL,
            "validation_examples_total": VAL_EXAMPLES_TOTAL,
            "checkpoint_event_upper_bound": CHECKPOINT_EVENT_UPPER_BOUND,
        },
        "profile_forecasts": forecasts,
        "errors": errors,
    }
    barrier["barrier_sha256"] = barrier_hash(barrier)
    return barrier


def validate_g2a_v122_barrier(
    barrier: dict[str, Any],
    *,
    expected_source_sha: str | None = None,
    expected_g1a_seal_sha256: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if barrier.get("schema_version") != "1.2.2" or barrier.get("barrier_kind") != "track_a_v12_g2a_full_run_forecast":
        errors.append("unsupported v1.2.2 G2A barrier schema/kind")
    if barrier.get("status") != "PASS" or barrier.get("errors"):
        errors.append("v1.2.2 G2A barrier is not terminal PASS")
    if barrier.get("science_authorized") is not False:
        errors.append("v1.2.2 G2A barrier may not independently authorize science")
    if barrier.get("barrier_sha256") != barrier_hash(barrier):
        errors.append("v1.2.2 G2A barrier self-hash mismatch")
    if set(barrier.get("covered_experiment_ids", [])) != set(EXPERIMENT_SPECS):
        errors.append("v1.2.2 G2A experiment coverage mismatch")
    if expected_source_sha and barrier.get("source_git_commit") != expected_source_sha:
        errors.append("v1.2.2 G2A source SHA mismatch")
    if expected_g1a_seal_sha256 and barrier.get("g1a_seal_sha256") != expected_g1a_seal_sha256:
        errors.append("v1.2.2 G2A G1A binding mismatch")
    forecasts = barrier.get("profile_forecasts", {})
    if set(forecasts) != set(REQUIRED_PROFILES):
        errors.append("v1.2.2 G2A profile forecast inventory mismatch")
    elif any(float(row.get("estimated_full_run_seconds_upper_bound", 0.0)) <= 0 for row in forecasts.values()):
        errors.append("v1.2.2 full-run forecast must be positive for every profile")
    return errors


def profile_for_experiment(experiment_id: str) -> str:
    matches = [profile for profile, coverage in PROFILE_COVERAGE.items() if experiment_id in coverage]
    if len(matches) != 1:
        raise TrackAV12G2AV122Error(f"experiment does not map to exactly one scheduling profile: {experiment_id}")
    return matches[0]


def _static_lpt_slot_queues(rows: list[tuple[str, str, float]]) -> tuple[dict[str, list[str]], dict[str, float]]:
    loads = {slot: 0.0 for slot in SLOT_ORDER}
    queues = {slot: [] for slot in SLOT_ORDER}
    slot_rank = {slot: index for index, slot in enumerate(SLOT_ORDER)}
    for experiment_id, _profile, cost in rows:
        slot = min(SLOT_ORDER, key=lambda candidate: (loads[candidate], slot_rank[candidate]))
        queues[slot].append(experiment_id)
        loads[slot] += float(cost)
    return queues, loads


def build_scheduler_freeze_v122(barrier: dict[str, Any]) -> dict[str, Any]:
    errors = validate_g2a_v122_barrier(barrier)
    if errors:
        raise TrackAV12G2AV122Error("cannot freeze v1.2.2 scheduler from invalid barrier: " + "; ".join(errors))
    forecasts = barrier["profile_forecasts"]
    rows = []
    for experiment_id in sorted(EXPERIMENT_SPECS):
        profile = profile_for_experiment(experiment_id)
        cost = float(forecasts[profile]["estimated_full_run_seconds_upper_bound"])
        rows.append((experiment_id, profile, cost))
    rows.sort(key=lambda row: (-row[2], row[0]))
    priority = [row[0] for row in rows]
    queues, loads = _static_lpt_slot_queues(rows)
    initial = [
        {"slot": slot, "experiment_id": queues[slot][0]}
        for slot in SLOT_ORDER
        if queues[slot]
    ]
    freeze = {
        "schema_version": "1.2.2",
        "freeze_kind": "track_a_v12_pre_science_scheduler_full_run",
        "status": "PASS",
        "science_authorized": False,
        "source_git_commit": barrier["source_git_commit"],
        "g1a_seal_sha256": barrier["g1a_seal_sha256"],
        "g2a_barrier_sha256": barrier["barrier_sha256"],
        "algorithm": "deterministic_lpt_list_scheduling_to_six_slots",
        "cost_source": "sealed non-scientific estimated full 30-epoch run seconds only",
        "slot_order": list(SLOT_ORDER),
        "priority": priority,
        "priority_rows": [
            {"experiment_id": eid, "profile": profile, "estimated_full_run_seconds_upper_bound": cost}
            for eid, profile, cost in rows
        ],
        "static_slot_queues": queues,
        "predicted_slot_load_seconds": loads,
        "initial_dispatch": initial,
        "continuation_dispatch": "each physical slot consumes its frozen queue sequentially; no mutable cross-account central queue",
        "checkpoint_identity_is_physical_slot_independent": True,
    }
    freeze["scheduler_freeze_sha256"] = scheduler_hash(freeze)
    return freeze


def validate_scheduler_freeze_v122(
    freeze: dict[str, Any],
    *,
    expected_g2a_barrier_sha256: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if freeze.get("schema_version") != "1.2.2" or freeze.get("freeze_kind") != "track_a_v12_pre_science_scheduler_full_run":
        errors.append("unsupported v1.2.2 scheduler schema/kind")
    if freeze.get("status") != "PASS":
        errors.append("v1.2.2 scheduler is not PASS")
    if freeze.get("science_authorized") is not False:
        errors.append("v1.2.2 scheduler may not independently authorize science")
    if freeze.get("scheduler_freeze_sha256") != scheduler_hash(freeze):
        errors.append("v1.2.2 scheduler self-hash mismatch")
    if tuple(freeze.get("slot_order", [])) != SLOT_ORDER:
        errors.append("v1.2.2 scheduler slot order drift")
    priority = freeze.get("priority", [])
    if len(priority) != len(EXPERIMENT_SPECS) or set(priority) != set(EXPERIMENT_SPECS):
        errors.append("v1.2.2 scheduler priority is not an exact permutation of 11 frozen states")
    queues = freeze.get("static_slot_queues", {})
    if set(queues) != set(SLOT_ORDER):
        errors.append("v1.2.2 scheduler static slot inventory mismatch")
    else:
        flattened = [experiment_id for slot in SLOT_ORDER for experiment_id in queues[slot]]
        if len(flattened) != len(EXPERIMENT_SPECS) or set(flattened) != set(EXPERIMENT_SPECS):
            errors.append("v1.2.2 static slot queues are not an exact one-time partition of 11 states")
        initial = freeze.get("initial_dispatch", [])
        expected_initial = [
            {"slot": slot, "experiment_id": queues[slot][0]}
            for slot in SLOT_ORDER
            if queues[slot]
        ]
        if initial != expected_initial or len(initial) != 6:
            errors.append("v1.2.2 initial dispatch does not match six frozen queue heads")
    loads = freeze.get("predicted_slot_load_seconds", {})
    if set(loads) != set(SLOT_ORDER) or any(float(loads.get(slot, 0.0)) <= 0 for slot in SLOT_ORDER):
        errors.append("v1.2.2 predicted slot-load inventory invalid")
    if freeze.get("checkpoint_identity_is_physical_slot_independent") is not True:
        errors.append("v1.2.2 scheduler does not affirm placement-neutral checkpoint identity")
    if expected_g2a_barrier_sha256 and freeze.get("g2a_barrier_sha256") != expected_g2a_barrier_sha256:
        errors.append("v1.2.2 scheduler/G2A binding mismatch")
    return errors
