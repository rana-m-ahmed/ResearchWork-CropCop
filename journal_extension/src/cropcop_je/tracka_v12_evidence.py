from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Iterable

from .tracka_v12_analysis import (
    ArchitectureSelectorRow,
    FAMILIES,
    ROBUSTNESS_CORRUPTIONS,
    ROBUSTNESS_SEVERITIES,
    SEED_LABELS,
    architecture_summary_from_evidence,
    select_journal_primary,
)

NUM_CLASSES = 120
VAL_ROWS = 16368
REPLAY_TOLERANCE = 1e-6
DIRECT_STATES = {
    "R04": tuple(f"R04-MNV4-DIRECT-{seed}" for seed in SEED_LABELS),
    "R06": tuple(f"R06-EFFB0-CONTEXT-{seed}" for seed in SEED_LABELS),
    "R07": tuple(f"R07-CNXTT-CONTEXT-{seed}" for seed in SEED_LABELS),
    "R13": tuple(f"R13-VIT-DLITTLE-DIFF-CONTEXT-{seed}" for seed in SEED_LABELS),
}
ALL_DIRECT_STATES = tuple(run for family in FAMILIES for run in DIRECT_STATES[family])
XAI_SAMPLE_SALT = "TRACKA-A1-XAI-V1"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_prediction_digest(rows: Iterable[dict[str, Any]]) -> str:
    canonical = []
    for row in rows:
        canonical.append(
            {
                "stable_row_id": str(row["stable_row_id"]),
                "target_class_index": int(row["target_class_index"]),
                "predicted_class_index": int(row["predicted_class_index"]),
                "true_class_nll": float(row["true_class_nll"]),
                "top1_confidence": float(row["top1_confidence"]),
            }
        )
    canonical.sort(key=lambda row: row["stable_row_id"])
    blob = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in canonical)
    return _sha256_text(blob)


def _validate_probability(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{label} must be finite and within [0,1]")


def validate_prediction_rows(rows: Iterable[dict[str, Any]], *, expected_count: int = VAL_ROWS) -> list[dict[str, Any]]:
    rows = list(rows)
    if len(rows) != expected_count:
        raise ValueError(f"prediction row count mismatch: expected {expected_count}, got {len(rows)}")
    seen: set[str] = set()
    normalized = []
    for row in rows:
        row_id = str(row.get("stable_row_id", ""))
        if not row_id or row_id in seen:
            raise ValueError("stable_row_id missing or duplicated")
        seen.add(row_id)
        target = int(row["target_class_index"])
        predicted = int(row["predicted_class_index"])
        if not 0 <= target < NUM_CLASSES or not 0 <= predicted < NUM_CLASSES:
            raise ValueError("class index outside frozen 120-class ontology")
        nll = float(row["true_class_nll"])
        confidence = float(row["top1_confidence"])
        if not math.isfinite(nll) or nll < 0.0:
            raise ValueError("true_class_nll must be finite and non-negative")
        _validate_probability(confidence, "top1_confidence")
        normalized.append(
            {
                "stable_row_id": row_id,
                "target_class_index": target,
                "predicted_class_index": predicted,
                "true_class_nll": nll,
                "top1_confidence": confidence,
            }
        )
    return normalized


def summarize_prediction_rows(rows: Iterable[dict[str, Any]], *, expected_count: int = VAL_ROWS) -> dict[str, Any]:
    rows = validate_prediction_rows(rows, expected_count=expected_count)
    confusion = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
    nll_sum = 0.0
    correct = 0
    for row in rows:
        target = row["target_class_index"]
        predicted = row["predicted_class_index"]
        confusion[target][predicted] += 1
        correct += int(target == predicted)
        nll_sum += row["true_class_nll"]

    classwise = []
    recalls = []
    f1s = []
    for class_index in range(NUM_CLASSES):
        tp = confusion[class_index][class_index]
        support = sum(confusion[class_index])
        predicted_count = sum(confusion[row][class_index] for row in range(NUM_CLASSES))
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
        classwise.append(
            {
                "class_index": class_index,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    off_diagonal = []
    for target in range(NUM_CLASSES):
        for predicted in range(NUM_CLASSES):
            count = confusion[target][predicted]
            if target != predicted and count:
                off_diagonal.append(
                    {"target_class_index": target, "predicted_class_index": predicted, "count": count}
                )
    off_diagonal.sort(key=lambda row: (-row["count"], row["target_class_index"], row["predicted_class_index"]))

    return {
        "schema_version": "1.0",
        "row_count": len(rows),
        "unique_row_id_count": len(rows),
        "predictions_sha256": canonical_prediction_digest(rows),
        "validation_accuracy": correct / len(rows),
        "validation_balanced_accuracy": sum(recalls) / NUM_CLASSES,
        "validation_macro_f1": sum(f1s) / NUM_CLASSES,
        "validation_nll": nll_sum / len(rows),
        "classwise": classwise,
        "class_f1": f1s,
        "top_20_directed_off_diagonal_confusions_by_count": off_diagonal[:20],
    }


def validate_selected_checkpoint_replay(summary: dict[str, Any], expected_metrics: dict[str, float], *, tolerance: float = REPLAY_TOLERANCE) -> dict[str, Any]:
    required = (
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
        "validation_nll",
    )
    differences = {name: abs(float(summary[name]) - float(expected_metrics[name])) for name in required}
    status = "PASS" if max(differences.values(), default=0.0) <= tolerance else "FAIL"
    return {
        "status": status,
        "metric_match_tolerance": tolerance,
        "absolute_metric_differences": differences,
        "all_metric_replay_max_abs_diff": max(differences.values(), default=0.0),
        "optimizer_state_advanced": False,
        "training_performed": False,
        "v1_test_accessed": False,
        "protected_external_surface_accessed": False,
    }


def robustness_seed(stable_row_id: str, corruption: str, severity: str) -> int:
    if corruption not in ROBUSTNESS_CORRUPTIONS or severity not in ROBUSTNESS_SEVERITIES:
        raise ValueError("corruption/severity outside frozen robustness protocol")
    digest = hashlib.sha256(f"{stable_row_id}|{corruption}|{severity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def xai_sample(rows: Iterable[dict[str, Any]]) -> list[str]:
    by_class: dict[int, list[str]] = {index: [] for index in range(NUM_CLASSES)}
    for row in rows:
        class_index = int(row["class_index"])
        row_id = str(row["stable_row_id"])
        if not 0 <= class_index < NUM_CLASSES or not row_id:
            raise ValueError("invalid XAI sampling row")
        by_class[class_index].append(row_id)
    selected = []
    for class_index in range(NUM_CLASSES):
        candidates = sorted(by_class[class_index], key=lambda row_id: (_sha256_text(f"{row_id}|{XAI_SAMPLE_SALT}"), row_id))
        if len(candidates) < 2:
            raise ValueError(f"class {class_index} has fewer than two rows for frozen XAI sampling")
        selected.extend(candidates[:2])
    if len(selected) != 240 or len(set(selected)) != 240:
        raise ValueError("frozen XAI sample did not resolve to exactly 240 unique rows")
    return selected


def deterministic_subset(row_ids: Iterable[str], *, count: int, salt: str) -> list[str]:
    unique = sorted(set(str(row_id) for row_id in row_ids), key=lambda row_id: (_sha256_text(f"{row_id}|{salt}"), row_id))
    if len(unique) < count:
        raise ValueError("insufficient rows for deterministic subset")
    return unique[:count]


def xai_random_control_seed(stable_row_id: str, model_run_id: str, fraction: float, control_index: int) -> int:
    if fraction not in {0.1, 0.2, 0.3} or not 0 <= control_index < 10:
        raise ValueError("XAI random-control request outside frozen protocol")
    text = f"{stable_row_id}|{model_run_id}|{fraction}|{control_index}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)


def validate_efficiency_identity(seed_rows: dict[str, dict[str, int]]) -> dict[str, int]:
    if set(seed_rows) != set(SEED_LABELS):
        raise ValueError("efficiency evidence must contain exactly S1,S2,S3")
    fields = ("model_state_tensor_bytes_fp32", "total_parameter_count", "trainable_parameter_count", "input_resolution")
    output = {}
    for field in fields:
        values = {int(seed_rows[seed][field]) for seed in SEED_LABELS}
        if len(values) != 1:
            raise ValueError(f"architecture efficiency identity differs across seeds: {field}")
        value = next(iter(values))
        if value <= 0:
            raise ValueError(f"architecture efficiency metric must be positive: {field}")
        output[field] = value
    return output


def build_family_selector_row(
    family: str,
    *,
    seed_summaries: dict[str, dict[str, Any]],
    corruption_macro_f1: dict[str, dict[str, dict[str, float]]],
    efficiency_by_seed: dict[str, dict[str, int]],
) -> ArchitectureSelectorRow:
    if family not in FAMILIES or set(seed_summaries) != set(SEED_LABELS):
        raise ValueError("family selector evidence inventory mismatch")
    clean_metrics = {}
    class_f1 = {}
    for seed in SEED_LABELS:
        clean_metrics[seed] = {
            "validation_macro_f1": float(seed_summaries[seed]["validation_macro_f1"]),
            "validation_balanced_accuracy": float(seed_summaries[seed]["validation_balanced_accuracy"]),
            "validation_nll": float(seed_summaries[seed]["validation_nll"]),
        }
        values = list(seed_summaries[seed]["class_f1"])
        if len(values) != NUM_CLASSES:
            raise ValueError("class-F1 vector does not contain 120 classes")
        class_f1[seed] = [float(value) for value in values]
    efficiency = validate_efficiency_identity(efficiency_by_seed)
    result = architecture_summary_from_evidence(
        family=family,
        seed_metrics=clean_metrics,
        class_f1_by_seed=class_f1,
        corruptions_by_seed=corruption_macro_f1,
        model_state_tensor_bytes_fp32=efficiency["model_state_tensor_bytes_fp32"],
        total_parameters=efficiency["total_parameter_count"],
    )
    return result["selector_row"]


def build_tracka_selection_closure(
    *,
    selector_rows: Iterable[ArchitectureSelectorRow],
    state_gates: dict[str, dict[str, Any]],
    xai_gates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selector_rows = list(selector_rows)
    if set(state_gates) != set(ALL_DIRECT_STATES) or set(xai_gates) != set(ALL_DIRECT_STATES):
        raise ValueError("final Track-A closure requires exact 12-state direct-candidate evidence")
    failures = []
    for experiment_id in ALL_DIRECT_STATES:
        state = state_gates[experiment_id]
        xai = xai_gates[experiment_id]
        required_true = (
            "terminal",
            "selected_checkpoint_verified",
            "replay_pass",
            "robustness_pass",
            "classwise_pass",
            "efficiency_pass",
        )
        for field in required_true:
            if state.get(field) is not True:
                failures.append(f"{experiment_id}:{field}")
        if state.get("v1_test_accessed") is not False or state.get("external_surface_accessed") is not False:
            failures.append(f"{experiment_id}:protected_surface")
        if xai.get("status") != "PASS" or xai.get("training_or_adaptation_performed") is not False:
            failures.append(f"{experiment_id}:xai")
    if failures:
        return {"status": "FAIL", "science_selection_sealed": False, "failures": sorted(failures)}
    selection = select_journal_primary(selector_rows)
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "science_selection_sealed": True,
        "direct_state_inventory": list(ALL_DIRECT_STATES),
        "selection": selection,
        "xai_used_as_weighted_selector": False,
        "v1_test_accessed": False,
        "external_predictions_opened": False,
        "track_c_candidate_runtime_opened": False,
        "claim_boundary": "best frozen pretrained candidate system under the common CropCop downstream protocol; no causal architecture-superiority claim",
    }
