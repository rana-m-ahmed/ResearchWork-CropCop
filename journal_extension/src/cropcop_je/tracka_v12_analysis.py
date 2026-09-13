from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any, Iterable

FAMILIES = ("R04", "R06", "R07", "R13")
SEED_LABELS = ("S1", "S2", "S3")
Q8 = Decimal("0.00000001")
ROBUSTNESS_CORRUPTIONS = (
    "brightness",
    "contrast",
    "gaussian_blur",
    "gaussian_noise_uint8",
    "jpeg",
)
ROBUSTNESS_SEVERITIES = ("1", "2", "3")


def q8(value: float | int | str | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(Q8, rounding=ROUND_HALF_EVEN)


def arithmetic_mean(values: Iterable[float]) -> float:
    values = [float(x) for x in values]
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def sample_sd(values: Iterable[float]) -> float:
    values = [float(x) for x in values]
    if len(values) < 2:
        raise ValueError("sample SD requires at least two values")
    mean = arithmetic_mean(values)
    return (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def class_tail_summary(class_f1_by_seed: dict[str, list[float]], *, num_classes: int = 120, bottom_k: int = 12) -> dict[str, Any]:
    if set(class_f1_by_seed) != set(SEED_LABELS):
        raise ValueError("class-F1 evidence must contain exactly S1,S2,S3")
    for label in SEED_LABELS:
        if len(class_f1_by_seed[label]) != num_classes:
            raise ValueError(f"{label} class-F1 vector must contain {num_classes} entries")
    class_means = []
    for class_index in range(num_classes):
        mean_f1 = arithmetic_mean(class_f1_by_seed[label][class_index] for label in SEED_LABELS)
        class_means.append((class_index, mean_f1))
    ordered = sorted(class_means, key=lambda row: (q8(row[1]), row[0]))
    bottom = ordered[:bottom_k]
    minimum = ordered[0]
    return {
        "bottom_class_indices": [idx for idx, _ in bottom],
        "bottom_12_class_mean_f1": arithmetic_mean(value for _, value in bottom),
        "minimum_class_index": minimum[0],
        "minimum_class_mean_f1": minimum[1],
        "class_mean_f1": [value for _, value in class_means],
    }


def corruption_summary(
    clean_macro_f1_by_seed: dict[str, float],
    corrupted_macro_f1_by_seed: dict[str, dict[str, dict[str, float]]],
    *,
    expected_corruptions: tuple[str, ...] = ROBUSTNESS_CORRUPTIONS,
    expected_severities: tuple[str, ...] = ROBUSTNESS_SEVERITIES,
) -> dict[str, Any]:
    if set(clean_macro_f1_by_seed) != set(SEED_LABELS) or set(corrupted_macro_f1_by_seed) != set(SEED_LABELS):
        raise ValueError("corruption evidence must contain exactly S1,S2,S3")
    state_means: dict[str, float] = {}
    state_worst: dict[str, float] = {}
    monotonicity: dict[str, int] = {}
    all_deltas = []
    for seed_label in SEED_LABELS:
        seed = corrupted_macro_f1_by_seed[seed_label]
        if set(seed) != set(expected_corruptions):
            raise ValueError(f"{seed_label} corruption inventory mismatch")
        deltas = []
        violations = 0
        clean = float(clean_macro_f1_by_seed[seed_label])
        for corruption in expected_corruptions:
            cells = seed[corruption]
            if set(cells) != set(expected_severities):
                raise ValueError(f"{seed_label}/{corruption} severity inventory mismatch")
            ordered_deltas = []
            for severity in expected_severities:
                delta_pp = (clean - float(cells[severity])) * 100.0
                deltas.append(delta_pp)
                ordered_deltas.append(delta_pp)
                all_deltas.append(delta_pp)
            for prior, later in zip(ordered_deltas, ordered_deltas[1:]):
                if q8(later) < q8(prior):
                    violations += 1
        state_means[seed_label] = arithmetic_mean(deltas)
        state_worst[seed_label] = max(deltas)
        monotonicity[seed_label] = violations
    return {
        "mean_corruption_degradation_pp": arithmetic_mean(state_means.values()),
        "worst_case_macro_f1_delta_pp": max(all_deltas),
        "severity_monotonicity_violations_count": sum(monotonicity.values()),
        "state_mean_corruption_degradation_pp": state_means,
        "state_worst_case_degradation_pp": state_worst,
        "state_monotonicity_violations": monotonicity,
    }


def canonical_model_state_bytes_fp32(state_dict: dict[str, Any]) -> int:
    total = 0
    for tensor in state_dict.values():
        if not hasattr(tensor, "numel") or not hasattr(tensor, "element_size"):
            raise TypeError("model state contains a non-tensor value")
        numel = int(tensor.numel())
        is_floating = bool(getattr(tensor, "is_floating_point")()) if callable(getattr(tensor, "is_floating_point", None)) else False
        total += numel * (4 if is_floating else int(tensor.element_size()))
    return total


def total_parameter_count(model) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


@dataclass(frozen=True)
class ArchitectureSelectorRow:
    family: str
    mean_validation_macro_f1: float
    worst_seed_validation_macro_f1: float
    bottom_12_class_mean_f1: float
    mean_corruption_degradation_pp: float
    model_state_tensor_bytes_fp32: int
    total_parameter_count: int
    sample_sd_validation_macro_f1: float | None = None
    mean_balanced_accuracy: float | None = None
    mean_validation_nll: float | None = None

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown candidate family {self.family}")
        if self.model_state_tensor_bytes_fp32 <= 0 or self.total_parameter_count <= 0:
            raise ValueError(f"invalid efficiency identity for {self.family}")

    def pareto_vector(self) -> dict[str, Decimal | int]:
        self.validate()
        return {
            "mean_validation_macro_f1": q8(self.mean_validation_macro_f1),
            "worst_seed_validation_macro_f1": q8(self.worst_seed_validation_macro_f1),
            "bottom_12_class_mean_f1": q8(self.bottom_12_class_mean_f1),
            "mean_corruption_degradation_pp": q8(self.mean_corruption_degradation_pp),
            "model_state_tensor_bytes_fp32": int(self.model_state_tensor_bytes_fp32),
        }

    def lexicographic_key(self) -> tuple[Decimal, Decimal, Decimal, Decimal, int, int]:
        self.validate()
        return (
            q8(self.mean_validation_macro_f1),
            q8(self.worst_seed_validation_macro_f1),
            -q8(self.mean_corruption_degradation_pp),
            q8(self.bottom_12_class_mean_f1),
            -int(self.model_state_tensor_bytes_fp32),
            -int(self.total_parameter_count),
        )


def dominates(a: ArchitectureSelectorRow, b: ArchitectureSelectorRow) -> bool:
    av = a.pareto_vector()
    bv = b.pareto_vector()
    comparisons = [
        (av["mean_validation_macro_f1"], bv["mean_validation_macro_f1"], "max"),
        (av["worst_seed_validation_macro_f1"], bv["worst_seed_validation_macro_f1"], "max"),
        (av["bottom_12_class_mean_f1"], bv["bottom_12_class_mean_f1"], "max"),
        (av["mean_corruption_degradation_pp"], bv["mean_corruption_degradation_pp"], "min"),
        (av["model_state_tensor_bytes_fp32"], bv["model_state_tensor_bytes_fp32"], "min"),
    ]
    no_worse = all((x >= y if direction == "max" else x <= y) for x, y, direction in comparisons)
    strictly_better = any((x > y if direction == "max" else x < y) for x, y, direction in comparisons)
    return no_worse and strictly_better


def pareto_frontier(rows: Iterable[ArchitectureSelectorRow]) -> list[ArchitectureSelectorRow]:
    rows = list(rows)
    if {row.family for row in rows} != set(FAMILIES) or len(rows) != len(FAMILIES):
        raise ValueError("selector requires exactly one row for each of R04,R06,R07,R13")
    for row in rows:
        row.validate()
    return [row for row in rows if not any(other.family != row.family and dominates(other, row) for other in rows)]


def select_journal_primary(rows: Iterable[ArchitectureSelectorRow]) -> dict[str, Any]:
    rows = list(rows)
    frontier = pareto_frontier(rows)
    ordered_frontier = sorted(frontier, key=lambda row: row.family)
    if len(frontier) == 1:
        winner = frontier[0]
        return {
            "status": "SELECTED",
            "selection_stage": "pareto_single_nondominated",
            "journal_primary_family": winner.family,
            "co_primary_families": [],
            "pareto_frontier": [winner.family],
        }
    best_key = max(row.lexicographic_key() for row in frontier)
    tied = sorted(row.family for row in frontier if row.lexicographic_key() == best_key)
    if len(tied) == 1:
        return {
            "status": "SELECTED",
            "selection_stage": "frozen_lexicographic",
            "journal_primary_family": tied[0],
            "co_primary_families": [],
            "pareto_frontier": [row.family for row in ordered_frontier],
        }
    return {
        "status": "CO_PRIMARY_TIE",
        "selection_stage": "exact_selector_precision_tie",
        "journal_primary_family": None,
        "co_primary_families": tied,
        "pareto_frontier": [row.family for row in ordered_frontier],
    }


def architecture_summary_from_evidence(
    *,
    family: str,
    seed_metrics: dict[str, dict[str, float]],
    class_f1_by_seed: dict[str, list[float]],
    corruptions_by_seed: dict[str, dict[str, dict[str, float]]],
    model_state_tensor_bytes_fp32: int,
    total_parameters: int,
) -> dict[str, Any]:
    if set(seed_metrics) != set(SEED_LABELS):
        raise ValueError("seed metrics must contain exactly S1,S2,S3")
    macro = [float(seed_metrics[label]["validation_macro_f1"]) for label in SEED_LABELS]
    balanced = [float(seed_metrics[label]["validation_balanced_accuracy"]) for label in SEED_LABELS]
    nll = [float(seed_metrics[label]["validation_nll"]) for label in SEED_LABELS]
    tail = class_tail_summary(class_f1_by_seed)
    corruption = corruption_summary(
        {label: float(seed_metrics[label]["validation_macro_f1"]) for label in SEED_LABELS},
        corruptions_by_seed,
    )
    row = ArchitectureSelectorRow(
        family=family,
        mean_validation_macro_f1=arithmetic_mean(macro),
        worst_seed_validation_macro_f1=min(macro),
        bottom_12_class_mean_f1=float(tail["bottom_12_class_mean_f1"]),
        mean_corruption_degradation_pp=float(corruption["mean_corruption_degradation_pp"]),
        model_state_tensor_bytes_fp32=int(model_state_tensor_bytes_fp32),
        total_parameter_count=int(total_parameters),
        sample_sd_validation_macro_f1=sample_sd(macro),
        mean_balanced_accuracy=arithmetic_mean(balanced),
        mean_validation_nll=arithmetic_mean(nll),
    )
    row.validate()
    return {
        "selector_row": row,
        "class_tail": tail,
        "corruption": corruption,
    }
