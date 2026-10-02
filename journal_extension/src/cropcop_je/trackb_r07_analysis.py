from __future__ import annotations

from typing import Any

from .trackb_r07 import TrackBError, mapped_scope_metrics, three_seed_summary, validate_same_prediction_surface

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 409883112


def _rows_to_arrays(seed_rows: dict[str, list[dict[str, Any]]]):
    import numpy as np

    validate_same_prediction_surface(seed_rows)
    row_ids = [str(r["stable_row_id"]) for r in seed_rows["S1"]]
    targets = np.asarray([int(r["target_class_index"]) for r in seed_rows["S1"]], dtype=np.int16)
    predictions = {
        seed: np.asarray([int(r["predicted_class_index"]) for r in seed_rows[seed]], dtype=np.int16)
        for seed in ("S1", "S2", "S3")
    }
    return row_ids, targets, predictions


def _macro_f1_from_arrays(targets, predictions, labels) -> float:
    values = []
    for cls in labels:
        truth = targets == cls
        pred = predictions == cls
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(values) / len(values))


def bootstrap_three_seed_macro_f1(
    seed_rows: dict[str, list[dict[str, Any]]],
    mapped_class_indices,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    chunk_replicates: int = 64,
) -> dict[str, Any]:
    import numpy as np

    row_ids, targets, predictions = _rows_to_arrays(seed_rows)
    labels = tuple(sorted({int(x) for x in mapped_class_indices}))
    if not labels:
        raise TrackBError("bootstrap mapped class set is empty")
    if set(np.unique(targets).tolist()) != set(labels):
        raise TrackBError("bootstrap targets do not match the frozen mapped class set")

    class_indices = {cls: np.flatnonzero(targets == cls) for cls in labels}
    if any(len(indices) < 1 for indices in class_indices.values()):
        raise TrackBError("bootstrap class has zero family representatives")

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    per_seed_replicates = {s: np.empty(int(replicates), dtype=np.float64) for s in ("S1", "S2", "S3")}
    family_mean = np.empty(int(replicates), dtype=np.float64)

    for start in range(0, int(replicates), int(chunk_replicates)):
        stop = min(int(replicates), start + int(chunk_replicates))
        chunk = stop - start
        sampled_by_class = {
            cls: indices[rng.integers(0, len(indices), size=(chunk, len(indices)), endpoint=False)]
            for cls, indices in class_indices.items()
        }
        for rep in range(chunk):
            sampled = np.concatenate([sampled_by_class[cls][rep] for cls in labels])
            t = targets[sampled]
            seed_values = []
            for seed_label in ("S1", "S2", "S3"):
                value = _macro_f1_from_arrays(t, predictions[seed_label][sampled], labels)
                per_seed_replicates[seed_label][start + rep] = value
                seed_values.append(value)
            family_mean[start + rep] = sum(seed_values) / 3.0

    seed_metrics = {seed_label: mapped_scope_metrics(seed_rows[seed_label], labels) for seed_label in ("S1", "S2", "S3")}
    summary = three_seed_summary(seed_metrics)
    low, high = np.percentile(family_mean, [2.5, 97.5])
    result = {
        "replicates": int(replicates),
        "seed": int(seed),
        "sampling_unit": "family_representative",
        "stratified_within_mapped_class": True,
        "same_resample_indices_for_all_three_seeds": True,
        "mapped_class_indices": list(labels),
        "class_family_counts": {str(cls): int(len(class_indices[cls])) for cls in labels},
        "seed_point_metrics": seed_metrics,
        "three_seed": {
            **summary,
            "bootstrap_mean_macro_f1_ci95_percentile": [float(low), float(high)],
        },
    }
    return result
