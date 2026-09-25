from __future__ import annotations

import csv
import math
import platform
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .trackb_eaai_common import (
    CHECKPOINTS,
    TrackBEAAIError,
    sha256_file,
)


from .data import ctc_v2_eval_transform

def _checkpoint_state(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in (
            "student",
            "state_dict",
            "model",
            "model_state_dict",
        ):
            state = payload.get(key)
            if isinstance(state, dict):
                return state
        if payload and all(
            hasattr(value, "shape")
            for value in payload.values()
        ):
            return payload
    raise TrackBEAAIError(
        "checkpoint does not expose a supported state dict"
    )


def load_model(
    checkpoint: Path,
    seed: str,
    device: str,
):
    import torch
    import torchvision
    from torchvision.models import convnext_tiny

    version = torchvision.__version__.split("+", 1)[0]
    if version != "0.27.1":
        raise TrackBEAAIError(
            f"torchvision version drift: expected 0.27.1, got {version}"
        )

    expected = CHECKPOINTS[seed]
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise TrackBEAAIError(
            f"{seed} checkpoint hash mismatch: "
            f"expected={expected}, actual={actual}"
        )

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = convnext_tiny(
        weights=None,
        num_classes=120,
    )
    model.load_state_dict(
        _checkpoint_state(payload),
        strict=True,
    )
    model.eval().to(device)
    return model


class ExternalDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        root: Path,
        class_map: dict[str, int],
    ):
        self.rows = rows
        self.root = root.resolve()
        self.class_map = class_map

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = self.rows[index]
        path = (
            self.root / row["relative_path"]
        ).resolve()
        if self.root not in path.parents:
            raise TrackBEAAIError(
                "external path escapes data root: "
                f"{row['relative_path']}"
            )
        with Image.open(path) as image:
            x = ctc_v2_eval_transform(image)

        return (
            x,
            int(
                self.class_map[
                    row["target_class_name"]
                ]
            ),
            row["row_id"],
        )


def run_three_seed_inference(
    models: dict[str, Any],
    rows: list[dict[str, Any]],
    root: Path,
    class_map: dict[str, int],
    *,
    device: str,
    batch_size: int,
    workers: int,
) -> dict[str, list[dict[str, Any]]]:
    """Decode/transform each external image once, then evaluate S1/S2/S3 in FP32."""
    import torch
    from torch.utils.data import DataLoader

    if set(models) != {"S1", "S2", "S3"}:
        raise TrackBEAAIError(
            "three-seed inference requires exactly S1/S2/S3 models"
        )

    dataset = ExternalDataset(rows, root, class_map)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=device.startswith("cuda"),
        persistent_workers=bool(int(workers) > 0),
    )
    by_id = {row["row_id"]: row for row in rows}
    output = {seed: [] for seed in ("S1", "S2", "S3")}

    for model in models.values():
        model.eval()

    total_batches = math.ceil(len(dataset) / int(batch_size))
    with torch.inference_mode():
        for batch_index, (x, target, row_ids) in enumerate(loader):
            x_device = x.to(device, non_blocking=True)
            targets = target.cpu().tolist()
            predictions_by_seed: dict[str, list[int]] = {}

            for seed in ("S1", "S2", "S3"):
                logits = models[seed](x_device)
                if logits.ndim != 2 or logits.shape[1] != 120:
                    raise TrackBEAAIError(
                        f"{seed} native classifier output must be [N,120], "
                        f"got {tuple(logits.shape)}"
                    )
                predictions_by_seed[seed] = (
                    logits.argmax(dim=1).cpu().tolist()
                )
                del logits

            for row_index, (row_id, target_idx) in enumerate(
                zip(row_ids, targets)
            ):
                source = by_id[str(row_id)]
                common = {
                    "row_id": str(row_id),
                    "relative_path": source["relative_path"],
                    "source_label": source["source_label"],
                    "target_class_name": source["target_class_name"],
                    "target_class_index": int(target_idx),
                }
                for seed in ("S1", "S2", "S3"):
                    pred_idx = int(predictions_by_seed[seed][row_index])
                    output[seed].append(
                        {
                            **common,
                            "predicted_class_index": pred_idx,
                            "correct": int(target_idx) == pred_idx,
                        }
                    )

            if (batch_index + 1) % 100 == 0 or (batch_index + 1) == total_batches:
                print(
                    "three-seed inference batches completed: "
                    f"{batch_index + 1:,}/{total_batches:,}",
                    flush=True,
                )

    expected_ids = [row["row_id"] for row in rows]
    for seed in ("S1", "S2", "S3"):
        observed_ids = [row["row_id"] for row in output[seed]]
        if observed_ids != expected_ids:
            raise TrackBEAAIError(
                f"{seed} prediction row order/identity mismatch"
            )

    return output

def metrics(
    rows: Iterable[dict[str, Any]],
    mapped_indices: Iterable[int],
) -> dict[str, Any]:
    rows = list(rows)
    labels = tuple(
        sorted(
            {
                int(value)
                for value in mapped_indices
            }
        )
    )
    if not rows or not labels:
        raise TrackBEAAIError(
            "cannot score empty rows/scope"
        )

    label_set = set(labels)
    correct = 0
    outside = 0
    per: dict[str, Any] = {}
    f1s: list[float] = []
    recalls: list[float] = []

    for row in rows:
        target = int(
            row["target_class_index"]
        )
        predicted = int(
            row["predicted_class_index"]
        )
        if (
            target not in label_set
            or not 0 <= predicted < 120
        ):
            raise TrackBEAAIError(
                "prediction/target outside "
                "frozen metric contract"
            )
        correct += int(
            target == predicted
        )
        outside += int(
            predicted not in label_set
        )

    for cls in labels:
        tp = sum(
            int(
                int(row["target_class_index"]) == cls
                and int(
                    row[
                        "predicted_class_index"
                    ]
                )
                == cls
            )
            for row in rows
        )
        fp = sum(
            int(
                int(row["target_class_index"]) != cls
                and int(
                    row[
                        "predicted_class_index"
                    ]
                )
                == cls
            )
            for row in rows
        )
        fn = sum(
            int(
                int(row["target_class_index"]) == cls
                and int(
                    row[
                        "predicted_class_index"
                    ]
                )
                != cls
            )
            for row in rows
        )
        support = sum(
            int(
                int(
                    row["target_class_index"]
                )
                == cls
            )
            for row in rows
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )
        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        per[str(cls)] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1s.append(f1)
        recalls.append(recall)

    confusion = {
        str(target): {
            str(pred): 0
            for pred in list(labels) + [-1]
        }
        for target in labels
    }
    for row in rows:
        target = int(
            row["target_class_index"]
        )
        predicted = int(
            row["predicted_class_index"]
        )
        bucket = (
            predicted
            if predicted in label_set
            else -1
        )
        confusion[str(target)][
            str(bucket)
        ] += 1

    return {
        "row_count": len(rows),
        "mapped_class_indices": list(labels),
        "accuracy": correct / len(rows),
        "macro_f1": sum(f1s) / len(f1s),
        "balanced_accuracy": (
            sum(recalls) / len(recalls)
        ),
        "out_of_mapped_scope_prediction_rate": (
            outside / len(rows)
        ),
        "per_class": per,
        "confusion_matrix_mapped_plus_oos": confusion,
    }


def paired_stratified_bootstrap(
    seed_rows: dict[
        str,
        list[dict[str, Any]],
    ],
    *,
    labels: Iterable[int],
    replicates: int,
    rng_seed: int,
) -> dict[str, Any]:
    import numpy as np

    labels = tuple(
        sorted(int(value) for value in labels)
    )
    seeds = ("S1", "S2", "S3")
    reference = [
        (
            row["row_id"],
            int(
                row["target_class_index"]
            ),
        )
        for row in seed_rows["S1"]
    ]
    by_seed = {
        seed: {
            row["row_id"]: int(
                row[
                    "predicted_class_index"
                ]
            )
            for row in seed_rows[seed]
        }
        for seed in seeds
    }

    for seed in seeds:
        observed = [
            (
                row["row_id"],
                int(
                    row[
                        "target_class_index"
                    ]
                ),
            )
            for row in seed_rows[seed]
        ]
        if observed != reference:
            raise TrackBEAAIError(
                "bootstrap requires identical "
                "ordered prediction surfaces "
                "across seeds"
            )

    grouped: dict[
        int,
        Counter[tuple[int, int, int]],
    ] = {
        cls: Counter()
        for cls in labels
    }
    for row_id, target in reference:
        grouped[target][
            tuple(
                by_seed[seed][row_id]
                for seed in seeds
            )
        ] += 1

    rng = np.random.default_rng(
        int(rng_seed)
    )
    distributions = {
        seed: []
        for seed in seeds
    }
    mean_distribution = []

    for _ in range(int(replicates)):
        confusion = {
            seed: {
                cls: Counter()
                for cls in labels
            }
            for seed in seeds
        }

        for cls in labels:
            categories = sorted(
                grouped[cls]
            )
            frequencies = np.asarray(
                [
                    grouped[cls][category]
                    for category in categories
                ],
                dtype=np.float64,
            )
            draw = rng.multinomial(
                int(frequencies.sum()),
                frequencies
                / frequencies.sum(),
            )
            for category, count in zip(
                categories,
                draw,
            ):
                if not count:
                    continue
                for seed_index, seed in enumerate(
                    seeds
                ):
                    confusion[seed][cls][
                        category[seed_index]
                    ] += int(count)

        values = []
        for seed in seeds:
            f1s = []
            for cls in labels:
                tp = confusion[seed][cls][cls]
                fp = sum(
                    confusion[seed][other][cls]
                    for other in labels
                    if other != cls
                )
                fn = (
                    sum(
                        confusion[seed][
                            cls
                        ].values()
                    )
                    - tp
                )
                precision = (
                    tp / (tp + fp)
                    if tp + fp
                    else 0.0
                )
                recall = (
                    tp / (tp + fn)
                    if tp + fn
                    else 0.0
                )
                f1s.append(
                    (
                        2.0
                        * precision
                        * recall
                        / (
                            precision
                            + recall
                        )
                    )
                    if precision + recall
                    else 0.0
                )
            value = float(
                sum(f1s) / len(f1s)
            )
            distributions[
                seed
            ].append(value)
            values.append(value)

        mean_distribution.append(
            float(sum(values) / 3.0)
        )

    def summarize(values):
        arr = np.asarray(
            values,
            dtype=np.float64,
        )
        return {
            "lower_2_5": float(
                np.quantile(arr, 0.025)
            ),
            "median": float(
                np.quantile(arr, 0.5)
            ),
            "upper_97_5": float(
                np.quantile(arr, 0.975)
            ),
        }

    return {
        "replicates": int(replicates),
        "seed": int(rng_seed),
        "per_seed_macro_f1_95pct": {
            seed: summarize(values)
            for seed, values
            in distributions.items()
        },
        "three_seed_mean_macro_f1_95pct": (
            summarize(mean_distribution)
        ),
    }


def write_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    mapped_indices: set[int],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fields = [
        "row_id",
        "relative_path",
        "source_label",
        "target_class_name",
        "target_class_index",
        "predicted_class_index",
        "correct",
        "in_mapped_scope",
    ]
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fields,
        )
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output[
                "in_mapped_scope"
            ] = (
                int(
                    output[
                        "predicted_class_index"
                    ]
                )
                in mapped_indices
            )
            writer.writerow(output)


def environment_record(
    device: str,
) -> dict[str, Any]:
    import numpy as np
    import PIL
    import torch
    import torchvision

    observed = {
        "torch": torch.__version__.split("+", 1)[0],
        "torchvision": torchvision.__version__.split("+", 1)[0],
        "numpy": np.__version__,
        "Pillow": PIL.__version__,
    }
    expected = {
        "torch": "2.12.1",
        "torchvision": "0.27.1",
        "numpy": "2.5.2",
        "Pillow": "12.3.0",
    }
    drift = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if drift:
        raise TrackBEAAIError(
            f"model-critical runtime drift: {drift}"
        )

    return {
        "python": platform.python_version(),
        **observed,
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "cuda_device_count": (
            torch.cuda.device_count()
        ),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(
                torch.cuda.device_count()
            )
        ],
        "device": device,
    }


def aggregate_three_seed(
    metric_by_seed: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    fields = (
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "out_of_mapped_scope_prediction_rate",
    )
    output = {}
    for field in fields:
        values = [
            float(
                metric_by_seed[seed][
                    field
                ]
            )
            for seed in (
                "S1",
                "S2",
                "S3",
            )
        ]
        output[field] = {
            "mean": sum(values) / 3.0,
            "sample_sd": statistics.stdev(
                values
            ),
            "min": min(values),
            "max": max(values),
        }
    return output


def write_confusion_csv(
    path: Path,
    metric: dict[str, Any],
    class_names_by_index: dict[int, str],
) -> None:
    labels = [
        int(value)
        for value
        in metric["mapped_class_indices"]
    ]
    columns = labels + [-1]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["true_class"]
            + [
                class_names_by_index[
                    value
                ]
                for value in labels
            ]
            + ["OUT_OF_MAPPED_SCOPE"]
        )
        for target in labels:
            row = metric[
                "confusion_matrix_mapped_plus_oos"
            ][str(target)]
            writer.writerow(
                [
                    class_names_by_index[
                        target
                    ]
                ]
                + [
                    row[str(predicted)]
                    for predicted
                    in columns
                ]
            )
