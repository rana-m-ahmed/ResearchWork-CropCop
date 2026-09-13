from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterable

from .data import ctc_v2_eval_transform
from .tracka_v12_analysis import (
    ROBUSTNESS_CORRUPTIONS,
    ROBUSTNESS_SEVERITIES,
    canonical_model_state_bytes_fp32,
    total_parameter_count,
)
from .tracka_v12_evidence import robustness_seed, summarize_prediction_rows

CORRUPTION_VALUES = {
    "brightness": {"1": 0.75, "2": 0.5, "3": 0.25},
    "contrast": {"1": 0.75, "2": 0.5, "3": 0.25},
    "gaussian_blur": {"1": 1.0, "2": 2.0, "3": 3.0},
    "gaussian_noise_uint8": {"1": 8.0, "2": 16.0, "3": 32.0},
    "jpeg": {"1": 70, "2": 40, "3": 20},
}


def apply_locked_corruption(image, *, stable_row_id: str, corruption: str, severity: str):
    if corruption not in ROBUSTNESS_CORRUPTIONS or severity not in ROBUSTNESS_SEVERITIES:
        raise ValueError("corruption request outside frozen Track-A protocol")
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    value = CORRUPTION_VALUES[corruption][severity]
    if corruption == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(value))
    if corruption == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(value))
    if corruption == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if corruption == "gaussian_noise_uint8":
        import numpy as np

        array = np.asarray(image, dtype=np.uint8)
        rng = np.random.Generator(np.random.PCG64(robustness_seed(stable_row_id, corruption, severity)))
        noise = rng.normal(0.0, float(value), size=array.shape)
        corrupted = np.rint(array.astype(np.float64) + noise).clip(0, 255).astype(np.uint8)
        return Image.fromarray(corrupted, mode="RGB")
    if corruption == "jpeg":
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=int(value),
            optimize=False,
            progressive=False,
            subsampling=2,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    raise AssertionError("unreachable corruption")


class LockedValidationDataset:
    def __init__(
        self,
        rows: Iterable[Any],
        image_root: str | Path,
        *,
        corruption: str | None = None,
        severity: str | None = None,
    ):
        self.rows = list(rows)
        self.image_root = Path(image_root).resolve()
        if (corruption is None) != (severity is None):
            raise ValueError("corruption and severity must both be set or both be omitted")
        if corruption is not None and (corruption not in ROBUSTNESS_CORRUPTIONS or severity not in ROBUSTNESS_SEVERITIES):
            raise ValueError("dataset corruption outside frozen protocol")
        self.corruption = corruption
        self.severity = severity

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = self.rows[int(index)]
        path = (self.image_root / row.relative_path).resolve()
        if self.image_root not in path.parents and path != self.image_root:
            raise ValueError(f"manifest path escapes image root: {row.relative_path}")
        with Image.open(path) as image:
            if self.corruption is not None:
                image = apply_locked_corruption(
                    image,
                    stable_row_id=row.stable_row_id,
                    corruption=self.corruption,
                    severity=str(self.severity),
                )
            x = ctc_v2_eval_transform(image)
        return x, int(row.class_index), str(row.stable_row_id)


def prediction_rows(model, dataset, device, *, batch_size: int = 16, num_workers: int = 4) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(int(num_workers) > 0),
        prefetch_factor=2 if int(num_workers) > 0 else None,
    )
    model = model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for x, y, row_ids in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            logp = torch.log_softmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            nll = -logp.gather(1, y[:, None]).squeeze(1)
            confidence = probs.max(dim=1).values
            for index, row_id in enumerate(row_ids):
                rows.append(
                    {
                        "stable_row_id": str(row_id),
                        "target_class_index": int(y[index].item()),
                        "predicted_class_index": int(pred[index].item()),
                        "true_class_nll": float(nll[index].item()),
                        "top1_confidence": float(confidence[index].item()),
                    }
                )
    return rows


def efficiency_evidence(model, *, input_resolution: int = 256) -> dict[str, int]:
    return {
        "total_parameter_count": int(total_parameter_count(model)),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "model_state_tensor_bytes_fp32": int(canonical_model_state_bytes_fp32(model.state_dict())),
        "input_resolution": int(input_resolution),
    }


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def clean_replay(model, rows, image_root, device, *, batch_size: int = 16, num_workers: int = 4) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = LockedValidationDataset(rows, image_root)
    predictions = prediction_rows(model, dataset, device, batch_size=batch_size, num_workers=num_workers)
    return predictions, summarize_prediction_rows(predictions)


def robustness_sweep(model, rows, image_root, device, *, batch_size: int = 16, num_workers: int = 4):
    cell_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    cell_predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for corruption in ROBUSTNESS_CORRUPTIONS:
        cell_summaries[corruption] = {}
        for severity in ROBUSTNESS_SEVERITIES:
            dataset = LockedValidationDataset(
                rows,
                image_root,
                corruption=corruption,
                severity=severity,
            )
            predictions = prediction_rows(model, dataset, device, batch_size=batch_size, num_workers=num_workers)
            summary = summarize_prediction_rows(predictions)
            summary["corruption"] = corruption
            summary["severity"] = severity
            cell_summaries[corruption][severity] = summary
            cell_predictions[(corruption, severity)] = predictions
    return cell_predictions, cell_summaries
