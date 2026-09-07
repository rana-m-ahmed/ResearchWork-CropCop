from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ValidationMetrics:
    macro_f1: float
    balanced_accuracy: float
    nll: float
    accuracy: float
    count: int
    wall_seconds: float
    examples_per_second: float

    def selection_key(self, epoch: int) -> tuple[float, float, float, int]:
        return (self.macro_f1, self.balanced_accuracy, -self.nll, -int(epoch))


@torch.no_grad()
def evaluate_classifier(model, loader, device: torch.device, *, num_classes: int = 120) -> ValidationMetrics:
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    nll_sum = 0.0
    count = 0
    correct = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for x, y, _row_ids in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        logp = torch.log_softmax(logits, dim=1)
        nll_sum += float((-logp.gather(1, y[:, None]).sum()).item())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        count += int(y.numel())
        idx = (y * num_classes + pred).detach().cpu()
        conf += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    if count == 0:
        raise ValueError("validation loader produced zero rows")
    tp = conf.diag().to(torch.float64)
    actual = conf.sum(dim=1).to(torch.float64)
    predicted = conf.sum(dim=0).to(torch.float64)
    recall = torch.where(actual > 0, tp / actual, torch.zeros_like(tp))
    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    return ValidationMetrics(
        float(f1.mean()), float(recall.mean()), nll_sum / count,
        correct / count, count, wall, count / max(wall, 1e-12)
    )


@torch.no_grad()
def benchmark_forward(model, loader, device: torch.device, *, max_batches: int = 20) -> dict[str, float | int]:
    model.eval()
    count = 0
    batches = 0
    loader_wait = 0.0
    forward_seconds = 0.0
    it = iter(loader)
    while batches < max_batches:
        wait_start = time.perf_counter()
        try:
            x, _y, _ids = next(it)
        except StopIteration:
            break
        loader_wait += time.perf_counter() - wait_start
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - start
        count += int(x.shape[0])
        batches += 1
    return {
        "batches": batches,
        "examples": count,
        "loader_wait_seconds": loader_wait,
        "forward_seconds": forward_seconds,
        "forward_examples_per_second": count / max(forward_seconds, 1e-12),
        "end_to_end_examples_per_second": count / max(loader_wait + forward_seconds, 1e-12),
    }
