from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def selection_key(metrics: dict[str, float], epoch: int) -> tuple[float, float, float, int]:
    return (
        float(metrics["validation_macro_f1"]),
        float(metrics["validation_balanced_accuracy"]),
        -float(metrics["validation_nll"]),
        -int(epoch),
    )


@dataclass
class SelectionState:
    history: list[dict[str, Any]] = field(default_factory=list)
    best: dict[str, Any] | None = None

    def consider(self, summary: dict[str, Any]) -> bool:
        epoch = int(summary["epoch"])
        metrics = {
            "validation_macro_f1": float(summary["validation_macro_f1"]),
            "validation_balanced_accuracy": float(summary["validation_balanced_accuracy"]),
            "validation_nll": float(summary["validation_nll"]),
            "validation_accuracy": float(summary.get("validation_accuracy", float("nan"))),
        }
        key = selection_key(metrics, epoch)
        improved = self.best is None or tuple(self.best["selection_key"]) < key
        if improved:
            self.best = {
                "epoch": epoch,
                "metrics": dict(summary),
                "selection_key": list(key),
                "checkpoint_sha256": None,
                "checkpoint_relative_path": None,
            }
        self.history.append(dict(summary))
        return improved

    def bind_selected_checkpoint(self, *, sha256: str, relative_path: str) -> None:
        if self.best is None:
            raise ValueError("cannot bind a selected checkpoint before a best epoch exists")
        self.best["checkpoint_sha256"] = sha256
        self.best["checkpoint_relative_path"] = relative_path

    def to_dict(self) -> dict[str, Any]:
        return {"history": self.history, "best": self.best}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SelectionState":
        payload = payload or {}
        return cls(history=list(payload.get("history", [])), best=payload.get("best"))
