from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .checkpointing import recover_latest, verify_selected


def completed_scientific_checkpoint_result(
    checkpoint_root: str | Path,
    *,
    expected_identity: dict[str, Any],
    locked_epochs: int,
) -> dict[str, Any] | None:
    """Recognize an already-complete scientific checkpoint without advancing science state."""
    started = time.perf_counter()
    try:
        _path, payload, recovery = recover_latest(
            checkpoint_root,
            expected_identity=expected_identity,
        )
    except Exception:
        return None

    if not recovery or recovery.get("candidate") != "latest":
        return None
    if int(payload.get("epoch", -1)) < int(locked_epochs):
        return None
    if int(payload.get("batch_in_epoch", -1)) != 0:
        return None
    data_order = payload.get("data_order_state") or {}
    if (
        int(data_order.get("epoch", -1)) < int(locked_epochs)
        or int(data_order.get("next_batch_in_epoch", -1)) != 0
    ):
        return None

    optimizer_step = int(payload.get("optimizer_step", -1))
    if optimizer_step <= 0:
        return None
    selection_state = payload.get("selection_state") or {}
    history = selection_state.get("history")
    best = selection_state.get("best")
    if not isinstance(history, list) or len(history) < int(locked_epochs) or not isinstance(best, dict):
        return None
    if "epoch" not in best or not isinstance(best.get("metrics"), dict):
        return None
    selected_sha = str(best.get("checkpoint_sha256", ""))
    if len(selected_sha) != 64:
        return None

    verify_selected(
        checkpoint_root,
        expected_identity=expected_identity,
        expected_sha256=selected_sha,
    )
    recovered = recovery.get("recovered") or {}
    latest_sha = str(recovered.get("sha256", ""))
    if len(latest_sha) != 64:
        return None

    elapsed = max(0.0, time.perf_counter() - started)
    return {
        "mode": "scientific_terminal_recovery",
        "terminal_checkpoint_recovery": True,
        "planned_rollover": False,
        "optimizer_steps_segment": 0,
        "optimizer_step_total": optimizer_step,
        "examples_segment": 0,
        "examples_total": int(payload.get("examples_seen", 0)),
        "wall_seconds_segment": elapsed,
        "sec_per_optimizer_step": 0.0,
        "examples_per_second": 0.0,
        "dataloader_wait_seconds": 0.0,
        "dataloader_examples_per_wait_second": 0.0,
        "peak_gpu_memory_bytes": 0,
        "checkpoint_save_seconds": 0.0,
        "checkpoint_load_seconds": elapsed,
        "history": list(history),
        "selected_epoch": int(best["epoch"]),
        "selected_metrics": dict(best["metrics"]),
        "selected_checkpoint_sha256": selected_sha,
        "latest_checkpoint_sha256": latest_sha,
        "validation_forward_benchmark": None,
        "recovery_events": [
            recovery,
            {
                "event": "TERMINAL_COMPLETED_CHECKPOINT_RECOVERY",
                "locked_epochs": int(locked_epochs),
                "recovered_epoch": int(payload["epoch"]),
                "optimizer_steps_advanced": 0,
            },
        ],
    }
