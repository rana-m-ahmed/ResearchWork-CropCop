from __future__ import annotations

import os
import time

from tracka_v12_kaggle_operator_v8 import OperatorError

DEFAULT_HARD_LIMIT_SECONDS = 12 * 3600.0
DEFAULT_FINALIZATION_MARGIN_SECONDS = 3600.0
POLL_SECONDS = 30.0


def dependency_deadline_monotonic() -> float:
    """Return the global dependency deadline for this notebook session.

    Waiting stages share one deadline rather than each receiving an independent
    short timeout. The finalization margin is always preserved.
    """
    raw_started = str(os.environ.get("CROPCOP_NOTEBOOK_STARTED_MONOTONIC", "") or "").strip()
    if not raw_started:
        raise OperatorError("notebook start monotonic clock is not bound before dependency wait")
    try:
        started = float(raw_started)
        hard = float(os.environ.get("CROPCOP_NOTEBOOK_HARD_LIMIT_SECONDS", DEFAULT_HARD_LIMIT_SECONDS))
        margin = float(os.environ.get("CROPCOP_NOTEBOOK_FINALIZATION_MARGIN_SECONDS", DEFAULT_FINALIZATION_MARGIN_SECONDS))
    except ValueError as exc:
        raise OperatorError("invalid notebook timing environment") from exc
    if hard <= 0 or margin < 0 or margin >= hard:
        raise OperatorError(f"invalid notebook timing budget: hard={hard}, margin={margin}")
    return started + hard - margin


def remaining_dependency_seconds() -> float:
    return max(0.0, dependency_deadline_monotonic() - time.monotonic())


def dependency_wait_expired() -> bool:
    return remaining_dependency_seconds() <= 0.0
