from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass

START_ENV = "CROPCOP_NOTEBOOK_STARTED_MONOTONIC"
HARD_LIMIT_ENV = "CROPCOP_NOTEBOOK_HARD_LIMIT_SECONDS"
FINALIZATION_MARGIN_ENV = "CROPCOP_NOTEBOOK_FINALIZATION_MARGIN_SECONDS"


@dataclass
class SessionBudget:
    hard_limit_seconds: float = 12 * 3600
    finalization_margin_seconds: float = 3600
    started_monotonic: float | None = None
    signal_reason: str | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        require_global_clock: bool = False,
        hard_limit_seconds: float = 12 * 3600,
        finalization_margin_seconds: float = 3600,
    ) -> "SessionBudget":
        start_raw = os.environ.get(START_ENV, "").strip()
        if require_global_clock and not start_raw:
            raise RuntimeError(f"notebook-global session clock missing: {START_ENV}")
        start = float(start_raw) if start_raw else None
        hard = float(os.environ.get(HARD_LIMIT_ENV, hard_limit_seconds))
        margin = float(os.environ.get(FINALIZATION_MARGIN_ENV, finalization_margin_seconds))
        if hard <= 0 or margin < 0 or margin >= hard:
            raise RuntimeError("invalid notebook-global hard-limit/finalization-margin configuration")
        return cls(hard_limit_seconds=hard, finalization_margin_seconds=margin, started_monotonic=start)

    @classmethod
    def establish_global_clock(
        cls,
        *,
        hard_limit_seconds: float = 12 * 3600,
        finalization_margin_seconds: float = 3600,
    ) -> "SessionBudget":
        if START_ENV not in os.environ:
            os.environ[START_ENV] = repr(time.monotonic())
        os.environ.setdefault(HARD_LIMIT_ENV, repr(float(hard_limit_seconds)))
        os.environ.setdefault(FINALIZATION_MARGIN_ENV, repr(float(finalization_margin_seconds)))
        return cls.from_environment(require_global_clock=True)

    def start(self) -> None:
        if self.started_monotonic is None:
            self.started_monotonic = time.monotonic()

    @property
    def elapsed(self) -> float:
        if self.started_monotonic is None:
            return 0.0
        return max(0.0, time.monotonic() - self.started_monotonic)

    @property
    def safe_deadline_seconds(self) -> float:
        return max(0.0, self.hard_limit_seconds - self.finalization_margin_seconds)

    @property
    def remaining_safe_seconds(self) -> float:
        return max(0.0, self.safe_deadline_seconds - self.elapsed)

    @property
    def remaining_hard_seconds(self) -> float:
        return max(0.0, self.hard_limit_seconds - self.elapsed)

    def can_start_phase(
        self,
        estimated_phase_seconds: float,
        *,
        estimated_checkpoint_seconds: float = 0.0,
        estimated_sync_seconds: float = 0.0,
        extra_reserve_seconds: float = 0.0,
    ) -> bool:
        required = (
            max(0.0, estimated_phase_seconds)
            + max(0.0, estimated_checkpoint_seconds)
            + max(0.0, estimated_sync_seconds)
            + max(0.0, extra_reserve_seconds)
        )
        return required < self.remaining_safe_seconds

    def should_finalize(
        self,
        *,
        estimated_checkpoint_seconds: float = 0.0,
        estimated_sync_seconds: float = 0.0,
    ) -> bool:
        if self.signal_reason is not None:
            return True
        reserve = max(0.0, estimated_checkpoint_seconds) + max(0.0, estimated_sync_seconds)
        return self.elapsed + reserve >= self.safe_deadline_seconds

    def snapshot(self) -> dict:
        return {
            "started_monotonic": self.started_monotonic,
            "hard_limit_seconds": self.hard_limit_seconds,
            "finalization_margin_seconds": self.finalization_margin_seconds,
            "elapsed_seconds": self.elapsed,
            "remaining_safe_seconds": self.remaining_safe_seconds,
            "remaining_hard_seconds": self.remaining_hard_seconds,
        }

    def install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            try:
                name = signal.Signals(signum).name
            except Exception:
                name = str(signum)
            self.signal_reason = name

        for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
            if sig is not None:
                try:
                    signal.signal(sig, _handler)
                except (ValueError, OSError):
                    pass
