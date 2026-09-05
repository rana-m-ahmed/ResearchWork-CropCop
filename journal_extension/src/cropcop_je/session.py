from __future__ import annotations

import signal
import time
from dataclasses import dataclass


@dataclass
class SessionBudget:
    hard_limit_seconds: float = 12 * 3600
    finalization_margin_seconds: float = 3600
    started_monotonic: float | None = None
    signal_reason: str | None = None

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
