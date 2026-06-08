"""Progress logging with ETA for long-running preprocess steps."""

from __future__ import annotations

import logging
import time


def format_duration(seconds: float) -> str:
    """Human-readable duration (e.g. 58m12s, 2h5m)."""
    if seconds < 0 or seconds != seconds:  # NaN
        return "?"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


class ProgressTracker:
    """Log incremental progress with elapsed time and ETA."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        label: str,
        total: int,
        log_every: int = 1,
        unit: str = "items",
    ) -> None:
        self.logger = logger
        self.label = label
        self.total = max(total, 1)
        self.log_every = max(log_every, 1)
        self.unit = unit
        self.current = 0
        self._start = time.perf_counter()
        logger.info("%s: starting (total=%d %s)", label, total, unit)

    def tick(self, n: int = 1, *, force: bool = False) -> None:
        self.current = min(self.current + n, self.total)
        if not force and self.current % self.log_every != 0 and self.current != self.total:
            return
        self._emit()

    def finish(self, *, message: str = "complete") -> None:
        self.current = self.total
        elapsed = time.perf_counter() - self._start
        self.logger.info(
            "%s: %s %d/%d (100%%) elapsed=%s",
            self.label,
            message,
            self.total,
            self.total,
            format_duration(elapsed),
        )

    def _emit(self) -> None:
        elapsed = time.perf_counter() - self._start
        pct = 100.0 * self.current / self.total
        rate = self.current / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.current) / rate if rate > 0 else 0.0
        self.logger.info(
            "%s: %d/%d %s (%.1f%%) elapsed=%s eta=%s",
            self.label,
            self.current,
            self.total,
            self.unit,
            pct,
            format_duration(elapsed),
            format_duration(remaining),
        )
