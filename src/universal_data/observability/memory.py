"""Memory accounting helpers.

Two different numbers matter when processing data:

* the *logical* size of a DataFrame (what pandas reports), and
* the *resident* size of the process (what the operating system sees).

The first tells us whether a dataset fits in a chunk; the second is what makes a
machine start swapping.  Both are exposed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:  # psutil is a hard dependency, but the toolkit must not die without it
    import psutil
except ImportError:  # pragma: no cover - exercised only on a broken install
    psutil = None

BYTES_PER_MB = 1024 * 1024


def dataframe_memory_mb(frame: pd.DataFrame, *, deep: bool = True) -> float:
    """Memory held by *frame* in megabytes.

    ``deep=True`` walks Python string objects, which is slower but is the only
    honest number for text-heavy datasets.
    """
    if frame.empty and frame.columns.empty:
        return 0.0
    return float(frame.memory_usage(deep=deep).sum()) / BYTES_PER_MB


def estimate_row_size_bytes(frame: pd.DataFrame) -> float:
    """Average bytes per row, used to pick a chunk size."""
    if len(frame) == 0:
        return 0.0
    return float(frame.memory_usage(deep=True).sum()) / len(frame)


def suggest_chunk_size(
    sample: pd.DataFrame, target_mb: float = 128.0, *, minimum: int = 1_000, maximum: int = 1_000_000
) -> int:
    """Pick a chunk size that keeps one chunk near *target_mb*.

    A sample of the dataset (the first few thousand rows) is enough: row width
    is far more stable than row count.
    """
    row_size = estimate_row_size_bytes(sample)
    if row_size <= 0:
        return minimum
    rows = int((target_mb * BYTES_PER_MB) / row_size)
    return max(minimum, min(maximum, rows))


@dataclass
class MemorySnapshot:
    process_mb: float
    available_mb: float
    percent_used: float


class MemoryTracker:
    """Tracks resident memory of the current process."""

    def __init__(self) -> None:
        self._peak = 0.0
        self._process = psutil.Process() if psutil is not None else None

    def process_memory_mb(self) -> float:
        if self._process is None:  # pragma: no cover - fallback path
            return 0.0
        value = self._process.memory_info().rss / BYTES_PER_MB
        self._peak = max(self._peak, value)
        return value

    def peak_memory_mb(self) -> float:
        self.process_memory_mb()
        return self._peak

    def snapshot(self) -> MemorySnapshot:
        if psutil is None:  # pragma: no cover - fallback path
            return MemorySnapshot(0.0, 0.0, 0.0)
        virtual = psutil.virtual_memory()
        return MemorySnapshot(
            process_mb=self.process_memory_mb(),
            available_mb=virtual.available / BYTES_PER_MB,
            percent_used=virtual.percent,
        )

    def check_headroom(self, required_mb: float, *, safety_factor: float = 2.0) -> str | None:
        """Return a warning message when *required_mb* looks risky, else ``None``.

        The safety factor accounts for pandas copying data during most
        operations: a 1 GB frame briefly needs about 2 GB to be transformed.
        """
        if psutil is None:  # pragma: no cover - fallback path
            return None
        available = psutil.virtual_memory().available / BYTES_PER_MB
        needed = required_mb * safety_factor
        if needed > available:
            return (
                f"Dataset needs roughly {needed:,.0f} MB with copy overhead but only "
                f"{available:,.0f} MB is available; consider chunked processing"
            )
        return None
