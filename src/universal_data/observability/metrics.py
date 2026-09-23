"""Execution metrics.

Metrics are plain dataclasses rather than a third-party client, so the core
package has no monitoring dependency.  :class:`MetricsSink` is the seam where a
Prometheus, OpenTelemetry or StatsD exporter plugs in later.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from universal_data.observability.memory import MemoryTracker


@dataclass
class StepMetrics:
    """Metrics for a single pipeline step."""

    name: str
    status: str = "pending"
    rows_in: int = 0
    rows_out: int = 0
    rows_failed: int = 0
    rows_skipped: int = 0
    duration_seconds: float = 0.0
    memory_delta_mb: float = 0.0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def throughput(self) -> float:
        """Rows processed per second (0 when the step was instantaneous)."""
        if self.duration_seconds <= 0:
            return 0.0
        return self.rows_out / self.duration_seconds

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["throughput"] = round(self.throughput, 2)
        data["duration_seconds"] = round(self.duration_seconds, 4)
        data["memory_delta_mb"] = round(self.memory_delta_mb, 3)
        return data


@dataclass
class RunMetrics:
    """Aggregated metrics for one pipeline execution."""

    execution_id: str
    pipeline: str = "pipeline"
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    steps: list[StepMetrics] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    # Chunked runs report totals explicitly; per-step counts are per chunk there.
    rows_input_total: int | None = None
    rows_output_total: int | None = None

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()

    @property
    def rows_input(self) -> int:
        if self.rows_input_total is not None:
            return self.rows_input_total
        return self.steps[0].rows_in if self.steps else 0

    @property
    def rows_output(self) -> int:
        if self.rows_output_total is not None:
            return self.rows_output_total
        for step in reversed(self.steps):
            if step.status == "success":
                return step.rows_out
        return 0

    @property
    def rows_failed(self) -> int:
        return sum(step.rows_failed for step in self.steps)

    @property
    def rows_skipped(self) -> int:
        return sum(step.rows_skipped for step in self.steps)

    @property
    def status(self) -> str:
        if any(step.status == "failed" for step in self.steps):
            return "failed"
        if not self.steps:
            return "empty"
        return "success"

    @property
    def throughput(self) -> float:
        duration = self.duration_seconds
        return self.rows_output / duration if duration > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "pipeline": self.pipeline,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": round(self.duration_seconds, 4),
            "rows_input": self.rows_input,
            "rows_output": self.rows_output,
            "rows_failed": self.rows_failed,
            "rows_skipped": self.rows_skipped,
            "throughput_rows_per_second": round(self.throughput, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "steps": [step.to_dict() for step in self.steps],
        }

    def summary(self) -> str:
        """Compact human readable summary used by the CLI."""
        lines = [
            "Processing Summary",
            "",
            f"Pipeline:        {self.pipeline}",
            f"Execution ID:    {self.execution_id}",
            f"Status:          {self.status}",
            f"Rows Input:      {self.rows_input:,}",
            f"Rows Processed:  {self.rows_output:,}",
            f"Rows Failed:     {self.rows_failed:,}",
            f"Rows Skipped:    {self.rows_skipped:,}",
            f"Duration:        {self.duration_seconds:.2f} sec",
            f"Throughput:      {self.throughput:,.0f} rows/sec",
            f"Peak Memory:     {self.peak_memory_mb:.1f} MB",
        ]
        return "\n".join(lines)


class MetricsSink(ABC):
    """Destination for finished run metrics."""

    @abstractmethod
    def emit(self, metrics: RunMetrics) -> None:
        """Publish the metrics of a completed run."""


class InMemorySink(MetricsSink):
    """Keeps runs in a list; used by tests and by the CLI summary output."""

    def __init__(self) -> None:
        self.runs: list[RunMetrics] = []

    def emit(self, metrics: RunMetrics) -> None:
        self.runs.append(metrics)


class PrometheusTextSink(MetricsSink):
    """Renders the Prometheus text exposition format.

    Writing the format by hand keeps ``prometheus_client`` out of the dependency
    list; the output can be served by any HTTP handler or written to a textfile
    collector directory.
    """

    def __init__(self, namespace: str = "universal_data") -> None:
        self.namespace = namespace
        self.lines: list[str] = []

    def emit(self, metrics: RunMetrics) -> None:
        labels = f'pipeline="{metrics.pipeline}",status="{metrics.status}"'
        samples = {
            "rows_input_total": metrics.rows_input,
            "rows_output_total": metrics.rows_output,
            "rows_failed_total": metrics.rows_failed,
            "duration_seconds": round(metrics.duration_seconds, 4),
            "peak_memory_megabytes": round(metrics.peak_memory_mb, 2),
        }
        for name, value in samples.items():
            self.lines.append(f"{self.namespace}_{name}{{{labels}}} {value}")

    def render(self) -> str:
        return "\n".join(self.lines) + ("\n" if self.lines else "")


class MetricsCollector:
    """Times steps and records their row counts."""

    def __init__(self, execution_id: str, pipeline: str = "pipeline") -> None:
        self.run = RunMetrics(execution_id=execution_id, pipeline=pipeline)
        self._memory = MemoryTracker()

    @contextmanager
    def step(self, name: str, rows_in: int = 0) -> Iterator[StepMetrics]:
        metrics = StepMetrics(name=name, rows_in=rows_in, status="running")
        self.run.steps.append(metrics)
        started = time.perf_counter()
        memory_before = self._memory.process_memory_mb()
        try:
            yield metrics
        except Exception as exc:
            metrics.status = "failed"
            metrics.error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            if metrics.status == "running":
                metrics.status = "success"
        finally:
            metrics.duration_seconds = time.perf_counter() - started
            metrics.memory_delta_mb = self._memory.process_memory_mb() - memory_before
            self.run.peak_memory_mb = max(self.run.peak_memory_mb, self._memory.peak_memory_mb())

    def finish(self) -> RunMetrics:
        self.run.finished_at = datetime.now(UTC)
        self.run.peak_memory_mb = max(self.run.peak_memory_mb, self._memory.peak_memory_mb())
        return self.run
