"""Benchmarks behind ``data-tool benchmark``.

Four scenarios over the same generated dataset:

* ``row_wise`` - a calculated column built with ``DataFrame.apply`` (the usual
  first draft);
* ``vectorised`` - the same column through the safe expression engine, which
  compiles down to pandas/NumPy operations;
* ``in_memory`` - read, clean and write the whole file at once;
* ``chunked`` - the same work streamed in chunks.

The numbers to compare are throughput (row_wise vs vectorised) and peak memory
(in_memory vs chunked).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from universal_data.core.dataset import Dataset
from universal_data.generator.synthetic import QualityIssues, SyntheticDataGenerator
from universal_data.observability.memory import MemoryTracker
from universal_data.pipeline.pipeline import Pipeline
from universal_data.rules.expression import SafeExpression


def _timed(label: str, rows: int, function: Any) -> dict[str, Any]:
    tracker = MemoryTracker()
    before = tracker.process_memory_mb()
    started = time.perf_counter()
    function()
    elapsed = time.perf_counter() - started
    peak = tracker.peak_memory_mb()
    return {
        "scenario": label,
        "seconds": elapsed,
        "throughput": rows / elapsed if elapsed else 0.0,
        "peak_mb": max(peak - before, 0.0),
    }


def run_benchmarks(
    *, rows: int = 200_000, chunk_size: int = 50_000, output_dir: Path = Path("benchmark_output")
) -> list[dict[str, Any]]:
    """Generate a dataset and time the four scenarios."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "benchmark_input.csv"

    generator = SyntheticDataGenerator(seed=7)
    frame = generator.orders(rows, issues=QualityIssues(missing=0.02, duplicates=0.01))
    Dataset(frame, name="orders").to_csv(source)

    dataset = Dataset.read(source)
    expression = SafeExpression("quantity * unit_price * (1 - discount_pct / 100)")

    return [
        _timed(
            "row_wise apply",
            len(dataset),
            lambda: dataset.frame.apply(
                lambda row: (row["quantity"] or 0)
                * (row["unit_price"] or 0)
                * (1 - (row["discount_pct"] or 0) / 100),
                axis=1,
            ),
        ),
        _timed("vectorised expression", len(dataset), lambda: expression.evaluate(dataset.frame)),
        _timed(
            "in_memory pipeline",
            len(dataset),
            lambda: (
                Pipeline("bench_memory")
                .read(source)
                .clean(missing_values="median", remove_duplicates=True)
                .write(output_dir / "bench_memory.parquet")
                .run()
            ),
        ),
        _timed(
            "chunked pipeline",
            len(dataset),
            lambda: (
                Pipeline("bench_chunked")
                .clean(missing_values="median")
                .run_chunked(
                    source, output_dir / "bench_chunked.parquet", chunk_size=chunk_size
                )
            ),
        ),
    ]
