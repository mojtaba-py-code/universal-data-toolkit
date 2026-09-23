"""Example 5 - a file larger than the chunk size, streamed to Parquet.

    python examples/05_large_csv_chunked.py --rows 500000

Memory stays proportional to the chunk size rather than to the file, which is
what makes a multi-gigabyte CSV workable on an ordinary laptop.  Steps that need
the whole dataset at once (global deduplication, sorting) cannot run in this
mode - the example uses per-row work only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from universal_data import Dataset, Pipeline, configure_logging
from universal_data.generator.synthetic import QualityIssues, SyntheticDataGenerator
from universal_data.observability.memory import MemoryTracker, suggest_chunk_size

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"


def build_source(rows: int) -> Path:
    path = OUTPUT_DIR / "large_orders.csv"
    if path.exists():
        return path
    print(f"Generating {rows:,} rows ...")
    frame = SyntheticDataGenerator(seed=11).orders(rows, issues=QualityIssues(missing=0.02))
    Dataset(frame, name="orders").to_csv(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--chunk-size", type=int, default=0, help="0 picks a size automatically")
    args = parser.parse_args()

    configure_logging("INFO")
    source = build_source(args.rows)
    target = OUTPUT_DIR / "large_orders.parquet"

    sample = Dataset.read(source, nrows=5_000)
    chunk_size = args.chunk_size or suggest_chunk_size(sample.frame, target_mb=64)
    print(f"Source: {source.stat().st_size / 1024 / 1024:.1f} MB, chunk size: {chunk_size:,} rows")

    tracker = MemoryTracker()
    before = tracker.process_memory_mb()

    result = (
        Pipeline("large_orders")
        .clean(
            convert_types={"quantity": "int", "unit_price": "float"},
            missing_values="median",
        )
        .filter("quantity > 0")
        .run_chunked(source, target, chunk_size=chunk_size)
    )

    print(result.metrics.summary())
    print(f"Process memory grew by {tracker.peak_memory_mb() - before:.1f} MB")


if __name__ == "__main__":
    main()
