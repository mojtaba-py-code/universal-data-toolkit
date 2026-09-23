"""Run the performance benchmarks and write the results as Markdown.

    python benchmarks/run_benchmarks.py --rows 200000

Compares row-wise processing against the vectorised expression engine, and
whole-file processing against chunked streaming.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from universal_data.cli.benchmarks import run_benchmarks

ROOT = Path(__file__).resolve().parent.parent


def render(results: list[dict[str, object]], rows: int) -> str:
    header = (
        f"# Benchmark results\n\n"
        f"- Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"- Rows: {rows:,}\n"
        f"- Python: {sys.version.split()[0]}\n"
        f"- Platform: {platform.platform()}\n"
        f"- CPU: {platform.processor() or 'unknown'}\n\n"
        "| Scenario | Seconds | Rows/sec | Peak memory (MB) |\n"
        "| --- | ---: | ---: | ---: |\n"
    )
    body = "".join(
        f"| {item['scenario']} | {item['seconds']:.3f} | "
        f"{item['throughput']:,.0f} | {item['peak_mb']:.1f} |\n"
        for item in results
    )
    return header + body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results.md")
    args = parser.parse_args()

    results = run_benchmarks(
        rows=args.rows,
        chunk_size=args.chunk_size,
        output_dir=ROOT / "output" / "benchmarks",
    )
    markdown = render(results, args.rows)
    print(markdown)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
