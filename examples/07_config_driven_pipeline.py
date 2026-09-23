"""Example 7 - run a YAML-defined pipeline from Python.

    python examples/07_config_driven_pipeline.py

Equivalent to::

    data-tool run configs/customer_pipeline.yaml

Use this form when the pipeline definition belongs to the operations team and
the Python process only needs to trigger it and inspect the result.
"""

from __future__ import annotations

from pathlib import Path

from universal_data import configure_logging
from universal_data.observability.metrics import PrometheusTextSink
from universal_data.pipeline.config import PipelineConfig

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "customer_pipeline.yaml"


def main() -> None:
    configure_logging("INFO")

    sink = PrometheusTextSink()
    pipeline = PipelineConfig.from_file(CONFIG, workspace=ROOT).build()
    pipeline.metrics_sink = sink
    print(f"Pipeline: {pipeline!r}\n")

    result = pipeline.run()
    print(result.metrics.summary())

    if result.validation:
        print()
        print(result.validation.summary())
    if result.rule_evaluation:
        print()
        for outcome in result.rule_evaluation.outcomes:
            print(f"  {outcome.name:<20} {outcome.pass_rate * 100:6.1f}% pass")
    if result.quality:
        print()
        print(result.quality.summary())

    print("\nPrometheus exposition:")
    print(sink.render())

    for artefact in result.context.artifacts.get("outputs", []):
        print(f"Output: {artefact}")
    for report in result.context.artifacts.get("reports", []):
        print(f"Report: {report}")


if __name__ == "__main__":
    main()
