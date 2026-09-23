"""The pipeline engine.

A pipeline is an ordered list of named steps, each one a function from
``Dataset`` to ``Dataset``.  The engine adds what a script normally lacks:
an execution id, per-step metrics, structured logging, conditional steps and a
choice between failing fast and skipping a broken step.

The builder methods return ``self`` so a pipeline reads top to bottom::

    result = (
        Pipeline("customers")
        .read("customers.csv")
        .clean(missing_values="median", remove_duplicates=True)
        .validate(schema)
        .write("customers.parquet")
        .run()
    )
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from universal_data.core.dataset import Dataset
from universal_data.core.exceptions import ConfigurationError, PipelineError
from universal_data.core.types import PathLike
from universal_data.observability.logging import get_logger, log_context, new_execution_id
from universal_data.observability.memory import MemoryTracker
from universal_data.observability.metrics import MetricsCollector, MetricsSink, RunMetrics
from universal_data.security.paths import PathPolicy

if TYPE_CHECKING:  # pragma: no cover
    from universal_data.cleaning.engine import CleaningConfig
    from universal_data.enrichment.enrichers import Enricher
    from universal_data.masking.maskers import MaskingPolicy
    from universal_data.quality.metrics import QualityReport
    from universal_data.quality.report import QualityDocument
    from universal_data.rules.engine import RuleEngine, RuleEvaluation
    from universal_data.schema.model import Schema
    from universal_data.transformation.base import Transformation
    from universal_data.validation.results import ValidationResult

logger = get_logger(__name__)

StepFunction = Callable[[Dataset, "PipelineContext"], Dataset]


@dataclass
class PipelineContext:
    """State shared by every step of one execution."""

    execution_id: str
    pipeline: str
    policy: PathPolicy
    metrics: MetricsCollector
    validation: ValidationResult | None = None
    rule_evaluation: RuleEvaluation | None = None
    quality: QualityReport | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """One unit of work."""

    name: str
    run: StepFunction
    condition: Callable[[Dataset], bool] | None = None
    optional: bool = False


@dataclass
class PipelineResult:
    """What a run produced."""

    dataset: Dataset
    metrics: RunMetrics
    context: PipelineContext
    skipped: list[str] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return not self.failures

    @property
    def validation(self) -> ValidationResult | None:
        return self.context.validation

    @property
    def rule_evaluation(self) -> RuleEvaluation | None:
        return self.context.rule_evaluation

    @property
    def quality(self) -> QualityReport | None:
        return self.context.quality

    def document(self) -> QualityDocument:
        """Bundle every report produced during the run."""
        from universal_data.quality.report import QualityDocument

        return QualityDocument(
            dataset=self.dataset.name,
            source=self.dataset.source,
            profile=self.dataset.profile(),
            validation=self.validation,
            rules=self.rule_evaluation,
            quality=self.quality,
            cleaning=self.dataset.metadata.get("cleaning_report"),
            metrics=self.metrics,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "succeeded": self.succeeded,
            "rows": self.dataset.n_rows,
            "columns": self.dataset.n_columns,
            "skipped_steps": self.skipped,
            "failures": self.failures,
            "metrics": self.metrics.to_dict(),
        }


class Pipeline:
    """Builds and runs a sequence of dataset operations."""

    def __init__(
        self,
        name: str = "pipeline",
        *,
        on_error: str = "fail",
        policy: PathPolicy | None = None,
        metrics_sink: MetricsSink | None = None,
        memory_warning_mb: float | None = 512.0,
    ) -> None:
        if on_error not in ("fail", "skip"):
            raise ConfigurationError("on_error must be 'fail' or 'skip'", given=on_error)
        self.name = name
        self.on_error = on_error
        self.policy = policy or PathPolicy()
        self.metrics_sink = metrics_sink
        self.memory_warning_mb = memory_warning_mb
        self.steps: list[Step] = []
        self._source: Callable[[PipelineContext], Dataset] | None = None

    # -- building -----------------------------------------------------------

    def add_step(
        self,
        name: str,
        function: StepFunction,
        *,
        condition: Callable[[Dataset], bool] | None = None,
        optional: bool = False,
    ) -> Pipeline:
        self.steps.append(Step(name=name, run=function, condition=condition, optional=optional))
        return self

    def read(self, path: PathLike, fmt: str | None = None, **options: Any) -> Pipeline:
        def source(context: PipelineContext) -> Dataset:
            return Dataset.read(path, fmt, policy=context.policy, **options)

        self._source = source
        return self

    def from_dataset(self, dataset: Dataset) -> Pipeline:
        self._source = lambda context: dataset
        return self

    def from_callable(self, factory: Callable[[], Dataset]) -> Pipeline:
        self._source = lambda context: factory()
        return self

    def clean(self, config: CleaningConfig | dict[str, Any] | None = None, **kwargs: Any) -> Pipeline:
        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            return dataset.clean(config, **kwargs)

        return self.add_step("clean", step)

    def remove_duplicates(self, subset: Sequence[str] | None = None, **kwargs: Any) -> Pipeline:
        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            return dataset.remove_duplicates(subset, **kwargs)

        return self.add_step("remove_duplicates", step)

    def validate(
        self,
        schema: Schema,
        *,
        mode: str = "report",
    ) -> Pipeline:
        """Validate against *schema*.

        ``mode='report'`` records the result and continues, ``'strict'`` fails
        the run, ``'filter'`` keeps only the rows that pass.
        """
        if mode not in ("report", "strict", "filter"):
            raise ConfigurationError("validate mode must be report, strict or filter", given=mode)

        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            result = dataset.validate(schema, raise_on_error=(mode == "strict"))
            context.validation = result
            metrics = context.metrics.run.steps[-1]
            metrics.rows_failed = result.invalid_rows
            if mode == "filter":
                valid, invalid = dataset.keep_valid(schema)
                context.artifacts["invalid_rows"] = invalid
                metrics.rows_skipped = invalid.n_rows
                return valid
            return dataset

        return self.add_step("validate", step)

    def transform(self, *steps: Transformation) -> Pipeline:
        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            return dataset.transform(*steps)

        return self.add_step("transform", step)

    def filter(self, condition: str) -> Pipeline:
        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            return dataset.filter(condition)

        return self.add_step("filter", step)

    def apply_rules(self, rules: RuleEngine, *, action: str = "flag") -> Pipeline:
        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            result, evaluation = dataset.apply_rules(rules, action=action)
            context.rule_evaluation = evaluation
            context.metrics.run.steps[-1].rows_failed = evaluation.rows_failed
            return result

        return self.add_step("business_rules", step)

    def mask(self, policy: MaskingPolicy | dict[str, Any]) -> Pipeline:
        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            return dataset.mask(policy)

        return self.add_step("mask", step)

    def mask_detected_pii(self, strategy: str = "hash", **options: Any) -> Pipeline:
        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            return dataset.mask_detected_pii(strategy, **options)

        return self.add_step("mask_pii", step)

    def enrich(self, enricher: Enricher) -> Pipeline:
        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            return dataset.enrich(enricher)

        return self.add_step("enrich", step)

    def handle_outliers(self, **kwargs: Any) -> Pipeline:
        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            return dataset.handle_outliers(**kwargs)

        return self.add_step("outliers", step)

    def quality_check(
        self,
        schema: Schema | None = None,
        rules: RuleEngine | None = None,
        *,
        minimum_score: float | None = None,
    ) -> Pipeline:
        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            report = dataset.quality(schema, rules)
            context.quality = report
            if minimum_score is not None and report.overall_score < minimum_score:
                raise PipelineError(
                    "Data quality score is below the configured minimum",
                    step="quality_check",
                    score=round(report.overall_score, 4),
                    minimum=minimum_score,
                )
            return dataset

        return self.add_step("quality_check", step)

    def write(self, target: PathLike, fmt: str | None = None, **options: Any) -> Pipeline:
        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            rows = dataset.write(target, fmt, **options)
            context.artifacts.setdefault("outputs", []).append({"path": str(target), "rows": rows})
            return dataset

        return self.add_step("write", step)

    def report(self, target: PathLike, *, fmt: str = "html") -> Pipeline:
        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            from universal_data.quality.report import QualityDocument

            document = QualityDocument(
                dataset=dataset.name,
                source=dataset.source,
                profile=dataset.profile(),
                validation=context.validation,
                rules=context.rule_evaluation,
                quality=context.quality or dataset.quality(),
                cleaning=dataset.metadata.get("cleaning_report"),
                metrics=context.metrics.run,
            )
            if fmt == "json":
                document.to_json(target, policy=context.policy)
            else:
                document.to_html(target, policy=context.policy)
            context.artifacts.setdefault("reports", []).append(str(target))
            return dataset

        return self.add_step("report", step)

    def when(self, predicate: Callable[[Dataset], bool]) -> Pipeline:
        """Make the previously added step conditional."""
        if not self.steps:
            raise ConfigurationError("when() must follow a step")
        self.steps[-1].condition = predicate
        return self

    # -- execution ----------------------------------------------------------

    def run(self, dataset: Dataset | None = None) -> PipelineResult:
        """Execute every step and return the result."""
        execution_id = new_execution_id()
        collector = MetricsCollector(execution_id, self.name)
        context = PipelineContext(
            execution_id=execution_id,
            pipeline=self.name,
            policy=self.policy,
            metrics=collector,
        )
        skipped: list[str] = []
        failures: list[dict[str, str]] = []

        with log_context(execution_id=execution_id, pipeline=self.name):
            logger.info("Pipeline '%s' started", self.name)
            current = self._resolve_source(dataset, context, collector)
            self._warn_about_memory(current)

            for step in self.steps:
                if step.condition is not None and not step.condition(current):
                    logger.info("Step '%s' skipped by its condition", step.name)
                    skipped.append(step.name)
                    continue
                with log_context(step=step.name):
                    try:
                        with collector.step(step.name, rows_in=current.n_rows) as metrics:
                            current = step.run(current, context)
                            metrics.rows_out = current.n_rows
                    except Exception as exc:
                        failures.append({"step": step.name, "error": f"{type(exc).__name__}: {exc}"})
                        if self.on_error == "fail" and not step.optional:
                            collector.finish()
                            raise PipelineError(
                                f"Step '{step.name}' failed: {exc}", step=step.name
                            ) from exc
                        logger.warning("Step '%s' failed and was skipped: %s", step.name, exc)
                        skipped.append(step.name)

            metrics_run = collector.finish()
            logger.info(
                "Pipeline '%s' finished in %.2fs with status %s",
                self.name,
                metrics_run.duration_seconds,
                metrics_run.status,
            )

        if self.metrics_sink is not None:
            self.metrics_sink.emit(metrics_run)
        return PipelineResult(
            dataset=current,
            metrics=metrics_run,
            context=context,
            skipped=skipped,
            failures=failures,
        )

    def _resolve_source(
        self, dataset: Dataset | None, context: PipelineContext, collector: MetricsCollector
    ) -> Dataset:
        if dataset is not None:
            return dataset
        if self._source is None:
            raise ConfigurationError("Pipeline has no input; call read() or pass a dataset")
        with collector.step("read") as metrics:
            loaded = self._source(context)
            metrics.rows_in = loaded.n_rows
            metrics.rows_out = loaded.n_rows
        return loaded

    def _warn_about_memory(self, dataset: Dataset) -> None:
        if self.memory_warning_mb is None:
            return
        size = dataset.memory_mb
        if size >= self.memory_warning_mb:
            warning = MemoryTracker().check_headroom(size)
            logger.warning(
                "Dataset holds %.1f MB in memory%s", size, f"; {warning}" if warning else ""
            )

    # -- chunked execution --------------------------------------------------

    def run_chunked(
        self,
        path: PathLike,
        target: PathLike,
        *,
        chunk_size: int = 50_000,
        fmt: str | None = None,
        output_format: str | None = None,
        **read_options: Any,
    ) -> PipelineResult:
        """Stream a file through the pipeline chunk by chunk.

        Memory stays proportional to ``chunk_size`` rather than to the file, so a
        20 GB CSV can be processed on a laptop.  Steps that need the whole
        dataset (deduplication across chunks, global aggregation) cannot work in
        this mode - the row counts they report are per chunk.
        """
        from pathlib import Path

        from universal_data.export.base import writers

        execution_id = new_execution_id()
        collector = MetricsCollector(execution_id, self.name)
        context = PipelineContext(
            execution_id=execution_id,
            pipeline=self.name,
            policy=self.policy,
            metrics=collector,
        )
        writer_cls = writers.get(output_format) if output_format else writers.for_path(Path(target))
        writer = writer_cls(policy=self.policy)
        failures: list[dict[str, str]] = []
        last: Dataset | None = None
        rows_read = 0

        def processed_chunks() -> Iterable[pd.DataFrame]:
            nonlocal last, rows_read
            for index, chunk in enumerate(
                Dataset.read_chunks(path, chunk_size, fmt, policy=self.policy, **read_options)
            ):
                rows_read += chunk.n_rows
                current = chunk
                with log_context(execution_id=execution_id, pipeline=self.name, chunk=index):
                    for step in self.steps:
                        if step.name == "write":
                            continue  # the streaming writer owns the output
                        if step.condition is not None and not step.condition(current):
                            continue
                        try:
                            with collector.step(f"{step.name}[{index}]", current.n_rows) as metrics:
                                current = step.run(current, context)
                                metrics.rows_out = current.n_rows
                        except Exception as exc:
                            failures.append(
                                {"step": f"{step.name}[{index}]", "error": str(exc)}
                            )
                            if self.on_error == "fail" and not step.optional:
                                raise PipelineError(
                                    f"Step '{step.name}' failed on chunk {index}: {exc}",
                                    step=step.name,
                                ) from exc
                last = current
                yield current.frame

        with log_context(execution_id=execution_id, pipeline=self.name):
            logger.info("Pipeline '%s' started in chunked mode (chunk_size=%s)", self.name, chunk_size)
            rows = writer.write_chunks(processed_chunks(), target)
            metrics_run = collector.finish()
            metrics_run.rows_input_total = rows_read
            metrics_run.rows_output_total = rows
            logger.info("Wrote %s rows to %s", rows, target)

        context.artifacts["outputs"] = [{"path": str(target), "rows": rows}]
        result_dataset = last or Dataset(pd.DataFrame(), name=self.name)
        if self.metrics_sink is not None:
            self.metrics_sink.emit(metrics_run)
        return PipelineResult(
            dataset=result_dataset, metrics=metrics_run, context=context, failures=failures
        )

    def __repr__(self) -> str:
        return f"Pipeline(name={self.name!r}, steps={[step.name for step in self.steps]})"
