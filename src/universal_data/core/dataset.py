"""The :class:`Dataset` - the toolkit's central abstraction.

A ``Dataset`` wraps a pandas DataFrame together with the metadata a pipeline
needs: where it came from, what has been done to it, and which schema it is
expected to satisfy.

Every operation returns a **new** ``Dataset``.  Immutability costs one shallow
copy per step and buys two things that matter in a data pipeline: a step can
fail without corrupting the input, and the operation history is a truthful
record of what produced the output.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from universal_data.core.exceptions import ConfigurationError, DataProcessingError
from universal_data.core.types import (
    DuplicateKeep,
    FileFormat,
    MissingStrategy,
    OutlierAction,
    OutlierMethod,
    PathLike,
)
from universal_data.observability.logging import get_logger
from universal_data.observability.memory import dataframe_memory_mb
from universal_data.security.paths import PathPolicy

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from universal_data.cleaning.engine import CleaningConfig, CleaningReport
    from universal_data.enrichment.enrichers import Enricher
    from universal_data.ingestion.api import APIClient
    from universal_data.ingestion.database import DatabaseClient
    from universal_data.inspection.profiler import DatasetProfile
    from universal_data.masking.maskers import MaskingPolicy
    from universal_data.quality.metrics import QualityReport
    from universal_data.rules.engine import RuleEngine, RuleEvaluation
    from universal_data.schema.model import Schema
    from universal_data.transformation.base import Transformation
    from universal_data.validation.results import ValidationResult

logger = get_logger(__name__)


class Dataset:
    """An in-memory tabular dataset with provenance and helper operations."""

    __slots__ = ("_frame", "name", "source", "metadata", "history", "policy", "schema")

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        name: str = "dataset",
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        policy: PathPolicy | None = None,
        schema: Schema | None = None,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise DataProcessingError(
                "Dataset expects a pandas DataFrame", given=type(frame).__name__
            )
        self._frame = frame
        self.name = name
        self.source = source
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.history: list[dict[str, Any]] = list(history or [])
        self.policy = policy or PathPolicy()
        self.schema = schema

    # -- construction -------------------------------------------------------

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, **kwargs: Any) -> Dataset:
        return cls(frame, **kwargs)

    @classmethod
    def from_records(cls, records: Sequence[dict[str, Any]], **kwargs: Any) -> Dataset:
        return cls(pd.json_normalize(list(records)), **kwargs)

    @classmethod
    def read(
        cls,
        path: PathLike,
        fmt: FileFormat | str | None = None,
        *,
        policy: PathPolicy | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Dataset:
        """Read a file, detecting the format when it is not given."""
        from universal_data.ingestion.base import readers
        from universal_data.ingestion.detect import detect_format

        policy = policy or PathPolicy()
        resolved = policy.resolve_input(path)
        file_format = FileFormat(fmt) if fmt else detect_format(resolved, policy=policy)
        reader = readers.create(file_format, policy=policy)
        frame = reader.read(resolved, **options)
        logger.info(
            "Read %s rows x %s columns from %s (%s)",
            len(frame),
            len(frame.columns),
            resolved.name,
            file_format,
        )
        return cls(
            frame,
            name=name or resolved.stem,
            source=str(resolved),
            policy=policy,
            metadata={"format": str(file_format)},
            history=[{"operation": "read", "source": str(resolved), "format": str(file_format)}],
        )

    @classmethod
    def read_chunks(
        cls,
        path: PathLike,
        chunk_size: int,
        fmt: FileFormat | str | None = None,
        *,
        policy: PathPolicy | None = None,
        **options: Any,
    ) -> Iterator[Dataset]:
        """Iterate over a file in chunks without loading it entirely."""
        from universal_data.ingestion.base import readers
        from universal_data.ingestion.detect import detect_format

        policy = policy or PathPolicy()
        resolved = policy.resolve_input(path)
        file_format = FileFormat(fmt) if fmt else detect_format(resolved, policy=policy)
        reader = readers.create(file_format, policy=policy)
        for index, chunk in enumerate(reader.read_chunks(resolved, chunk_size, **options)):
            yield cls(
                chunk,
                name=f"{resolved.stem}[{index}]",
                source=str(resolved),
                policy=policy,
                metadata={"chunk_index": index, "format": str(file_format)},
            )

    @classmethod
    def from_csv(cls, path: PathLike, **kwargs: Any) -> Dataset:
        return cls.read(path, FileFormat.CSV, **kwargs)

    @classmethod
    def from_excel(cls, path: PathLike, **kwargs: Any) -> Dataset:
        return cls.read(path, FileFormat.EXCEL, **kwargs)

    @classmethod
    def from_json(cls, path: PathLike, **kwargs: Any) -> Dataset:
        return cls.read(path, FileFormat.JSON, **kwargs)

    @classmethod
    def from_jsonl(cls, path: PathLike, **kwargs: Any) -> Dataset:
        return cls.read(path, FileFormat.JSONL, **kwargs)

    @classmethod
    def from_parquet(cls, path: PathLike, **kwargs: Any) -> Dataset:
        return cls.read(path, FileFormat.PARQUET, **kwargs)

    @classmethod
    def from_xml(cls, path: PathLike, **kwargs: Any) -> Dataset:
        return cls.read(path, FileFormat.XML, **kwargs)

    @classmethod
    def from_yaml(cls, path: PathLike, **kwargs: Any) -> Dataset:
        return cls.read(path, FileFormat.YAML, **kwargs)

    @classmethod
    def from_sqlite(cls, path: PathLike, table: str | None = None, **kwargs: Any) -> Dataset:
        return cls.read(path, FileFormat.SQLITE, table=table, **kwargs)

    @classmethod
    def from_database(
        cls,
        client: DatabaseClient,
        *,
        table: str | None = None,
        query: str | None = None,
        params: dict[str, Any] | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Dataset:
        """Read a table or a parameterised query into a dataset."""
        if bool(table) == bool(query):
            raise ConfigurationError("Provide exactly one of 'table' or 'query'")
        if table is not None:
            frame = client.read_table(table, **options)
        else:
            assert query is not None
            frame = client.read_query(query, params=params, **options)
        assert isinstance(frame, pd.DataFrame)
        return cls(
            frame,
            name=name or table or "query",
            source=client.config.safe_url(),
            metadata={"driver": client.config.driver},
            history=[{"operation": "read_database", "table": table, "query": bool(query)}],
        )

    @classmethod
    def from_api(cls, client: APIClient, *, name: str = "api") -> Dataset:
        frame = client.fetch()
        return cls(
            frame,
            name=name,
            source=client.config.url,
            history=[{"operation": "read_api", "url": client.config.url, "rows": len(frame)}],
        )

    # -- basic properties ---------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        """The underlying DataFrame (treat it as read-only)."""
        return self._frame

    @property
    def shape(self) -> tuple[int, int]:
        return self._frame.shape

    @property
    def columns(self) -> list[str]:
        return [str(column) for column in self._frame.columns]

    @property
    def dtypes(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self._frame.dtypes.items()}

    @property
    def n_rows(self) -> int:
        return len(self._frame)

    @property
    def n_columns(self) -> int:
        return len(self._frame.columns)

    @property
    def memory_mb(self) -> float:
        return dataframe_memory_mb(self._frame)

    @property
    def empty(self) -> bool:
        return self._frame.empty

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, rows={self.n_rows}, columns={self.n_columns}, "
            f"memory={self.memory_mb:.2f}MB)"
        )

    def _derive(self, frame: pd.DataFrame, operation: dict[str, Any]) -> Dataset:
        """Build the next dataset in the chain, carrying provenance forward."""
        entry = {"timestamp": datetime.now(UTC).isoformat(), **operation}
        return Dataset(
            frame,
            name=self.name,
            source=self.source,
            metadata=dict(self.metadata),
            history=[*self.history, entry],
            policy=self.policy,
            schema=self.schema,
        )

    def copy(self) -> Dataset:
        return self._derive(self._frame.copy(), {"operation": "copy"})

    # -- inspection ---------------------------------------------------------

    def head(self, rows: int = 5) -> pd.DataFrame:
        return self._frame.head(rows)

    def tail(self, rows: int = 5) -> pd.DataFrame:
        return self._frame.tail(rows)

    def sample(self, rows: int = 5, *, random_state: int | None = None) -> pd.DataFrame:
        return self._frame.sample(min(rows, len(self._frame)), random_state=random_state)

    def profile(self, *, with_outliers: bool = True) -> DatasetProfile:
        from universal_data.inspection.profiler import profile_dataset

        return profile_dataset(
            self._frame, name=self.name, source=self.source, with_outliers=with_outliers
        )

    def detect_schema(self, *, infer_constraints: bool = False) -> Schema:
        from universal_data.schema.detect import detect_schema

        return detect_schema(
            self._frame, name=f"{self.name}_schema", infer_constraints=infer_constraints
        )

    def quality(
        self,
        schema: Schema | None = None,
        rules: RuleEngine | None = None,
        **kwargs: Any,
    ) -> QualityReport:
        from universal_data.quality.metrics import QualityAnalyzer

        return QualityAnalyzer(schema or self.schema, rules, **kwargs).analyze(
            self._frame, name=self.name
        )

    # -- validation ---------------------------------------------------------

    def validate(
        self, schema: Schema | None = None, *, raise_on_error: bool = False
    ) -> ValidationResult:
        from universal_data.validation.engine import SchemaValidator

        target = schema or self.schema
        if target is None:
            raise ConfigurationError("No schema was supplied and the dataset has none attached")
        return SchemaValidator(target).validate(self._frame, raise_on_error=raise_on_error)

    def with_schema(self, schema: Schema) -> Dataset:
        result = self._derive(self._frame, {"operation": "attach_schema", "schema": schema.name})
        result.schema = schema
        return result

    def keep_valid(self, schema: Schema | None = None) -> tuple[Dataset, Dataset]:
        """Split into ``(valid, invalid)`` according to a schema."""
        result = self.validate(schema)
        valid = self._derive(
            result.valid_frame(self._frame),
            {"operation": "keep_valid", "invalid_rows": result.invalid_rows},
        )
        invalid = self._derive(
            result.invalid_frame(self._frame), {"operation": "quarantine_invalid"}
        )
        invalid.name = f"{self.name}_invalid"
        return valid, invalid

    # -- cleaning -----------------------------------------------------------

    def clean(
        self, config: CleaningConfig | dict[str, Any] | None = None, **kwargs: Any
    ) -> Dataset:
        """Run the cleaning engine.  The report is stored in ``metadata``."""
        from universal_data.cleaning.engine import CleaningConfig, DataCleaner

        if isinstance(config, dict):
            settings = CleaningConfig.from_dict({**config, **kwargs})
        elif config is None:
            settings = CleaningConfig(**kwargs) if kwargs else CleaningConfig()
        else:
            settings = config
        frame, report = DataCleaner(settings).clean(self._frame)
        result = self._derive(frame, {"operation": "clean", **report.to_dict()})
        result.metadata["cleaning_report"] = report
        return result

    def last_cleaning_report(self) -> CleaningReport | None:
        return self.metadata.get("cleaning_report")

    def remove_duplicates(
        self,
        subset: Sequence[str] | None = None,
        keep: DuplicateKeep | str = DuplicateKeep.FIRST,
    ) -> Dataset:
        from universal_data.cleaning.operations import remove_duplicates

        frame, report = remove_duplicates(self._frame, subset=subset, keep=keep)
        return self._derive(frame, report)

    def fill_missing(
        self,
        strategy: MissingStrategy | str = MissingStrategy.MEDIAN,
        *,
        columns: Sequence[str] | None = None,
        value: Any = None,
    ) -> Dataset:
        from universal_data.cleaning.operations import handle_missing

        frame, report = handle_missing(self._frame, strategy, columns=columns, value=value)
        return self._derive(frame, report)

    def drop_missing(self, columns: Sequence[str] | None = None) -> Dataset:
        from universal_data.cleaning.operations import handle_missing

        frame, report = handle_missing(
            self._frame, MissingStrategy.DROP_ROWS, columns=columns
        )
        return self._derive(frame, report)

    # -- transformation -----------------------------------------------------

    def transform(self, *steps: Transformation) -> Dataset:
        from universal_data.transformation.base import TransformationChain

        chain = TransformationChain(steps)
        frame = chain.apply(self._frame)
        return self._derive(
            frame, {"operation": "transform", "steps": [step.describe() for step in chain]}
        )

    def filter(self, condition: str) -> Dataset:
        from universal_data.transformation.operations import FilterRows

        return self.transform(FilterRows(condition))

    def select(self, columns: Sequence[str] | str) -> Dataset:
        from universal_data.transformation.operations import SelectColumns

        return self.transform(SelectColumns(columns))

    def drop(self, columns: Sequence[str] | str) -> Dataset:
        from universal_data.transformation.operations import DropColumns

        return self.transform(DropColumns(columns))

    def rename(self, mapping: dict[str, str]) -> Dataset:
        from universal_data.transformation.operations import RenameColumns

        return self.transform(RenameColumns(mapping))

    def sort(self, by: Sequence[str] | str, *, ascending: bool = True) -> Dataset:
        from universal_data.transformation.operations import SortRows

        return self.transform(SortRows(by, ascending=ascending))

    def add_column(self, name: str, expression: str) -> Dataset:
        from universal_data.transformation.operations import CreateColumn

        return self.transform(CreateColumn(name, expression))

    def join(self, other: Dataset | pd.DataFrame, **kwargs: Any) -> Dataset:
        from universal_data.transformation.operations import Join

        return self.transform(Join(other, **kwargs))

    def aggregate(self, group_by: Sequence[str] | str, aggregations: dict[str, Any]) -> Dataset:
        from universal_data.transformation.operations import Aggregate

        return self.transform(Aggregate(group_by, aggregations))

    # -- rules, outliers, masking, enrichment -------------------------------

    def apply_rules(
        self, rules: RuleEngine, *, action: str = "flag"
    ) -> tuple[Dataset, RuleEvaluation]:
        """Evaluate business rules.

        ``action='flag'`` keeps every row and adds a ``_rule_failures`` column;
        ``action='drop'`` keeps only the rows that pass every error-level rule.
        """
        if action not in ("flag", "drop"):
            raise ConfigurationError("action must be 'flag' or 'drop'", given=action)
        from universal_data.rules.engine import FAILURE_COLUMN

        evaluation = rules.evaluate(self._frame)
        if action == "drop":
            frame = evaluation.passed_frame(self._frame)
        else:
            frame = self._frame.assign(**{FAILURE_COLUMN: evaluation.reasons})
        result = self._derive(
            frame,
            {
                "operation": "apply_rules",
                "action": action,
                "rows_failed": evaluation.rows_failed,
                "rules": len(rules),
            },
        )
        return result, evaluation

    def detect_outliers(
        self,
        columns: Sequence[str] | None = None,
        *,
        method: OutlierMethod | str = OutlierMethod.IQR,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        from universal_data.quality.outliers import handle_outliers

        _, report = handle_outliers(
            self._frame, columns, method=method, threshold=threshold,
            action=OutlierAction.REPORT,
        )
        return report

    def handle_outliers(
        self,
        columns: Sequence[str] | None = None,
        *,
        method: OutlierMethod | str = OutlierMethod.IQR,
        threshold: float | None = None,
        action: OutlierAction | str = OutlierAction.CAP,
    ) -> Dataset:
        from universal_data.quality.outliers import handle_outliers

        frame, report = handle_outliers(
            self._frame, columns, method=method, threshold=threshold, action=action
        )
        return self._derive(frame, report)

    def mask(self, policy: MaskingPolicy | dict[str, Any]) -> Dataset:
        from universal_data.masking.maskers import MaskingPolicy

        masking = MaskingPolicy.from_dict(policy) if isinstance(policy, dict) else policy
        frame, report = masking.apply(self._frame)
        return self._derive(frame, report)

    def mask_detected_pii(self, strategy: str = "hash", **options: Any) -> Dataset:
        """Mask every column that schema detection flagged as personal data."""
        from universal_data.masking.maskers import MaskingPolicy

        schema = self.schema or self.detect_schema()
        columns = schema.pii_columns
        if not columns:
            logger.info("No PII columns detected; nothing to mask")
            # Still a new dataset: every operation returns one, so a caller can
            # keep chaining without checking whether anything was masked.
            return self._derive(self._frame, {"operation": "mask", "applied": {}})
        return self.mask(MaskingPolicy.for_columns(columns, strategy, **options))

    def enrich(self, enricher: Enricher) -> Dataset:
        frame = enricher.enrich(self._frame)
        return self._derive(
            frame, {"operation": "enrich", "enricher": type(enricher).__name__}
        )

    # -- export -------------------------------------------------------------

    def write(
        self,
        target: PathLike,
        fmt: FileFormat | str | None = None,
        **options: Any,
    ) -> int:
        """Write the dataset to a file, inferring the format from the extension."""
        from universal_data.export.base import writers

        writer_cls = writers.get(fmt) if fmt else writers.for_path(Path(target))
        writer = writer_cls(policy=self.policy)
        rows = writer.write(self._frame, target, **options)
        logger.info("Wrote %s rows to %s", rows, target)
        return rows

    def to_csv(self, target: PathLike, **options: Any) -> int:
        return self.write(target, FileFormat.CSV, **options)

    def to_json(self, target: PathLike, **options: Any) -> int:
        return self.write(target, FileFormat.JSON, **options)

    def to_jsonl(self, target: PathLike, **options: Any) -> int:
        return self.write(target, FileFormat.JSONL, **options)

    def to_excel(self, target: PathLike, **options: Any) -> int:
        return self.write(target, FileFormat.EXCEL, **options)

    def to_parquet(self, target: PathLike, **options: Any) -> int:
        return self.write(target, FileFormat.PARQUET, **options)

    def to_sqlite(self, target: PathLike, table: str = "data", **options: Any) -> int:
        return self.write(target, FileFormat.SQLITE, table=table, **options)

    def to_database(
        self,
        client: DatabaseClient,
        table: str,
        *,
        if_exists: str = "append",
        schema: str | None = None,
        chunk_size: int = 10_000,
    ) -> int:
        from universal_data.export.file_writers import DatabaseExporter

        rows = DatabaseExporter(client).write(
            self._frame, table, schema=schema, if_exists=if_exists, chunk_size=chunk_size
        )
        logger.info("Wrote %s rows to table %s", rows, table)
        return rows

    def to_records(self) -> list[dict[str, Any]]:
        return self._frame.to_dict(orient="records")

    # -- reporting ----------------------------------------------------------

    def report(
        self,
        *,
        schema: Schema | None = None,
        rules: RuleEngine | None = None,
    ) -> Any:
        """Build a :class:`~universal_data.quality.report.QualityDocument`."""
        from universal_data.quality.report import QualityDocument

        target_schema = schema or self.schema
        evaluation = rules.evaluate(self._frame) if rules else None
        return QualityDocument(
            dataset=self.name,
            source=self.source,
            profile=self.profile(),
            validation=self.validate(target_schema) if target_schema else None,
            rules=evaluation,
            quality=self.quality(target_schema, rules),
            cleaning=self.metadata.get("cleaning_report"),
        )
