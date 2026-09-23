"""Cleaning orchestration.

:class:`CleaningConfig` is what a YAML pipeline maps onto, and
:class:`DataCleaner` applies the steps in a fixed, sensible order: structural
fixes first, then text normalisation, then types, then missing values, then
duplicates.  Running duplicates *after* text normalisation matters - " Ali " and
"Ali" are only duplicates once the whitespace is gone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from universal_data.cleaning import operations as ops
from universal_data.core.types import DuplicateKeep, MissingStrategy
from universal_data.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningReport:
    """What a cleaning run changed."""

    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    def record(self, report: dict[str, Any]) -> None:
        self.steps.append(report)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_removed": self.rows_removed,
            "columns_before": self.columns_before,
            "columns_after": self.columns_after,
            "steps": self.steps,
        }

    def summary(self) -> str:
        lines = [
            f"Cleaning removed {self.rows_removed:,} of {self.rows_before:,} rows",
            f"Columns: {self.columns_before} -> {self.columns_after}",
        ]
        for step in self.steps:
            name = step.get("operation", "step")
            detail = ", ".join(
                f"{key}={value}"
                for key, value in step.items()
                if key != "operation" and value not in (None, [], {}, 0)
            )
            lines.append(f"  - {name}{': ' + detail if detail else ''}")
        return "\n".join(lines)


@dataclass
class CleaningConfig:
    """Declarative cleaning settings."""

    normalize_columns: bool = False
    drop_empty_rows: bool = True
    drop_empty_columns: bool = False
    trim_whitespace: bool = True
    normalize_spaces: bool = True
    case: str | None = None
    remove_special_characters: bool = False
    normalize_unicode: bool = False
    string_columns: Sequence[str] | None = None
    replacements: Sequence[tuple[str, str]] = ()
    convert_types: dict[str, str] = field(default_factory=dict)
    date_columns: Sequence[str] = ()
    date_format: str | None = None
    dayfirst: bool = False
    timezone: str | None = None
    numeric_columns: Sequence[str] = ()
    numeric_min: float | None = None
    numeric_max: float | None = None
    missing_values: MissingStrategy | str = MissingStrategy.NONE
    missing_value: Any = None
    missing_columns: Sequence[str] | None = None
    missing_threshold: float = 0.5
    remove_duplicates: bool = False
    duplicate_subset: Sequence[str] | None = None
    duplicate_keep: DuplicateKeep | str = DuplicateKeep.FIRST

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CleaningConfig:
        payload = dict(data)
        known = set(cls.__dataclass_fields__)
        # Accept the shorthand used in pipeline YAML: `clean: {missing_values: median}`
        aliases = {
            "duplicates": "remove_duplicates",
            "lowercase": "case",
            "columns": "string_columns",
        }
        for alias, target in aliases.items():
            if alias in payload:
                payload[target] = payload.pop(alias)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown cleaning option(s): {sorted(unknown)}")
        if "replacements" in payload and isinstance(payload["replacements"], dict):
            payload["replacements"] = tuple(payload["replacements"].items())
        return cls(**payload)


class DataCleaner:
    """Applies a :class:`CleaningConfig` to a DataFrame."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()

    def clean(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        config = self.config
        report = CleaningReport(
            rows_before=len(frame), columns_before=len(frame.columns)
        )
        result = frame

        if config.normalize_columns:
            result, step = ops.normalize_column_names(result)
            report.record(step)

        if config.drop_empty_rows or config.drop_empty_columns:
            result, step = ops.drop_empty(
                result, rows=config.drop_empty_rows, columns=config.drop_empty_columns
            )
            report.record(step)

        if config.trim_whitespace or config.case or config.remove_special_characters:
            result, step = ops.clean_strings(
                result,
                columns=config.string_columns,
                trim=config.trim_whitespace,
                normalize_spaces=config.normalize_spaces,
                case=config.case,
                remove_special=config.remove_special_characters,
                normalize_unicode=config.normalize_unicode,
                replacements=config.replacements,
            )
            report.record(step)

        if config.date_columns:
            result, step = ops.parse_dates(
                result,
                config.date_columns,
                date_format=config.date_format,
                dayfirst=config.dayfirst,
                timezone=config.timezone,
            )
            report.record(step)

        if config.numeric_columns:
            result, step = ops.clean_numeric(
                result,
                config.numeric_columns,
                minimum=config.numeric_min,
                maximum=config.numeric_max,
            )
            report.record(step)

        if config.convert_types:
            result, step = ops.convert_types(result, config.convert_types)
            report.record(step)

        if MissingStrategy(config.missing_values) is not MissingStrategy.NONE:
            result, step = ops.handle_missing(
                result,
                config.missing_values,
                columns=config.missing_columns,
                value=config.missing_value,
                threshold=config.missing_threshold,
            )
            report.record(step)

        if config.remove_duplicates:
            result, step = ops.remove_duplicates(
                result, subset=config.duplicate_subset, keep=config.duplicate_keep
            )
            report.record(step)

        report.rows_after = len(result)
        report.columns_after = len(result.columns)
        logger.debug("Cleaning finished: %s rows -> %s rows", report.rows_before, report.rows_after)
        return result, report
