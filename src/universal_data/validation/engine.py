"""Schema validation engine."""

from __future__ import annotations

from typing import Any

import pandas as pd

from universal_data.core.exceptions import DataValidationError
from universal_data.core.types import Severity
from universal_data.observability.logging import get_logger
from universal_data.schema.model import ColumnSchema, Schema
from universal_data.validation import checks
from universal_data.validation.results import MAX_SAMPLES, ValidationIssue, ValidationResult

logger = get_logger(__name__)


class SchemaValidator:
    """Validates a DataFrame against a :class:`~universal_data.schema.model.Schema`.

    The validator never mutates the frame.  It returns a
    :class:`~universal_data.validation.results.ValidationResult` that carries a
    row mask, so a caller can quarantine bad rows instead of losing the batch.
    """

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def validate(self, frame: pd.DataFrame, *, raise_on_error: bool = False) -> ValidationResult:
        result = ValidationResult(total_rows=len(frame), schema_name=self.schema.name)
        row_valid = pd.Series(True, index=frame.index)

        self._check_structure(frame, result)

        for name, column in self.schema.columns.items():
            if name not in frame.columns:
                continue
            column_valid = self._validate_column(frame[name], column, result)
            row_valid &= column_valid

        row_valid &= self._check_primary_key(frame, result)
        result.row_mask = row_valid

        if raise_on_error and not result.is_valid:
            raise DataValidationError(
                f"Dataset does not satisfy schema '{self.schema.name}'",
                errors=[issue.to_dict() for issue in result.errors],
                invalid_rows=result.invalid_rows,
            )
        return result

    def _check_structure(self, frame: pd.DataFrame, result: ValidationResult) -> None:
        columns = set(frame.columns)
        for name, column in self.schema.columns.items():
            if name in columns:
                continue
            severity = Severity.ERROR if column.required else Severity.WARNING
            result.add(
                ValidationIssue(
                    column=name,
                    check="missing_column",
                    message=f"Column '{name}' is declared in the schema but missing",
                    severity=severity,
                    failed_rows=len(frame) if severity is Severity.ERROR else 0,
                    total_rows=len(frame),
                )
            )
        extra = columns - set(self.schema.columns)
        if extra:
            result.add(
                ValidationIssue(
                    column=None,
                    check="unexpected_columns",
                    message=f"Dataset has columns not declared in the schema: {sorted(extra)}",
                    severity=Severity.ERROR if self.schema.strict else Severity.WARNING,
                    total_rows=len(frame),
                )
            )

    def _validate_column(
        self, series: pd.Series, column: ColumnSchema, result: ValidationResult
    ) -> pd.Series:
        valid = pd.Series(True, index=series.index)

        if not column.nullable:
            null_mask = series.isna()
            valid &= ~null_mask
            self._report(result, column.name, "not_null", null_mask, series,
                         f"Column '{column.name}' contains null values but is declared not null")

        type_mask = checks.check_type(series, column.type)
        valid &= type_mask
        self._report(result, column.name, "type", ~type_mask, series,
                     f"Column '{column.name}' has values that are not a valid {column.type}")

        if column.min is not None:
            mask = checks.check_min(series, column.min, column.type)
            valid &= mask
            self._report(result, column.name, "min", ~mask, series,
                         f"Column '{column.name}' has values below the minimum {column.min}")

        if column.max is not None:
            mask = checks.check_max(series, column.max, column.type)
            valid &= mask
            self._report(result, column.name, "max", ~mask, series,
                         f"Column '{column.name}' has values above the maximum {column.max}")

        if column.min_length is not None or column.max_length is not None:
            mask = checks.check_length(series, column.min_length, column.max_length)
            valid &= mask
            self._report(result, column.name, "length", ~mask, series,
                         f"Column '{column.name}' has values outside the allowed length")

        if column.pattern:
            mask = checks.check_pattern(series, column.pattern)
            valid &= mask
            self._report(result, column.name, "pattern", ~mask, series,
                         f"Column '{column.name}' has values that do not match the pattern")

        if column.allowed:
            mask = checks.check_allowed(series, column.allowed)
            valid &= mask
            self._report(result, column.name, "allowed", ~mask, series,
                         f"Column '{column.name}' has values outside the allowed set")

        if column.unique:
            mask = checks.check_unique(series)
            valid &= mask
            self._report(result, column.name, "unique", ~mask, series,
                         f"Column '{column.name}' contains duplicate values")

        return valid

    def _check_primary_key(self, frame: pd.DataFrame, result: ValidationResult) -> pd.Series:
        keys = [key for key in self.schema.primary_key if key in frame.columns]
        if not keys:
            return pd.Series(True, index=frame.index)
        duplicated = frame.duplicated(subset=keys, keep=False)
        null_key = frame[keys].isna().any(axis=1)
        invalid = duplicated | null_key
        if invalid.any():
            result.add(
                ValidationIssue(
                    column=", ".join(keys),
                    check="primary_key",
                    message=f"Primary key {keys} is not unique or contains nulls",
                    failed_rows=int(invalid.sum()),
                    total_rows=len(frame),
                )
            )
        return ~invalid

    @staticmethod
    def _report(
        result: ValidationResult,
        column: str,
        check: str,
        failed_mask: pd.Series,
        series: pd.Series,
        message: str,
    ) -> None:
        failed = int(failed_mask.sum())
        if not failed:
            return
        samples = series.loc[failed_mask].head(MAX_SAMPLES).tolist()
        result.add(
            ValidationIssue(
                column=column,
                check=check,
                message=message,
                failed_rows=failed,
                total_rows=len(series),
                samples=[_safe_sample(value) for value in samples],
            )
        )


def _safe_sample(value: Any) -> Any:
    """Samples end up in reports and logs, so keep them short and printable."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    return text if len(text) <= 64 else f"{text[:61]}..."


def validate_frame(
    frame: pd.DataFrame, schema: Schema, *, raise_on_error: bool = False
) -> ValidationResult:
    return SchemaValidator(schema).validate(frame, raise_on_error=raise_on_error)
