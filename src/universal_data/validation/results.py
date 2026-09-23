"""Validation result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from universal_data.core.types import Severity

MAX_SAMPLES = 5


@dataclass
class ValidationIssue:
    """One failed expectation, aggregated over the rows it affects."""

    column: str | None
    check: str
    message: str
    severity: Severity = Severity.ERROR
    failed_rows: int = 0
    total_rows: int = 0
    samples: list[Any] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failed_rows / self.total_rows if self.total_rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "check": self.check,
            "severity": str(self.severity),
            "message": self.message,
            "failed_rows": self.failed_rows,
            "total_rows": self.total_rows,
            "failure_rate": round(self.failure_rate, 4),
            "samples": self.samples,
        }

    def __str__(self) -> str:
        location = f"[{self.column}] " if self.column else ""
        return f"{location}{self.message} ({self.failed_rows} rows)"


@dataclass
class ValidationResult:
    """Outcome of validating a frame against a schema."""

    issues: list[ValidationIssue] = field(default_factory=list)
    row_mask: pd.Series | None = None
    total_rows: int = 0
    schema_name: str = "schema"

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def invalid_rows(self) -> int:
        if self.row_mask is None:
            return 0
        return int((~self.row_mask).sum())

    @property
    def valid_rows(self) -> int:
        return self.total_rows - self.invalid_rows

    def valid_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.row_mask is None:
            return frame
        return frame.loc[self.row_mask]

    def invalid_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.row_mask is None:
            return frame.iloc[0:0]
        return frame.loc[~self.row_mask]

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "valid": self.is_valid,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def summary(self) -> str:
        if self.is_valid and not self.warnings:
            return f"Validation passed: {self.total_rows:,} rows match '{self.schema_name}'"
        lines = [
            f"Validation of '{self.schema_name}': "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)",
            f"Rows: {self.valid_rows:,} valid / {self.invalid_rows:,} invalid",
            "",
        ]
        lines.extend(f"  {issue.severity.upper():7} {issue}" for issue in self.issues)
        return "\n".join(lines)
