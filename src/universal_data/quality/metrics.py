"""Data quality scoring.

Six dimensions are measured.  Each one is a ratio in ``[0, 1]`` so they can be
combined into a single weighted score, and each one keeps the raw counts it was
computed from - a score without its evidence is not actionable.

Dimensions that cannot be measured for a given dataset (timeliness without a
date column, for instance) are omitted and the remaining weights are
renormalised, rather than being scored as zero.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from universal_data.core.types import FieldType
from universal_data.quality.outliers import detect_outliers
from universal_data.rules.engine import RuleEngine
from universal_data.schema.detect import detect_column_type
from universal_data.schema.model import Schema
from universal_data.validation import checks
from universal_data.validation.engine import SchemaValidator

SAMPLE_SIZE = 5_000
DEFAULT_WEIGHTS: dict[str, float] = {
    "completeness": 0.25,
    "validity": 0.25,
    "uniqueness": 0.15,
    "consistency": 0.15,
    "accuracy": 0.10,
    "timeliness": 0.10,
}

_SHAPE_DIGIT = re.compile(r"\d")
_SHAPE_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass
class QualityDimension:
    """One measured dimension."""

    name: str
    score: float
    weight: float
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def percentage(self) -> float:
        return round(self.score * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "percentage": self.percentage,
            "weight": self.weight,
            "details": self.details,
        }


@dataclass
class QualityReport:
    """Result of a quality analysis."""

    dataset: str
    rows: int
    columns: int
    dimensions: list[QualityDimension] = field(default_factory=list)
    column_scores: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        total_weight = sum(dimension.weight for dimension in self.dimensions)
        if not total_weight:
            return 0.0
        return sum(d.score * d.weight for d in self.dimensions) / total_weight

    @property
    def scored(self) -> bool:
        """False when there was nothing to measure (an empty dataset)."""
        return bool(self.dimensions)

    @property
    def grade(self) -> str:
        if not self.scored:
            # An empty dataset is not bad data, it is absent data; grading it F
            # sends a reader looking for a quality problem that does not exist.
            return "n/a"
        score = self.overall_score
        if score >= 0.95:
            return "A"
        if score >= 0.90:
            return "B"
        if score >= 0.80:
            return "C"
        if score >= 0.70:
            return "D"
        return "F"

    def dimension(self, name: str) -> QualityDimension | None:
        return next((d for d in self.dimensions if d.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "columns": self.columns,
            "scored": self.scored,
            "overall_score": round(self.overall_score, 4),
            "overall_percentage": round(self.overall_score * 100, 1),
            "grade": self.grade,
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
            "column_scores": {k: round(v, 4) for k, v in self.column_scores.items()},
            "recommendations": self.recommendations,
        }

    def summary(self) -> str:
        lines = [f"Dataset Quality Report: {self.dataset}", ""]
        if not self.scored:
            lines.append("Not scored: the dataset has no rows.")
            lines.extend(["", *(f"  - {item}" for item in self.recommendations)])
            return "\n".join(lines)
        width = max((len(d.name) for d in self.dimensions), default=12)
        for dimension in self.dimensions:
            lines.append(f"{dimension.name.capitalize().ljust(width)}  {dimension.percentage:5.1f}%")
        lines.extend(
            ["", f"Overall Score: {self.overall_score * 100:.1f}% (grade {self.grade})"]
        )
        if self.recommendations:
            lines.extend(["", "Recommendations:"])
            lines.extend(f"  - {item}" for item in self.recommendations)
        return "\n".join(lines)


class QualityAnalyzer:
    """Computes the quality dimensions for a DataFrame."""

    def __init__(
        self,
        schema: Schema | None = None,
        rules: RuleEngine | None = None,
        *,
        weights: dict[str, float] | None = None,
        timeliness_column: str | None = None,
        freshness_days: int = 30,
    ) -> None:
        self.schema = schema
        self.rules = rules
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.timeliness_column = timeliness_column
        self.freshness_days = freshness_days

    def analyze(self, frame: pd.DataFrame, *, name: str = "dataset") -> QualityReport:
        report = QualityReport(dataset=name, rows=len(frame), columns=len(frame.columns))
        if frame.empty:
            report.recommendations.append("Dataset is empty; nothing to score")
            return report

        report.dimensions.append(self._completeness(frame))
        report.dimensions.append(self._validity(frame))
        report.dimensions.append(self._uniqueness(frame))
        report.dimensions.append(self._consistency(frame))
        report.dimensions.append(self._accuracy(frame))
        timeliness = self._timeliness(frame)
        if timeliness is not None:
            report.dimensions.append(timeliness)

        report.column_scores = self._column_scores(frame)
        report.recommendations = self._recommendations(frame, report)
        return report

    # -- dimensions ---------------------------------------------------------

    def _completeness(self, frame: pd.DataFrame) -> QualityDimension:
        cells = frame.size
        missing = int(frame.isna().sum().sum())
        worst = frame.isna().mean().sort_values(ascending=False).head(5)
        return QualityDimension(
            name="completeness",
            score=1 - (missing / cells) if cells else 1.0,
            weight=self.weights["completeness"],
            details={
                "missing_cells": missing,
                "total_cells": int(cells),
                "worst_columns": {str(k): round(float(v), 4) for k, v in worst.items() if v > 0},
            },
        )

    def _validity(self, frame: pd.DataFrame) -> QualityDimension:
        if self.schema is not None:
            result = SchemaValidator(self.schema).validate(frame)
            score = result.valid_rows / result.total_rows if result.total_rows else 1.0
            return QualityDimension(
                name="validity",
                score=score,
                weight=self.weights["validity"],
                details={
                    "invalid_rows": result.invalid_rows,
                    "issues": len(result.errors),
                    "checked_against": self.schema.name,
                },
            )

        # Without a schema, validity means "values agree with the type the
        # column already looks like" - it catches mixed-type columns.
        invalid_cells = 0
        checked_cells = 0
        per_column: dict[str, int] = {}
        for column in frame.columns:
            series = frame[column]
            field_type = detect_column_type(series, str(column))
            if field_type is FieldType.UNKNOWN:
                continue
            valid = checks.check_type(series, field_type)
            failures = int((~valid).sum())
            checked_cells += int(series.notna().sum())
            invalid_cells += failures
            if failures:
                per_column[str(column)] = failures
        score = 1 - (invalid_cells / checked_cells) if checked_cells else 1.0
        return QualityDimension(
            name="validity",
            score=max(score, 0.0),
            weight=self.weights["validity"],
            details={"invalid_cells": invalid_cells, "by_column": per_column},
        )

    def _uniqueness(self, frame: pd.DataFrame) -> QualityDimension:
        duplicates = int(frame.duplicated().sum())
        score = 1 - (duplicates / len(frame))
        details: dict[str, Any] = {"duplicate_rows": duplicates}
        if self.schema and self.schema.primary_key:
            keys = [key for key in self.schema.primary_key if key in frame.columns]
            if keys:
                key_duplicates = int(frame.duplicated(subset=keys).sum())
                details["primary_key"] = keys
                details["primary_key_duplicates"] = key_duplicates
                score = min(score, 1 - key_duplicates / len(frame))
        return QualityDimension(
            name="uniqueness", score=score, weight=self.weights["uniqueness"], details=details
        )

    def _consistency(self, frame: pd.DataFrame) -> QualityDimension:
        """How uniform the formatting of each text column is."""
        scores: dict[str, float] = {}
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                continue
            values = series.dropna()
            if values.empty:
                continue
            sample = values.head(SAMPLE_SIZE).astype("string")
            shapes = sample.map(_value_shape)
            dominant = shapes.value_counts(normalize=True)
            if not dominant.empty:
                scores[str(column)] = float(dominant.iloc[0])

        rule_score: float | None = None
        if self.rules is not None and len(self.rules):
            evaluation = self.rules.evaluate(frame)
            rule_score = evaluation.rows_passed / evaluation.total_rows if evaluation.total_rows else 1.0

        if not scores and rule_score is None:
            return QualityDimension(
                name="consistency", score=1.0, weight=self.weights["consistency"], details={}
            )
        base = sum(scores.values()) / len(scores) if scores else 1.0
        score = base if rule_score is None else (base + rule_score) / 2
        return QualityDimension(
            name="consistency",
            score=score,
            weight=self.weights["consistency"],
            details={
                "format_consistency": {k: round(v, 4) for k, v in sorted(scores.items())},
                "business_rule_pass_rate": round(rule_score, 4) if rule_score is not None else None,
            },
        )

    def _accuracy(self, frame: pd.DataFrame) -> QualityDimension:
        """Proxy metric: how many numeric values sit outside a plausible range."""
        numeric_columns = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        total = 0
        outliers = 0
        per_column: dict[str, int] = {}
        for column in numeric_columns:
            series = frame[column]
            valid_count = int(series.notna().sum())
            if valid_count < 4:
                continue
            count = int(detect_outliers(series).sum())
            total += valid_count
            outliers += count
            if count:
                per_column[str(column)] = count
        score = 1 - (outliers / total) if total else 1.0
        return QualityDimension(
            name="accuracy",
            score=score,
            weight=self.weights["accuracy"],
            details={"outlier_values": outliers, "by_column": per_column, "method": "iqr"},
        )

    def _timeliness(self, frame: pd.DataFrame) -> QualityDimension | None:
        column = self.timeliness_column or self._pick_date_column(frame)
        if column is None or column not in frame.columns:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed")
        valid = parsed.dropna()
        if valid.empty:
            return None
        now = pd.Timestamp.now(tz=valid.dt.tz) if valid.dt.tz is not None else pd.Timestamp.now()
        cutoff = now - pd.Timedelta(days=self.freshness_days)
        fresh = int((valid >= cutoff).sum())
        future = int((valid > now).sum())

        # Timeliness is about how current the dataset is, so it is driven by the
        # newest record rather than by the share of recent rows: a five-year
        # order history updated yesterday is timely, not 2% timely.
        age_days = max((now - valid.max()).days, 0)
        if age_days <= self.freshness_days:
            recency = 1.0
        else:
            decay_window = self.freshness_days * 6
            recency = max(1 - (age_days - self.freshness_days) / decay_window, 0.0)
        future_penalty = future / len(valid)
        score = max(recency * (1 - future_penalty), 0.0)
        return QualityDimension(
            name="timeliness",
            score=score,
            weight=self.weights["timeliness"],
            details={
                "column": str(column),
                "freshness_days": self.freshness_days,
                "rows_within_window": fresh,
                "future_dated_rows": future,
                "latest": str(valid.max()),
                "age_of_latest_days": age_days,
            },
        )

    @staticmethod
    def _pick_date_column(frame: pd.DataFrame) -> str | None:
        preferred = ("updated_at", "modified_at", "created_at", "date", "timestamp")
        lowered = {str(column).lower(): str(column) for column in frame.columns}
        for candidate in preferred:
            if candidate in lowered:
                return lowered[candidate]
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                return str(column)
        return None

    # -- helpers ------------------------------------------------------------

    def _column_scores(self, frame: pd.DataFrame) -> dict[str, float]:
        """A per-column completeness/validity blend, useful for ranking columns."""
        scores: dict[str, float] = {}
        for column in frame.columns:
            series = frame[column]
            completeness = float(series.notna().mean())
            field_type = detect_column_type(series, str(column))
            if field_type is FieldType.UNKNOWN:
                validity = 1.0
            else:
                validity = float(checks.check_type(series, field_type).mean())
            scores[str(column)] = round((completeness + validity) / 2, 4)
        return scores

    def _recommendations(self, frame: pd.DataFrame, report: QualityReport) -> list[str]:
        advice: list[str] = []
        completeness = report.dimension("completeness")
        if completeness and completeness.score < 0.98:
            worst = completeness.details.get("worst_columns", {})
            if worst:
                names = ", ".join(list(worst)[:3])
                advice.append(f"Fill or drop missing values, worst columns: {names}")
        uniqueness = report.dimension("uniqueness")
        if uniqueness and uniqueness.details.get("duplicate_rows"):
            advice.append(
                f"Remove {uniqueness.details['duplicate_rows']} duplicate rows before loading"
            )
        validity = report.dimension("validity")
        if validity and validity.score < 0.99:
            advice.append("Review type conversions; some values do not match their column type")
        accuracy = report.dimension("accuracy")
        if accuracy and accuracy.details.get("outlier_values"):
            advice.append(
                "Inspect outliers before aggregating; cap or exclude them if they are errors"
            )
        consistency = report.dimension("consistency")
        if consistency and consistency.score < 0.9:
            advice.append("Standardise text formatting (case, spacing, date formats)")
        timeliness = report.dimension("timeliness")
        if timeliness and timeliness.details.get("future_dated_rows"):
            advice.append("Some rows are dated in the future; check the source system clock")
        return advice


def _value_shape(value: Any) -> str:
    """Reduce a value to a coarse format signature."""
    text = str(value)
    if len(text) > 40:
        return f"long:{len(text) // 10}"
    shaped = _SHAPE_DIGIT.sub("9", text)
    shaped = _SHAPE_ALPHA.sub("a", shaped)
    return re.sub(r"(.)\1{2,}", r"\1+", shaped)


def analyze_quality(
    frame: pd.DataFrame,
    schema: Schema | None = None,
    rules: RuleEngine | None = None,
    *,
    name: str = "dataset",
    **kwargs: Any,
) -> QualityReport:
    return QualityAnalyzer(schema, rules, **kwargs).analyze(frame, name=name)
