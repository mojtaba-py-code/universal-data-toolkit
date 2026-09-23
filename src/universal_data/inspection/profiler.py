"""Dataset profiling.

Profiling answers the first question anybody asks about an unfamiliar file:
what is in it, how much of it is missing, and which columns look wrong.
Statistics are computed with vectorised pandas calls; the only Python-level loop
is over columns, not rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from universal_data.core.types import FieldType
from universal_data.observability.memory import dataframe_memory_mb
from universal_data.quality.outliers import detect_outliers
from universal_data.schema.detect import detect_column_type, looks_like_pii

TOP_VALUES = 5


@dataclass
class NumericStats:
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    p25: float
    p75: float
    p95: float
    zeros: int
    negatives: int
    skew: float

    def to_dict(self) -> dict[str, Any]:
        return {key: _round(value) for key, value in asdict(self).items()}


@dataclass
class CategoricalStats:
    distinct: int
    top_values: list[tuple[Any, int]]
    empty_strings: int
    average_length: float
    max_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "distinct": self.distinct,
            "top_values": [{"value": value, "count": count} for value, count in self.top_values],
            "empty_strings": self.empty_strings,
            "average_length": _round(self.average_length),
            "max_length": self.max_length,
        }


@dataclass
class TemporalStats:
    earliest: str | None
    latest: str | None
    range_days: float | None
    future_values: int
    unparseable: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    inferred_type: FieldType
    count: int
    missing: int
    unique: int
    memory_bytes: int
    is_pii: bool = False
    outliers: int = 0
    numeric: NumericStats | None = None
    categorical: CategoricalStats | None = None
    temporal: TemporalStats | None = None
    samples: list[Any] = field(default_factory=list)

    @property
    def missing_pct(self) -> float:
        total = self.count + self.missing
        return (self.missing / total * 100) if total else 0.0

    @property
    def unique_pct(self) -> float:
        return (self.unique / self.count * 100) if self.count else 0.0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "dtype": self.dtype,
            "inferred_type": str(self.inferred_type),
            "count": self.count,
            "missing": self.missing,
            "missing_pct": _round(self.missing_pct),
            "unique": self.unique,
            "unique_pct": _round(self.unique_pct),
            "memory_bytes": self.memory_bytes,
            "is_pii": self.is_pii,
            "outliers": self.outliers,
            "samples": self.samples,
        }
        if self.numeric:
            data["numeric"] = self.numeric.to_dict()
        if self.categorical:
            data["categorical"] = self.categorical.to_dict()
        if self.temporal:
            data["temporal"] = self.temporal.to_dict()
        return data


@dataclass
class DatasetProfile:
    """Full profile of one dataset."""

    name: str
    rows: int
    columns: int
    memory_mb: float
    duplicate_rows: int
    columns_profile: list[ColumnProfile] = field(default_factory=list)
    source: str | None = None

    @property
    def duplicate_pct(self) -> float:
        return (self.duplicate_rows / self.rows * 100) if self.rows else 0.0

    @property
    def total_missing(self) -> int:
        return sum(column.missing for column in self.columns_profile)

    @property
    def missing_pct(self) -> float:
        cells = self.rows * self.columns
        return (self.total_missing / cells * 100) if cells else 0.0

    @property
    def pii_columns(self) -> list[str]:
        return [column.name for column in self.columns_profile if column.is_pii]

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns_profile if c.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "rows": self.rows,
            "columns": self.columns,
            "memory_mb": _round(self.memory_mb),
            "duplicate_rows": self.duplicate_rows,
            "duplicate_pct": _round(self.duplicate_pct),
            "missing_cells": self.total_missing,
            "missing_pct": _round(self.missing_pct),
            "pii_columns": self.pii_columns,
            "columns_profile": [column.to_dict() for column in self.columns_profile],
        }

    def summary(self) -> str:
        lines = [
            f"Dataset Profile: {self.name}",
            "",
            f"Shape:          {self.rows:,} rows x {self.columns} columns",
            f"Memory:         {self.memory_mb:.2f} MB",
            f"Missing cells:  {self.total_missing:,} ({self.missing_pct:.1f}%)",
            f"Duplicate rows: {self.duplicate_rows:,} ({self.duplicate_pct:.1f}%)",
        ]
        if self.pii_columns:
            lines.append(f"Possible PII:   {', '.join(self.pii_columns)}")
        lines.extend(["", "Columns:"])
        width = max((len(c.name) for c in self.columns_profile), default=6)
        for column in self.columns_profile:
            lines.append(
                f"  {column.name.ljust(width)}  {str(column.inferred_type):<11} "
                f"missing={column.missing_pct:5.1f}%  unique={column.unique:,}"
            )
        return "\n".join(lines)


def _round(value: Any) -> Any:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return round(float(value), 4)
    return value


def _numeric_stats(series: pd.Series) -> NumericStats:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        nan = float("nan")
        return NumericStats(
            mean=nan,
            median=nan,
            std=nan,
            minimum=nan,
            maximum=nan,
            p25=nan,
            p75=nan,
            p95=nan,
            zeros=0,
            negatives=0,
            skew=nan,
        )
    return NumericStats(
        mean=float(values.mean()),
        median=float(values.median()),
        std=float(values.std()) if len(values) > 1 else 0.0,
        minimum=float(values.min()),
        maximum=float(values.max()),
        p25=float(values.quantile(0.25)),
        p75=float(values.quantile(0.75)),
        p95=float(values.quantile(0.95)),
        zeros=int((values == 0).sum()),
        negatives=int((values < 0).sum()),
        skew=float(values.skew()) if len(values) > 2 else 0.0,
    )


def _categorical_stats(series: pd.Series) -> CategoricalStats:
    values = series.dropna()
    counts = values.value_counts().head(TOP_VALUES)
    text = values.astype("string")
    lengths = text.str.len()
    return CategoricalStats(
        distinct=int(values.nunique()),
        top_values=[(_json_value(index), int(count)) for index, count in counts.items()],
        empty_strings=int((text.str.strip() == "").sum()),
        average_length=float(lengths.mean()) if not lengths.empty else 0.0,
        max_length=int(lengths.max()) if not lengths.empty else 0,
    )


def _temporal_stats(series: pd.Series) -> TemporalStats:
    import warnings

    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    valid = parsed.dropna()
    unparseable = int(parsed.isna().sum() - series.isna().sum())
    if valid.empty:
        return TemporalStats(None, None, None, 0, max(unparseable, 0))
    now = pd.Timestamp.now(tz=valid.dt.tz) if valid.dt.tz is not None else pd.Timestamp.now()
    return TemporalStats(
        earliest=str(valid.min()),
        latest=str(valid.max()),
        range_days=float((valid.max() - valid.min()).days),
        future_values=int((valid > now).sum()),
        unparseable=max(unparseable, 0),
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def profile_column(series: pd.Series, name: str, *, with_outliers: bool = True) -> ColumnProfile:
    inferred = detect_column_type(series, name)
    profile = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        inferred_type=inferred,
        count=int(series.notna().sum()),
        missing=int(series.isna().sum()),
        unique=int(series.nunique(dropna=True)),
        memory_bytes=int(series.memory_usage(deep=True)),
        is_pii=looks_like_pii(name, inferred),
        samples=[_json_value(value) for value in series.dropna().head(3).tolist()],
    )
    if inferred.is_numeric:
        profile.numeric = _numeric_stats(series)
        if with_outliers and profile.count > 3:
            profile.outliers = int(detect_outliers(series).sum())
    elif inferred.is_temporal:
        profile.temporal = _temporal_stats(series)
    else:
        profile.categorical = _categorical_stats(series)
    return profile


def profile_dataset(
    frame: pd.DataFrame,
    *,
    name: str = "dataset",
    source: str | None = None,
    with_outliers: bool = True,
) -> DatasetProfile:
    """Build a :class:`DatasetProfile` for *frame*."""
    profile = DatasetProfile(
        name=name,
        rows=len(frame),
        columns=len(frame.columns),
        memory_mb=dataframe_memory_mb(frame),
        duplicate_rows=int(frame.duplicated().sum()) if len(frame) else 0,
        source=source,
    )
    profile.columns_profile = [
        profile_column(frame[column], str(column), with_outliers=with_outliers)
        for column in frame.columns
    ]
    return profile
