"""Cleaning operations.

Each function takes a DataFrame and returns a new one plus a small report of
what it changed.  Nothing is done in place: pipelines need the original frame to
stay valid so that a failed step can be retried or skipped.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from universal_data.core.exceptions import TransformationError
from universal_data.core.types import DuplicateKeep, MissingStrategy

_MULTISPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^\w\s\-.@]", re.UNICODE)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

Report = dict[str, Any]


def _target_columns(
    frame: pd.DataFrame, columns: Sequence[str] | None, *, numeric_only: bool = False
) -> list[str]:
    if columns is None:
        if numeric_only:
            return [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        return list(frame.columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TransformationError("Unknown column(s)", columns=missing)
    return list(columns)


def _text_columns(frame: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    candidates = _target_columns(frame, columns)
    return [
        column
        for column in candidates
        if frame[column].dtype == object or str(frame[column].dtype) in ("string", "str")
    ]


def handle_missing(
    frame: pd.DataFrame,
    strategy: MissingStrategy | str = MissingStrategy.NONE,
    *,
    columns: Sequence[str] | None = None,
    value: Any = None,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, Report]:
    """Apply a missing-value strategy.

    ``threshold`` is only used by ``drop_columns``: a column is dropped when the
    share of nulls exceeds it.
    """
    strategy = MissingStrategy(strategy)
    targets = _target_columns(frame, columns)
    before = int(frame[targets].isna().sum().sum())
    result = frame

    if strategy is MissingStrategy.NONE:
        return result, {"operation": "handle_missing", "strategy": str(strategy), "filled": 0}

    if strategy is MissingStrategy.DROP_ROWS:
        result = frame.dropna(subset=targets)
        return result, {
            "operation": "handle_missing",
            "strategy": str(strategy),
            "rows_dropped": len(frame) - len(result),
        }

    if strategy is MissingStrategy.DROP_COLUMNS:
        ratios = frame[targets].isna().mean()
        dropped = [column for column in targets if ratios[column] > threshold]
        result = frame.drop(columns=dropped)
        return result, {
            "operation": "handle_missing",
            "strategy": str(strategy),
            "columns_dropped": dropped,
            "threshold": threshold,
        }

    result = frame.copy()
    filled_per_column: dict[str, int] = {}
    for column in targets:
        series = result[column]
        missing = int(series.isna().sum())
        if not missing:
            continue
        replacement = _replacement_value(series, strategy, value)
        if replacement is _SKIP:
            continue
        if strategy is MissingStrategy.FORWARD_FILL:
            result[column] = series.ffill()
        elif strategy is MissingStrategy.BACKWARD_FILL:
            result[column] = series.bfill()
        else:
            # An integer column cannot hold the float mean/median of itself.
            if pd.api.types.is_integer_dtype(series) and isinstance(replacement, float):
                replacement = int(round(replacement))
            result[column] = series.fillna(replacement)
        filled_per_column[column] = missing - int(result[column].isna().sum())

    return result, {
        "operation": "handle_missing",
        "strategy": str(strategy),
        "missing_before": before,
        "filled": sum(filled_per_column.values()),
        "columns": filled_per_column,
    }


class _Skip:
    """Sentinel: this column cannot use the requested strategy."""


_SKIP = _Skip()


def _replacement_value(series: pd.Series, strategy: MissingStrategy, value: Any) -> Any:
    if strategy is MissingStrategy.CONSTANT:
        if value is None:
            raise TransformationError("The 'constant' strategy needs a value")
        return value
    if strategy in (MissingStrategy.FORWARD_FILL, MissingStrategy.BACKWARD_FILL):
        return None
    if strategy is MissingStrategy.MODE:
        mode = series.mode(dropna=True)
        return mode.iloc[0] if not mode.empty else _SKIP
    if not pd.api.types.is_numeric_dtype(series):
        # Mean/median are meaningless for text; fall back to the mode.
        mode = series.mode(dropna=True)
        return mode.iloc[0] if not mode.empty else _SKIP
    if strategy is MissingStrategy.MEAN:
        return series.mean()
    if strategy is MissingStrategy.MEDIAN:
        return series.median()
    return _SKIP


def remove_duplicates(
    frame: pd.DataFrame,
    *,
    subset: Sequence[str] | None = None,
    keep: DuplicateKeep | str = DuplicateKeep.FIRST,
) -> tuple[pd.DataFrame, Report]:
    """Drop duplicate rows and report how many were removed."""
    keep_mode = DuplicateKeep(keep)
    if subset:
        _target_columns(frame, subset)
    pandas_keep: Any = False if keep_mode is DuplicateKeep.NONE else str(keep_mode)
    duplicated = frame.duplicated(subset=list(subset) if subset else None, keep=False)
    result = frame.drop_duplicates(subset=list(subset) if subset else None, keep=pandas_keep)
    return result, {
        "operation": "remove_duplicates",
        "keep": str(keep_mode),
        "subset": list(subset) if subset else None,
        "duplicate_rows": int(duplicated.sum()),
        "rows_removed": len(frame) - len(result),
    }


def find_duplicates(
    frame: pd.DataFrame, *, subset: Sequence[str] | None = None
) -> pd.DataFrame:
    """Return every row that participates in a duplicate group."""
    mask = frame.duplicated(subset=list(subset) if subset else None, keep=False)
    return frame.loc[mask]


def clean_strings(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    trim: bool = True,
    normalize_spaces: bool = True,
    case: str | None = None,
    remove_special: bool = False,
    normalize_unicode: bool = False,
    replacements: Sequence[tuple[str, str]] | None = None,
    empty_as_null: bool = True,
) -> tuple[pd.DataFrame, Report]:
    """Normalise text columns."""
    if case not in (None, "lower", "upper", "title"):
        raise TransformationError("case must be one of: lower, upper, title", given=case)
    targets = _text_columns(frame, columns)
    if not targets:
        return frame, {"operation": "clean_strings", "columns": [], "cells_changed": 0}

    result = frame.copy()
    changed = 0
    for column in targets:
        original = result[column]
        text = original.astype("string")
        if normalize_unicode:
            text = text.map(
                lambda v: unicodedata.normalize("NFKC", v) if isinstance(v, str) else v
            )
        if trim:
            text = text.str.strip()
        if normalize_spaces:
            text = text.str.replace(_MULTISPACE, " ", regex=True)
        if remove_special:
            text = text.str.replace(_NON_ALNUM, "", regex=True)
        for pattern, replacement in replacements or ():
            text = text.str.replace(pattern, replacement, regex=True)
        if case == "lower":
            text = text.str.lower()
        elif case == "upper":
            text = text.str.upper()
        elif case == "title":
            text = text.str.title()
        if empty_as_null:
            text = text.replace("", pd.NA)
        changed += int((text.fillna("\x00") != original.astype("string").fillna("\x00")).sum())
        result[column] = text
    return result, {
        "operation": "clean_strings",
        "columns": targets,
        "cells_changed": changed,
        "case": case,
    }


def normalize_column_names(frame: pd.DataFrame) -> tuple[pd.DataFrame, Report]:
    """Rename columns to lower snake_case."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for column in frame.columns:
        name = _CAMEL_BOUNDARY.sub("_", str(column)).strip()
        name = _MULTISPACE.sub("_", name)
        name = re.sub(r"[^\w]+", "_", name, flags=re.UNICODE).strip("_").lower()
        name = re.sub(r"_{2,}", "_", name) or "column"
        candidate = name
        index = 1
        while candidate in used:
            index += 1
            candidate = f"{name}_{index}"
        used.add(candidate)
        mapping[str(column)] = candidate
    renamed = {old: new for old, new in mapping.items() if old != new}
    return frame.rename(columns=mapping), {
        "operation": "normalize_column_names",
        "renamed": renamed,
    }


def convert_types(
    frame: pd.DataFrame, mapping: dict[str, str], *, errors: str = "coerce"
) -> tuple[pd.DataFrame, Report]:
    """Convert columns to the requested dtypes, coercing failures to null."""
    _target_columns(frame, list(mapping))
    result = frame.copy()
    failures: dict[str, int] = {}
    for column, target in mapping.items():
        series = result[column]
        before_null = int(series.isna().sum())
        if target in ("int", "integer", "int64", "Int64"):
            converted = pd.to_numeric(series, errors=errors)
            converted = converted.round().astype("Int64")
        elif target in ("float", "float64", "double"):
            converted = pd.to_numeric(series, errors=errors).astype("Float64")
        elif target in ("bool", "boolean"):
            converted = _to_boolean(series)
        elif target in ("datetime", "date", "timestamp"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_datetime(series, errors=errors, format="mixed")
            if target == "date":
                converted = converted.dt.normalize()
        elif target in ("str", "string", "text"):
            converted = series.astype("string")
        elif target in ("category", "categorical"):
            converted = series.astype("category")
        else:
            raise TransformationError("Unsupported target type", column=column, type=target)
        lost = int(converted.isna().sum()) - before_null
        if lost > 0:
            failures[column] = lost
        result[column] = converted
    return result, {"operation": "convert_types", "mapping": mapping, "coerced_to_null": failures}


def _to_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    truthy = {"true", "yes", "y", "t", "1", "on"}
    falsy = {"false", "no", "n", "f", "0", "off"}
    text = series.astype("string").str.strip().str.lower()
    return text.map(lambda v: True if v in truthy else (False if v in falsy else pd.NA)).astype(
        "boolean"
    )


def parse_dates(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_format: str | None = None,
    dayfirst: bool = False,
    timezone: str | None = None,
) -> tuple[pd.DataFrame, Report]:
    """Parse date columns, optionally normalising to a timezone."""
    targets = _target_columns(frame, columns)
    result = frame.copy()
    invalid: dict[str, int] = {}
    for column in targets:
        series = result[column]
        before_null = int(series.isna().sum())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(
                series,
                errors="coerce",
                dayfirst=dayfirst,
                format=date_format if date_format else "mixed",
            )
        if timezone:
            parsed = (
                parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
                if parsed.dt.tz is None
                else parsed.dt.tz_convert(timezone)
            )
        failed = int(parsed.isna().sum()) - before_null
        if failed > 0:
            invalid[column] = failed
        result[column] = parsed
    return result, {"operation": "parse_dates", "columns": targets, "unparseable": invalid}


def clean_numeric(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    strip_symbols: bool = True,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[pd.DataFrame, Report]:
    """Coerce columns to numbers and null out impossible values."""
    targets = _target_columns(frame, columns)
    result = frame.copy()
    coerced: dict[str, int] = {}
    out_of_range: dict[str, int] = {}
    for column in targets:
        series = result[column]
        if strip_symbols and not pd.api.types.is_numeric_dtype(series):
            series = (
                series.astype("string")
                .str.replace(r"[,\s]", "", regex=True)
                .str.replace(r"^[^\d\-+.]+", "", regex=True)
                .str.replace(r"[^\d]+$", "", regex=True)
            )
        numeric = pd.to_numeric(series, errors="coerce")
        lost = int(numeric.isna().sum()) - int(result[column].isna().sum())
        if lost > 0:
            coerced[column] = lost
        if minimum is not None or maximum is not None:
            invalid = pd.Series(False, index=numeric.index)
            if minimum is not None:
                invalid |= numeric < minimum
            if maximum is not None:
                invalid |= numeric > maximum
            count = int(invalid.sum())
            if count:
                out_of_range[column] = count
                numeric = numeric.mask(invalid, np.nan)
        result[column] = numeric
    return result, {
        "operation": "clean_numeric",
        "columns": targets,
        "coerced_to_null": coerced,
        "out_of_range": out_of_range,
    }


def drop_empty(
    frame: pd.DataFrame, *, rows: bool = True, columns: bool = True
) -> tuple[pd.DataFrame, Report]:
    """Remove fully empty rows and columns."""
    result = frame
    dropped_columns: list[str] = []
    rows_dropped = 0
    if columns and not result.empty:
        empty = [column for column in result.columns if result[column].isna().all()]
        dropped_columns = empty
        result = result.drop(columns=empty)
    if rows:
        before = len(result)
        result = result.dropna(how="all")
        rows_dropped = before - len(result)
    return result, {
        "operation": "drop_empty",
        "columns_dropped": dropped_columns,
        "rows_dropped": rows_dropped,
    }
