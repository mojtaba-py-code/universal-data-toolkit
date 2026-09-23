"""Column level checks.

Every check returns a boolean Series that is ``True`` for values that satisfy
the expectation.  Null values are always reported as valid here; nullability is a
separate check, otherwise a single missing value would be counted twice.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pandas as pd

from universal_data.core.types import FieldType
from universal_data.schema.detect import PATTERNS

_BOOLEAN_TOKENS = {
    "true", "false", "yes", "no", "y", "n", "t", "f", "1", "0", "on", "off",
}


def _keep_nulls(series: pd.Series, mask: pd.Series) -> pd.Series:
    return mask | series.isna()


def check_integer(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series):
        return pd.Series(True, index=series.index)
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & (numeric % 1 == 0)
    return _keep_nulls(series, valid)


def check_float(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return pd.Series(True, index=series.index)
    return _keep_nulls(series, pd.to_numeric(series, errors="coerce").notna())


def check_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return pd.Series(True, index=series.index)
    tokens = series.astype("string").str.strip().str.lower()
    return _keep_nulls(series, tokens.isin(_BOOLEAN_TOKENS).fillna(False))


def check_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.Series(True, index=series.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return _keep_nulls(series, parsed.notna())


def check_string(series: pd.Series) -> pd.Series:
    invalid = series.map(lambda value: isinstance(value, (list, dict, set, tuple)))
    return _keep_nulls(series, ~invalid.fillna(False).astype(bool))


def _pattern_check(field_type: FieldType) -> Callable[[pd.Series], pd.Series]:
    pattern = PATTERNS[field_type]

    def check(series: pd.Series) -> pd.Series:
        text = series.astype("string").str.strip()
        return _keep_nulls(series, text.str.match(pattern, na=False).fillna(False))

    return check


TYPE_CHECKS: dict[FieldType, Callable[[pd.Series], pd.Series]] = {
    FieldType.INTEGER: check_integer,
    FieldType.FLOAT: check_float,
    FieldType.BOOLEAN: check_boolean,
    FieldType.DATETIME: check_datetime,
    FieldType.DATE: check_datetime,
    FieldType.STRING: check_string,
    FieldType.CATEGORICAL: check_string,
    FieldType.EMAIL: _pattern_check(FieldType.EMAIL),
    FieldType.URL: _pattern_check(FieldType.URL),
    FieldType.UUID: _pattern_check(FieldType.UUID),
    FieldType.IP_ADDRESS: _pattern_check(FieldType.IP_ADDRESS),
    FieldType.PHONE: _pattern_check(FieldType.PHONE),
    FieldType.UNKNOWN: lambda series: pd.Series(True, index=series.index),
}


def check_type(series: pd.Series, field_type: FieldType) -> pd.Series:
    return TYPE_CHECKS[field_type](series)


def comparable(series: pd.Series, field_type: FieldType) -> pd.Series:
    """Coerce a column so that range comparisons make sense."""
    if field_type.is_temporal:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(series, errors="coerce", format="mixed")
    return pd.to_numeric(series, errors="coerce")


def bound_value(bound: object, field_type: FieldType) -> object:
    if field_type.is_temporal:
        return pd.Timestamp(str(bound))
    return float(bound)  # type: ignore[arg-type]


def check_min(series: pd.Series, minimum: object, field_type: FieldType) -> pd.Series:
    values = comparable(series, field_type)
    return _keep_nulls(series, (values >= bound_value(minimum, field_type)).fillna(False))


def check_max(series: pd.Series, maximum: object, field_type: FieldType) -> pd.Series:
    values = comparable(series, field_type)
    return _keep_nulls(series, (values <= bound_value(maximum, field_type)).fillna(False))


def check_pattern(series: pd.Series, pattern: str) -> pd.Series:
    text = series.astype("string")
    return _keep_nulls(series, text.str.fullmatch(pattern, na=False).fillna(False))


def check_length(
    series: pd.Series, minimum: int | None = None, maximum: int | None = None
) -> pd.Series:
    lengths = series.astype("string").str.len()
    valid = pd.Series(True, index=series.index)
    if minimum is not None:
        valid &= (lengths >= minimum).fillna(False)
    if maximum is not None:
        valid &= (lengths <= maximum).fillna(False)
    return _keep_nulls(series, valid)


def check_allowed(series: pd.Series, allowed: list[object]) -> pd.Series:
    if not allowed:
        return pd.Series(True, index=series.index)
    direct = series.isin(allowed)
    # Enum values written in YAML are strings; compare on text as a fallback.
    as_text = series.astype("string").isin([str(value) for value in allowed])
    return _keep_nulls(series, (direct | as_text).fillna(False))


def check_unique(series: pd.Series) -> pd.Series:
    duplicated = series.duplicated(keep=False) & series.notna()
    return ~duplicated
