"""Schema inference.

Detection works on a sample rather than the whole column: 5,000 non-null values
are enough to decide a type with high confidence and keep inference cheap on
large datasets.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import pandas as pd

from universal_data.core.types import FieldType
from universal_data.schema.model import ColumnSchema, Schema

SAMPLE_SIZE = 5_000
MATCH_THRESHOLD = 0.9
CATEGORICAL_MAX_UNIQUE = 50
CATEGORICAL_MAX_RATIO = 0.5

PATTERNS: dict[FieldType, re.Pattern[str]] = {
    FieldType.EMAIL: re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"),
    FieldType.URL: re.compile(r"^https?://[^\s/$.?#][^\s]*$", re.IGNORECASE),
    FieldType.UUID: re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
    ),
    FieldType.IP_ADDRESS: re.compile(r"^(\d{1,3}\.){3}\d{1,3}$|^[0-9a-f:]{3,45}$", re.IGNORECASE),
    FieldType.PHONE: re.compile(r"^\+?[\d\s().\-]{7,20}$"),
}

# Column names that mark personal data even when the values look ordinary.
# The boundaries are written by hand rather than with \b so that an underscore
# separates words: \bphone\b would not match "phone_number".
PII_NAME_PATTERNS = re.compile(
    r"(?i)(?<![a-z0-9])(email|e_mail|mail|phone|mobile|tel|telephone|ssn|social_security|"
    r"passport|national_id|nid|iban|card|credit_card|cvv|address|street|postcode|zip|birth|"
    r"dob|first_name|last_name|full_name|surname|latitude|longitude|ip_address)(?![a-z0-9])"
)

PII_VALUE_TYPES = frozenset(
    {FieldType.EMAIL, FieldType.PHONE, FieldType.IP_ADDRESS}
)

IDENTIFIER_NAME = re.compile(r"(?i)(^id$|_id$|^id_|uuid|guid|code$|key$|number$|no$)")

_BOOLEAN_TOKENS = {
    "true", "false", "yes", "no", "y", "n", "t", "f", "1", "0", "on", "off",
}


@dataclass
class ColumnProfileHints:
    """Signals collected while inferring a single column."""

    field_type: FieldType
    nullable: bool
    unique: bool
    is_identifier: bool
    pii: bool
    null_ratio: float
    unique_ratio: float


def _sample(series: pd.Series, size: int = SAMPLE_SIZE) -> pd.Series:
    values = series.dropna()
    if len(values) > size:
        return values.head(size)
    return values


def _matches_pattern(values: pd.Series, pattern: re.Pattern[str]) -> float:
    if values.empty:
        return 0.0
    text = values.astype("string").str.strip()
    matched = text.str.match(pattern, na=False)
    return float(matched.mean())


def _looks_like_datetime(values: pd.Series) -> bool:
    if values.empty:
        return False
    text = values.astype("string").str.strip()
    # Cheap pre-filter: at least one separator that dates normally carry.
    if not text.str.contains(r"[-/:]", regex=True, na=False).mean() > 0.8:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(text, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            return False
    return float(parsed.notna().mean()) >= MATCH_THRESHOLD


def _looks_like_boolean(values: pd.Series) -> bool:
    if values.empty:
        return False
    tokens = set(values.astype("string").str.strip().str.lower().dropna().unique())
    return bool(tokens) and tokens <= _BOOLEAN_TOKENS and len(tokens) <= 4


def _looks_numeric(values: pd.Series) -> tuple[bool, bool]:
    """Return ``(is_numeric, is_integer)`` for a text column."""
    if values.empty:
        return False, False
    cleaned = values.astype("string").str.strip().str.replace(",", "", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if float(numeric.notna().mean()) < MATCH_THRESHOLD:
        return False, False
    non_null = numeric.dropna()
    is_integer = bool(non_null.empty or (non_null % 1 == 0).all())
    return True, is_integer


def detect_column_type(series: pd.Series, name: str = "") -> FieldType:
    """Infer the logical type of one column."""
    if pd.api.types.is_bool_dtype(series):
        return FieldType.BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(series):
        return FieldType.DATETIME
    if pd.api.types.is_integer_dtype(series):
        return FieldType.INTEGER
    if pd.api.types.is_float_dtype(series):
        values = series.dropna()
        if not values.empty and (values % 1 == 0).all() and values.abs().max() < 2**53:
            return FieldType.INTEGER
        return FieldType.FLOAT
    if isinstance(series.dtype, pd.CategoricalDtype):
        return FieldType.CATEGORICAL

    sample = _sample(series)
    if sample.empty:
        return FieldType.UNKNOWN

    for field_type, pattern in PATTERNS.items():
        if field_type is FieldType.PHONE:
            continue  # too permissive to test before the numeric check
        if _matches_pattern(sample, pattern) >= MATCH_THRESHOLD:
            return field_type

    if _looks_like_boolean(sample):
        return FieldType.BOOLEAN
    if _looks_like_datetime(sample):
        return FieldType.DATETIME

    # Phone numbers parse as integers, so the name has to be consulted first.
    lowered = name.lower()
    if ("phone" in lowered or "mobile" in lowered or lowered.endswith("_tel")) and (
        _matches_pattern(sample, PATTERNS[FieldType.PHONE]) >= MATCH_THRESHOLD
    ):
        return FieldType.PHONE

    is_numeric, is_integer = _looks_numeric(sample)
    if is_numeric:
        return FieldType.INTEGER if is_integer else FieldType.FLOAT

    unique_ratio = sample.nunique() / len(sample)
    if sample.nunique() <= CATEGORICAL_MAX_UNIQUE and unique_ratio <= CATEGORICAL_MAX_RATIO:
        return FieldType.CATEGORICAL
    return FieldType.STRING


def looks_like_pii(name: str, field_type: FieldType) -> bool:
    return bool(PII_NAME_PATTERNS.search(name)) or field_type in PII_VALUE_TYPES


def profile_column(series: pd.Series, name: str) -> ColumnProfileHints:
    total = len(series)
    non_null = int(series.notna().sum())
    null_ratio = 1 - (non_null / total) if total else 0.0
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = (unique_count / non_null) if non_null else 0.0
    field_type = detect_column_type(series, name)
    is_identifier = bool(
        non_null > 0
        and unique_ratio >= 0.99
        and (IDENTIFIER_NAME.search(name) or field_type in (FieldType.INTEGER, FieldType.UUID))
    )
    return ColumnProfileHints(
        field_type=field_type,
        nullable=null_ratio > 0,
        unique=non_null > 0 and unique_count == non_null,
        is_identifier=is_identifier,
        pii=looks_like_pii(name, field_type),
        null_ratio=null_ratio,
        unique_ratio=unique_ratio,
    )


def detect_schema(
    frame: pd.DataFrame,
    *,
    name: str = "detected_schema",
    infer_constraints: bool = False,
) -> Schema:
    """Build a :class:`Schema` from a DataFrame.

    ``infer_constraints`` also fills numeric bounds and enum values from the
    observed data.  That is useful as a starting point for a hand-written
    schema, but it should not be used as a validation contract directly: the
    bounds describe *this* sample, not the business rule.
    """
    columns: list[ColumnSchema] = []
    for column_name in frame.columns:
        series = frame[column_name]
        hints = profile_column(series, str(column_name))
        column = ColumnSchema(
            name=str(column_name),
            type=hints.field_type,
            required=True,
            nullable=hints.nullable,
            unique=hints.unique and hints.is_identifier,
            pii=hints.pii,
        )
        if infer_constraints:
            _fill_constraints(column, series, hints)
        columns.append(column)

    primary_key = [
        column.name
        for column in columns
        if column.unique and not column.nullable and column.type is not FieldType.FLOAT
    ][:1]
    return Schema.from_columns(columns, name=name, primary_key=primary_key)


def _fill_constraints(column: ColumnSchema, series: pd.Series, hints: ColumnProfileHints) -> None:
    if hints.field_type.is_numeric:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if not numeric.empty:
            column.min = float(numeric.min())
            column.max = float(numeric.max())
            if hints.field_type is FieldType.INTEGER:
                column.min = int(column.min)
                column.max = int(column.max)
    elif hints.field_type is FieldType.CATEGORICAL:
        values = series.dropna().unique().tolist()
        if len(values) <= CATEGORICAL_MAX_UNIQUE:
            column.allowed = sorted(str(value) for value in values)
    elif hints.field_type in (FieldType.STRING, FieldType.EMAIL):
        lengths = series.dropna().astype("string").str.len()
        if not lengths.empty:
            column.max_length = int(lengths.max())
