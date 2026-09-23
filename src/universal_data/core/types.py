"""Shared enums and type aliases."""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Any

type PathLike = str | os.PathLike[str]
type Row = dict[str, Any]
type Records = list[Row]
type JSONValue = Any


class FileFormat(StrEnum):
    """Formats the ingestion and export layers know about."""

    CSV = "csv"
    TSV = "tsv"
    EXCEL = "excel"
    JSON = "json"
    JSONL = "jsonl"
    XML = "xml"
    YAML = "yaml"
    PARQUET = "parquet"
    FEATHER = "feather"
    PICKLE = "pickle"
    SQLITE = "sqlite"


class FieldType(StrEnum):
    """Logical column types used by schema detection and validation.

    These are deliberately coarser than pandas dtypes: they describe intent
    (``email``) rather than storage (``object``).
    """

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    DATE = "date"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    UUID = "uuid"
    IP_ADDRESS = "ip"
    UNKNOWN = "unknown"

    @property
    def is_numeric(self) -> bool:
        return self in (FieldType.INTEGER, FieldType.FLOAT)

    @property
    def is_temporal(self) -> bool:
        return self in (FieldType.DATETIME, FieldType.DATE)

    @property
    def is_textual(self) -> bool:
        return self in (
            FieldType.STRING,
            FieldType.CATEGORICAL,
            FieldType.EMAIL,
            FieldType.URL,
            FieldType.PHONE,
            FieldType.UUID,
            FieldType.IP_ADDRESS,
        )


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class MissingStrategy(StrEnum):
    """How to deal with missing values during cleaning."""

    DROP_ROWS = "drop_rows"
    DROP_COLUMNS = "drop_columns"
    CONSTANT = "constant"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    NONE = "none"


class DuplicateKeep(StrEnum):
    FIRST = "first"
    LAST = "last"
    NONE = "none"


class OutlierMethod(StrEnum):
    IQR = "iqr"
    ZSCORE = "zscore"
    STDDEV = "stddev"
    PERCENTILE = "percentile"


class OutlierAction(StrEnum):
    REPORT = "report"
    REMOVE = "remove"
    CAP = "cap"
    REPLACE = "replace"
