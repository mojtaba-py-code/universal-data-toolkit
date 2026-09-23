"""File readers.

Each reader owns exactly one format.  Detection defaults (encoding, delimiter)
are resolved here so that ``Dataset.from_csv("x.csv")`` works on a Windows-1252
semicolon file without the caller passing anything.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import yaml

from universal_data.core.exceptions import ExtractionError, SecurityError
from universal_data.core.types import FileFormat, PathLike
from universal_data.ingestion.base import Reader, register_reader
from universal_data.ingestion.detect import detect_delimiter, detect_encoding
from universal_data.observability.logging import get_logger

logger = get_logger(__name__)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Ragged rows are the most common defect in a hand-edited export, so the error
# points at the way out instead of leaving the caller to find it.
_PARSER_HINT = (
    "rows have inconsistent field counts; pass on_bad_lines='skip' (or 'warn') "
    "to drop them, or check the separator"
)


def _records_to_frame(payload: Any, *, source: str, record_path: str | None = None) -> pd.DataFrame:
    """Turn a decoded JSON/YAML document into a tabular frame."""
    if record_path:
        for key in record_path.split("."):
            if not isinstance(payload, dict) or key not in payload:
                raise ExtractionError(
                    "record_path does not exist in the document", path=record_path, source=source
                )
            payload = payload[key]

    if isinstance(payload, list):
        if payload and not isinstance(payload[0], dict):
            return pd.DataFrame({"value": payload})
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        # A single wrapper key holding the records is the common API shape.
        list_values = [value for value in payload.values() if isinstance(value, list)]
        if len(list_values) == 1 and len(payload) <= 3:
            return pd.json_normalize(list_values[0])
        return pd.json_normalize(payload)
    raise ExtractionError("Document does not contain tabular data", source=source)


@register_reader
class CSVReader(Reader):
    format: ClassVar[FileFormat] = FileFormat.CSV
    extensions: ClassVar[tuple[str, ...]] = (".csv", ".txt")
    supports_chunking: ClassVar[bool] = True
    default_separator: ClassVar[str | None] = None

    def _options(self, path: Path, options: dict[str, Any]) -> dict[str, Any]:
        settings = dict(options)
        settings.setdefault("encoding", detect_encoding(path))
        if self.default_separator is not None:
            settings.setdefault("sep", self.default_separator)
        else:
            settings.setdefault("sep", detect_delimiter(path, settings["encoding"]))
        settings.setdefault("skipinitialspace", True)
        return settings

    def read(self, source: PathLike, **options: Any) -> pd.DataFrame:
        """Read the file.

        Rows with the wrong number of fields are an error by default, because
        silently dropping data is worse than stopping.  Pass
        ``on_bad_lines="skip"`` (or ``"warn"``) to salvage what is readable.
        """
        path = self.resolve(source)
        settings = self._options(path, options)
        try:
            return pd.read_csv(path, **settings)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise ExtractionError(
                f"Could not parse the delimited file: {exc}",
                path=str(path),
                encoding=settings.get("encoding"),
                separator=settings.get("sep"),
                hint=_PARSER_HINT if isinstance(exc, pd.errors.ParserError) else None,
            ) from exc

    def read_chunks(
        self, source: PathLike, chunk_size: int, **options: Any
    ) -> Iterator[pd.DataFrame]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        path = self.resolve(source)
        settings = self._options(path, options)
        try:
            with pd.read_csv(path, chunksize=chunk_size, **settings) as reader:
                yield from reader
        except pd.errors.EmptyDataError:
            return
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise ExtractionError(
                f"Could not parse the delimited file: {exc}", path=str(path)
            ) from exc


@register_reader
class TSVReader(CSVReader):
    format: ClassVar[FileFormat] = FileFormat.TSV
    extensions: ClassVar[tuple[str, ...]] = (".tsv", ".tab")
    default_separator: ClassVar[str | None] = "\t"


@register_reader
class ExcelReader(Reader):
    format: ClassVar[FileFormat] = FileFormat.EXCEL
    extensions: ClassVar[tuple[str, ...]] = (".xlsx", ".xlsm", ".xls")
    binary: ClassVar[bool] = True

    def read(self, source: PathLike, **options: Any) -> pd.DataFrame:
        path = self.resolve(source)
        options.setdefault("sheet_name", 0)
        try:
            frame = pd.read_excel(path, **options)
        except ValueError as exc:
            raise ExtractionError(f"Could not read the workbook: {exc}", path=str(path)) from exc
        if isinstance(frame, dict):  # sheet_name=None returns every sheet
            frame = pd.concat(
                [sheet.assign(_sheet=name) for name, sheet in frame.items()], ignore_index=True
            )
        return frame

    def sheet_names(self, source: PathLike) -> list[str]:
        path = self.resolve(source)
        with pd.ExcelFile(path) as workbook:
            return list(workbook.sheet_names)


@register_reader
class JSONReader(Reader):
    format: ClassVar[FileFormat] = FileFormat.JSON
    extensions: ClassVar[tuple[str, ...]] = (".json",)

    def read(self, source: PathLike, **options: Any) -> pd.DataFrame:
        path = self.resolve(source)
        encoding = options.pop("encoding", None) or detect_encoding(path)
        record_path = options.pop("record_path", None)
        try:
            payload = json.loads(path.read_text(encoding=encoding))
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"Invalid JSON: {exc.msg}", path=str(path), line=exc.lineno, column=exc.colno
            ) from exc
        return _records_to_frame(payload, source=str(path), record_path=record_path)


@register_reader
class JSONLinesReader(Reader):
    format: ClassVar[FileFormat] = FileFormat.JSONL
    extensions: ClassVar[tuple[str, ...]] = (".jsonl", ".ndjson")
    supports_chunking: ClassVar[bool] = True

    def read(self, source: PathLike, **options: Any) -> pd.DataFrame:
        path = self.resolve(source)
        encoding = options.pop("encoding", None) or detect_encoding(path)
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding=encoding) as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ExtractionError(
                        f"Invalid JSON on line {line_number}: {exc.msg}", path=str(path)
                    ) from exc
        return pd.json_normalize(rows) if rows else pd.DataFrame()

    def read_chunks(
        self, source: PathLike, chunk_size: int, **options: Any
    ) -> Iterator[pd.DataFrame]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        path = self.resolve(source)
        encoding = options.pop("encoding", None) or detect_encoding(path)
        buffer: list[dict[str, Any]] = []
        with path.open("r", encoding=encoding) as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    buffer.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ExtractionError(
                        f"Invalid JSON on line {line_number}: {exc.msg}", path=str(path)
                    ) from exc
                if len(buffer) >= chunk_size:
                    yield pd.json_normalize(buffer)
                    buffer = []
        if buffer:
            yield pd.json_normalize(buffer)


@register_reader
class XMLReader(Reader):
    format: ClassVar[FileFormat] = FileFormat.XML
    extensions: ClassVar[tuple[str, ...]] = (".xml",)

    def read(self, source: PathLike, **options: Any) -> pd.DataFrame:
        path = self.resolve(source)
        # The stdlib parser is used on purpose: it does not resolve external
        # entities, which closes the XXE hole that lxml leaves open by default.
        options.setdefault("parser", "etree")
        try:
            return pd.read_xml(path, **options)
        except (ValueError, SyntaxError) as exc:
            raise ExtractionError(f"Could not parse the XML file: {exc}", path=str(path)) from exc


@register_reader
class YAMLReader(Reader):
    format: ClassVar[FileFormat] = FileFormat.YAML
    extensions: ClassVar[tuple[str, ...]] = (".yaml", ".yml")

    def read(self, source: PathLike, **options: Any) -> pd.DataFrame:
        path = self.resolve(source)
        encoding = options.pop("encoding", None) or detect_encoding(path)
        record_path = options.pop("record_path", None)
        try:
            payload = yaml.safe_load(path.read_text(encoding=encoding))
        except yaml.YAMLError as exc:
            raise ExtractionError(f"Invalid YAML: {exc}", path=str(path)) from exc
        if payload is None:
            return pd.DataFrame()
        return _records_to_frame(payload, source=str(path), record_path=record_path)


@register_reader
class ParquetReader(Reader):
    format: ClassVar[FileFormat] = FileFormat.PARQUET
    extensions: ClassVar[tuple[str, ...]] = (".parquet", ".pq")
    supports_chunking: ClassVar[bool] = True
    binary: ClassVar[bool] = True

    def read(self, source: PathLike, **options: Any) -> pd.DataFrame:
        path = self.resolve(source)
        try:
            return pd.read_parquet(path, **options)
        except Exception as exc:  # pyarrow raises a wide range of errors
            raise ExtractionError(f"Could not read the parquet file: {exc}", path=str(path)) from exc

    def read_chunks(
        self, source: PathLike, chunk_size: int, **options: Any
    ) -> Iterator[pd.DataFrame]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        import pyarrow.parquet as pq

        path = self.resolve(source)
        parquet_file = pq.ParquetFile(path)
        columns = options.get("columns")
        for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
            yield batch.to_pandas()


@register_reader
class FeatherReader(Reader):
    format: ClassVar[FileFormat] = FileFormat.FEATHER
    extensions: ClassVar[tuple[str, ...]] = (".feather",)
    binary: ClassVar[bool] = True

    def read(self, source: PathLike, **options: Any) -> pd.DataFrame:
        # Read through the Arrow IPC API: pandas.read_feather goes through
        # pyarrow.feather.read_table, which is deprecated as of pyarrow 24.
        import pyarrow as pa

        path = self.resolve(source)
        columns = options.pop("columns", None)
        try:
            with path.open("rb") as handle:
                table = pa.ipc.open_file(handle).read_all()
        except Exception as exc:
            raise ExtractionError(f"Could not read the feather file: {exc}", path=str(path)) from exc
        if columns:
            table = table.select(list(columns))
        return table.to_pandas()


@register_reader
class PickleReader(Reader):
    """Pickle support, disabled unless the caller opts in.

    Unpickling executes arbitrary code, so a pickle file is equivalent to a
    script.  Reading one therefore requires ``allow_pickle=True`` (or the
    ``UD_ALLOW_PICKLE=1`` environment variable) to make the risk explicit.
    """

    format: ClassVar[FileFormat] = FileFormat.PICKLE
    extensions: ClassVar[tuple[str, ...]] = (".pkl", ".pickle")
    binary: ClassVar[bool] = True

    def read(self, source: PathLike, **options: Any) -> pd.DataFrame:
        allow = options.pop("allow_pickle", None)
        if allow is None:
            allow = os.environ.get("UD_ALLOW_PICKLE", "").lower() in ("1", "true", "yes")
        if not allow:
            raise SecurityError(
                "Reading pickle files executes arbitrary code; pass allow_pickle=True "
                "(or set UD_ALLOW_PICKLE=1) if the file is trusted",
                path=str(source),
            )
        path = self.resolve(source)
        # Reaching this line requires allow_pickle=True or UD_ALLOW_PICKLE=1.
        payload = pd.read_pickle(path, **options)  # noqa: S301  # nosec B301
        if isinstance(payload, pd.Series):
            return payload.to_frame()
        if not isinstance(payload, pd.DataFrame):
            raise ExtractionError(
                "Pickle does not contain a DataFrame", path=str(path), type=type(payload).__name__
            )
        return payload


@register_reader
class SQLiteReader(Reader):
    """Reads a table or a parameterised query from a SQLite file."""

    format: ClassVar[FileFormat] = FileFormat.SQLITE
    extensions: ClassVar[tuple[str, ...]] = (".db", ".sqlite", ".sqlite3")
    supports_chunking: ClassVar[bool] = True
    binary: ClassVar[bool] = True

    def tables(self, source: PathLike) -> list[str]:
        path = self.resolve(source)
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [row[0] for row in rows]

    def _statement(self, path: Path, options: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
        query = options.pop("query", None)
        params = tuple(options.pop("params", ()) or ())
        if query:
            return query, params
        table = options.pop("table", None)
        available = self.tables(path)
        if table is None:
            if not available:
                raise ExtractionError("SQLite file contains no tables", path=str(path))
            table = available[0]
        if not _IDENTIFIER.match(str(table)):
            raise SecurityError("Invalid table name", table=str(table))
        if table not in available:
            raise ExtractionError("Table not found", table=str(table), available=available)
        # The name matched _IDENTIFIER, was found in sqlite_master and is quoted;
        # SQLite has no bind parameter for identifiers, so this is the safe form.
        return f'SELECT * FROM "{table}"', ()  # noqa: S608  # nosec B608

    def read(self, source: PathLike, **options: Any) -> pd.DataFrame:
        path = self.resolve(source)
        statement, params = self._statement(path, options)
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            return pd.read_sql_query(statement, connection, params=params or None, **options)

    def read_chunks(
        self, source: PathLike, chunk_size: int, **options: Any
    ) -> Iterator[pd.DataFrame]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        path = self.resolve(source)
        statement, params = self._statement(path, options)
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            yield from pd.read_sql_query(
                statement, connection, params=params or None, chunksize=chunk_size, **options
            )
