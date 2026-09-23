"""File writers.

Writes go to a temporary file next to the destination and are renamed at the
end, so a crash halfway through never leaves a half-written dataset in place of
a good one.  Streaming writers (CSV, JSON Lines, Parquet) append chunk by chunk
and keep memory flat.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from universal_data.core.exceptions import ExportError, SecurityError
from universal_data.core.types import FileFormat, PathLike
from universal_data.export.base import Writer, register_writer
from universal_data.ingestion.files import _IDENTIFIER
from universal_data.observability.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def atomic_path(target: Path) -> Iterator[Path]:
    """Yield a temporary path and move it onto *target* on success."""
    temporary = target.with_name(f".{target.name}.tmp{os.getpid()}")
    try:
        yield temporary
        temporary.replace(target)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:  # never mask the original failure with a cleanup error
            logger.warning("Could not remove the temporary file %s", temporary)
        raise


@register_writer
class CSVWriter(Writer):
    format: ClassVar[FileFormat] = FileFormat.CSV
    extensions: ClassVar[tuple[str, ...]] = (".csv", ".txt")
    supports_streaming: ClassVar[bool] = True
    default_separator: ClassVar[str] = ","

    def _options(self, options: dict[str, Any]) -> dict[str, Any]:
        settings = dict(options)
        settings.setdefault("index", False)
        settings.setdefault("sep", self.default_separator)
        settings.setdefault("encoding", "utf-8")
        return settings

    def write(self, frame: pd.DataFrame, target: PathLike, **options: Any) -> int:
        path = self.resolve(target)
        settings = self._options(options)
        with atomic_path(path) as temporary:
            frame.to_csv(temporary, **settings)
        return len(frame)

    def write_chunks(
        self, chunks: Iterable[pd.DataFrame], target: PathLike, **options: Any
    ) -> int:
        path = self.resolve(target)
        settings = self._options(options)
        rows = 0
        with atomic_path(path) as temporary:
            first = True
            for chunk in chunks:
                chunk.to_csv(temporary, mode="w" if first else "a", header=first, **settings)
                rows += len(chunk)
                first = False
            if first:  # no chunks at all
                pd.DataFrame().to_csv(temporary, **settings)
        return rows


@register_writer
class TSVWriter(CSVWriter):
    format: ClassVar[FileFormat] = FileFormat.TSV
    extensions: ClassVar[tuple[str, ...]] = (".tsv", ".tab")
    default_separator: ClassVar[str] = "\t"


@register_writer
class ExcelWriter(Writer):
    format: ClassVar[FileFormat] = FileFormat.EXCEL
    extensions: ClassVar[tuple[str, ...]] = (".xlsx", ".xlsm")

    MAX_ROWS = 1_048_575  # Excel's limit, minus the header row

    def write(self, frame: pd.DataFrame, target: PathLike, **options: Any) -> int:
        path = self.resolve(target)
        if len(frame) > self.MAX_ROWS:
            raise ExportError(
                "Dataset exceeds the maximum number of rows a worksheet can hold; "
                "export to CSV or parquet instead",
                rows=len(frame),
                limit=self.MAX_ROWS,
            )
        options.setdefault("index", False)
        options.setdefault("sheet_name", "data")
        with atomic_path(path) as temporary:
            frame.to_excel(temporary, engine="openpyxl", **options)
        return len(frame)


@register_writer
class JSONWriter(Writer):
    format: ClassVar[FileFormat] = FileFormat.JSON
    extensions: ClassVar[tuple[str, ...]] = (".json",)

    def write(self, frame: pd.DataFrame, target: PathLike, **options: Any) -> int:
        path = self.resolve(target)
        indent = options.pop("indent", 2)
        orient = options.pop("orient", "records")
        with atomic_path(path) as temporary:
            frame.to_json(
                temporary,
                orient=orient,
                indent=indent,
                date_format="iso",
                force_ascii=False,
                **options,
            )
        return len(frame)


@register_writer
class JSONLinesWriter(Writer):
    format: ClassVar[FileFormat] = FileFormat.JSONL
    extensions: ClassVar[tuple[str, ...]] = (".jsonl", ".ndjson")
    supports_streaming: ClassVar[bool] = True

    def write(self, frame: pd.DataFrame, target: PathLike, **options: Any) -> int:
        return self.write_chunks([frame], target, **options)

    def write_chunks(
        self, chunks: Iterable[pd.DataFrame], target: PathLike, **options: Any
    ) -> int:
        path = self.resolve(target)
        rows = 0
        with atomic_path(path) as temporary, temporary.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                for record in chunk.to_dict(orient="records"):
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                rows += len(chunk)
        return rows


@register_writer
class ParquetWriter(Writer):
    format: ClassVar[FileFormat] = FileFormat.PARQUET
    extensions: ClassVar[tuple[str, ...]] = (".parquet", ".pq")
    supports_streaming: ClassVar[bool] = True

    def write(self, frame: pd.DataFrame, target: PathLike, **options: Any) -> int:
        path = self.resolve(target)
        options.setdefault("index", False)
        options.setdefault("compression", "snappy")
        with atomic_path(path) as temporary:
            frame.to_parquet(temporary, engine="pyarrow", **options)
        return len(frame)

    def write_chunks(
        self, chunks: Iterable[pd.DataFrame], target: PathLike, **options: Any
    ) -> int:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.resolve(target)
        compression = options.pop("compression", "snappy")
        rows = 0
        writer: pq.ParquetWriter | None = None
        with atomic_path(path) as temporary:
            try:
                for chunk in chunks:
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(temporary, table.schema, compression=compression)
                    elif not table.schema.equals(writer.schema):
                        table = _cast_to_schema(table, writer.schema)
                    writer.write_table(table)
                    rows += len(chunk)
                if writer is None:
                    pd.DataFrame().to_parquet(temporary, engine="pyarrow", index=False)
            finally:
                if writer is not None:
                    writer.close()
        return rows


def _cast_to_schema(table: Any, schema: Any) -> Any:
    """Align a chunk with the schema the parquet file was opened with.

    Chunked CSV reading infers dtypes per chunk, so a column that is numeric in
    the first million rows can arrive as text later.  Casting keeps the file
    consistent; when the cast is impossible the error names the columns instead
    of leaving a cryptic pyarrow message.
    """
    import pyarrow as pa

    try:
        return table.cast(schema)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, ValueError, KeyError) as exc:
        drifted = [
            field.name
            for field in table.schema
            if field.name not in schema.names or schema.field(field.name).type != field.type
        ]
        raise ExportError(
            "A chunk has a different schema than the first chunk and could not be cast; "
            "pin the column types when reading (dtype=...) or convert them in a cleaning step",
            columns=drifted,
            reason=str(exc),
        ) from exc


@register_writer
class FeatherWriter(Writer):
    format: ClassVar[FileFormat] = FileFormat.FEATHER
    extensions: ClassVar[tuple[str, ...]] = (".feather",)

    def write(self, frame: pd.DataFrame, target: PathLike, **options: Any) -> int:
        # Written through the Arrow IPC API directly: pyarrow.feather.write_feather
        # (which pandas.to_feather calls) is deprecated as of pyarrow 24.
        import pyarrow as pa

        path = self.resolve(target)
        table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
        # The sink is opened explicitly so it is definitely closed before the
        # rename; Windows refuses to replace a file that is still open.
        with (
            atomic_path(path) as temporary,
            temporary.open("wb") as sink,
            pa.ipc.new_file(sink, table.schema, **options) as writer,
        ):
            writer.write_table(table)
        return len(frame)


@register_writer
class SQLiteWriter(Writer):
    """Writes into a SQLite file; the table name is validated and quoted."""

    format: ClassVar[FileFormat] = FileFormat.SQLITE
    extensions: ClassVar[tuple[str, ...]] = (".db", ".sqlite", ".sqlite3")
    supports_streaming: ClassVar[bool] = True

    def write(self, frame: pd.DataFrame, target: PathLike, **options: Any) -> int:
        return self.write_chunks([frame], target, **options)

    def write_chunks(
        self, chunks: Iterable[pd.DataFrame], target: PathLike, **options: Any
    ) -> int:
        path = self.resolve(target)
        table = str(options.pop("table", "data"))
        if not _IDENTIFIER.match(table):
            raise SecurityError("Invalid table name", table=table)
        if_exists = options.pop("if_exists", "replace")
        if if_exists not in ("fail", "replace", "append"):
            raise ExportError("if_exists must be fail, replace or append", given=if_exists)
        rows = 0
        connection = sqlite3.connect(path)
        try:
            mode = if_exists
            for chunk in chunks:
                chunk.to_sql(table, connection, if_exists=mode, index=False, **options)
                mode = "append"
                rows += len(chunk)
            if rows == 0:
                # SQLite cannot create a table with no columns, so an empty
                # input leaves an empty database rather than an empty table.
                logger.warning("No rows to write; table '%s' was not created", table)
            connection.commit()
        except (sqlite3.Error, ValueError) as exc:
            connection.rollback()
            raise ExportError(f"Could not write to SQLite: {exc}", table=table) from exc
        finally:
            connection.close()
        return rows


class DatabaseExporter:
    """Writes a frame to PostgreSQL/MySQL/SQLite through :class:`DatabaseClient`.

    Kept out of the file-writer registry on purpose: a database target is
    configured, not a path, so it does not fit the same interface.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def write(
        self,
        frame: pd.DataFrame,
        table: str,
        *,
        schema: str | None = None,
        if_exists: str = "append",
        chunk_size: int = 10_000,
    ) -> int:
        return int(
            self.client.write_frame(
                frame, table, schema=schema, if_exists=if_exists, chunk_size=chunk_size
            )
        )

    def write_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        table: str,
        *,
        schema: str | None = None,
        if_exists: str = "replace",
        chunk_size: int = 10_000,
    ) -> int:
        rows = 0
        mode = if_exists
        for chunk in chunks:
            rows += self.write(
                chunk, table, schema=schema, if_exists=mode, chunk_size=chunk_size
            )
            mode = "append"
        return rows


def write_frame(
    frame: pd.DataFrame,
    target: PathLike,
    fmt: FileFormat | str | None = None,
    *,
    policy: Any = None,
    **options: Any,
) -> int:
    """Write *frame* to *target*, inferring the format from the extension."""
    from universal_data.export.base import writers

    writer_cls = writers.get(fmt) if fmt else writers.for_path(Path(target))
    return writer_cls(policy=policy).write(frame, target, **options)
