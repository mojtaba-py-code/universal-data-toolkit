"""Writers and the writer registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from universal_data.core.exceptions import ExportError, SecurityError, UnsupportedFormatError
from universal_data.core.types import FileFormat
from universal_data.export.base import Writer, WriterRegistry, writers
from universal_data.export.file_writers import (
    CSVWriter,
    ExcelWriter,
    JSONLinesWriter,
    ParquetWriter,
    SQLiteWriter,
    atomic_path,
    write_frame,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})


class TestAtomicWrites:
    def test_no_partial_file_is_left_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "out.csv"
        with pytest.raises(RuntimeError), atomic_path(target) as temporary:
            temporary.write_text("partial", encoding="utf-8")
            raise RuntimeError("boom")
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_existing_file_survives_a_failed_write(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        target = tmp_path / "out.csv"
        frame.to_csv(target, index=False)
        original = target.read_text(encoding="utf-8")
        with pytest.raises(RuntimeError), atomic_path(target) as temporary:
            temporary.write_text("garbage", encoding="utf-8")
            raise RuntimeError("boom")
        assert target.read_text(encoding="utf-8") == original


class TestWriters:
    @pytest.mark.parametrize(
        ("suffix", "fmt"),
        [
            (".csv", FileFormat.CSV),
            (".tsv", FileFormat.TSV),
            (".json", FileFormat.JSON),
            (".jsonl", FileFormat.JSONL),
            (".parquet", FileFormat.PARQUET),
            (".feather", FileFormat.FEATHER),
            (".xlsx", FileFormat.EXCEL),
        ],
    )
    def test_round_trip(self, tmp_path: Path, frame: pd.DataFrame, suffix: str, fmt: FileFormat) -> None:
        from universal_data.core.dataset import Dataset

        target = tmp_path / f"out{suffix}"
        assert writers.create(fmt).write(frame, target) == 3
        assert len(Dataset.read(target)) == 3

    def test_csv_uses_utf8_and_no_index(self, tmp_path: Path) -> None:
        target = tmp_path / "out.csv"
        CSVWriter().write(pd.DataFrame({"n": ["Alí"]}), target)
        assert target.read_text(encoding="utf-8").strip() == "n\nAlí"

    def test_csv_streaming_writes_one_header(self, tmp_path: Path) -> None:
        target = tmp_path / "out.csv"
        chunks = [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3]})]
        assert CSVWriter().write_chunks(chunks, target) == 3
        assert target.read_text(encoding="utf-8").count("a\n") == 1

    def test_csv_streaming_with_no_chunks(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.csv"
        assert CSVWriter().write_chunks([], target) == 0
        assert target.exists()

    def test_jsonl_streaming(self, tmp_path: Path) -> None:
        target = tmp_path / "out.jsonl"
        chunks = [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})]
        assert JSONLinesWriter().write_chunks(chunks, target) == 2
        assert len(target.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_parquet_streaming(self, tmp_path: Path) -> None:
        target = tmp_path / "out.parquet"
        chunks = [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3]})]
        assert ParquetWriter().write_chunks(chunks, target) == 3
        assert len(pd.read_parquet(target)) == 3

    def test_parquet_streaming_with_no_chunks(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.parquet"
        assert ParquetWriter().write_chunks([], target) == 0
        assert target.exists()

    def test_parquet_streaming_reports_schema_drift(self, tmp_path: Path) -> None:
        target = tmp_path / "drift.parquet"
        chunks = [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": ["x"]})]
        with pytest.raises(ExportError, match="different schema"):
            ParquetWriter().write_chunks(chunks, target)

    def test_parquet_streaming_casts_compatible_chunks(self, tmp_path: Path) -> None:
        target = tmp_path / "cast.parquet"
        chunks = [pd.DataFrame({"a": [1.0, 2.0]}), pd.DataFrame({"a": [3]})]
        assert ParquetWriter().write_chunks(chunks, target) == 3

    def test_excel_row_limit(self, tmp_path: Path) -> None:
        writer = ExcelWriter()
        big = pd.DataFrame({"a": range(3)})
        object.__setattr__(writer, "MAX_ROWS", 2)
        with pytest.raises(ExportError, match="maximum number of rows"):
            writer.write(big, tmp_path / "big.xlsx")

    def test_non_streaming_writer_concatenates(self, tmp_path: Path) -> None:
        target = tmp_path / "out.xlsx"
        chunks = [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})]
        assert ExcelWriter().write_chunks(chunks, target) == 2

    def test_non_streaming_writer_with_no_chunks(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.xlsx"
        assert ExcelWriter().write_chunks([], target) == 0


class TestSQLiteWriter:
    def test_writes_and_appends(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        target = tmp_path / "out.db"
        assert SQLiteWriter().write(frame, target, table="items") == 3
        assert SQLiteWriter().write(frame, target, table="items", if_exists="append") == 3
        from universal_data.ingestion.files import SQLiteReader

        assert len(SQLiteReader().read(target, table="items")) == 6

    def test_rejects_bad_table_name(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        with pytest.raises(SecurityError, match="Invalid table name"):
            SQLiteWriter().write(frame, tmp_path / "out.db", table="items; DROP TABLE items")

    def test_rejects_bad_mode(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        with pytest.raises(ExportError, match="if_exists"):
            SQLiteWriter().write(frame, tmp_path / "out.db", if_exists="upsert")

    def test_streaming(self, tmp_path: Path) -> None:
        target = tmp_path / "stream.db"
        chunks = [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})]
        assert SQLiteWriter().write_chunks(chunks, target, table="t") == 2

    def test_streaming_with_no_chunks(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.db"
        assert SQLiteWriter().write_chunks([], target, table="t") == 0


class TestRegistry:
    def test_format_inference_from_extension(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        target = tmp_path / "out.parquet"
        assert write_frame(frame, target) == 3

    def test_unknown_extension(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        with pytest.raises(UnsupportedFormatError, match="Cannot infer"):
            write_frame(frame, tmp_path / "out.unknown")

    def test_unknown_format(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            writers.get("avro")

    def test_duplicate_registration(self) -> None:
        registry = WriterRegistry()

        class Dummy(Writer):
            format = FileFormat.CSV
            extensions = (".dummy",)

            def write(self, frame: pd.DataFrame, target: object, **options: object) -> int:
                return 0

        registry.register(Dummy)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(Dummy)
        assert registry.formats() == ["csv"]
        assert ".dummy" in registry.extensions()

    def test_path_policy_is_enforced(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        from universal_data.security.paths import PathPolicy

        policy = PathPolicy.confined_to(tmp_path / "allowed")
        writer = CSVWriter(policy=policy)
        with pytest.raises(SecurityError):
            writer.write(frame, tmp_path / "outside.csv")
