"""Format detection and file readers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from universal_data.core.exceptions import ExtractionError, SecurityError, UnsupportedFormatError
from universal_data.core.types import FileFormat
from universal_data.export.file_writers import FeatherWriter
from universal_data.ingestion.base import Reader, ReaderRegistry, readers
from universal_data.ingestion.detect import (
    detect_delimiter,
    detect_encoding,
    detect_format,
    human_size,
    inspect_file,
)
from universal_data.ingestion.files import SQLiteReader


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [1.5, 2.5, 3.5]})


class TestDetection:
    def test_detects_csv(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        path = tmp_path / "data.csv"
        frame.to_csv(path, index=False)
        assert detect_format(path) is FileFormat.CSV

    def test_detects_parquet_by_magic_bytes(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        path = tmp_path / "misnamed.csv"
        frame.to_parquet(path, index=False)
        assert detect_format(path) is FileFormat.PARQUET

    def test_detects_sqlite_by_magic_bytes(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        import sqlite3

        path = tmp_path / "store.bin"
        with sqlite3.connect(path) as connection:
            frame.to_sql("items", connection, index=False)
        assert detect_format(path) is FileFormat.SQLITE

    def test_detects_json_and_jsonl(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        json_path = tmp_path / "records.json"
        json_path.write_text(frame.to_json(orient="records"), encoding="utf-8")
        jsonl_path = tmp_path / "records.jsonl"
        jsonl_path.write_text(
            "\n".join(frame.to_json(orient="records", lines=True).splitlines()), encoding="utf-8"
        )
        assert detect_format(json_path) is FileFormat.JSON
        assert detect_format(jsonl_path) is FileFormat.JSONL

    def test_detects_xml_content_in_csv_file(self, tmp_path: Path) -> None:
        path = tmp_path / "weird.csv"
        path.write_text("<?xml version='1.0'?><rows><row><a>1</a></row></rows>", encoding="utf-8")
        assert detect_format(path) is FileFormat.XML

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "mystery.bin"
        path.write_bytes(b"\x99\x98\x97 not a known format")
        with pytest.raises(UnsupportedFormatError):
            detect_format(path)

    @pytest.mark.parametrize(
        ("content", "encoding"),
        [
            (b"id,name\n1,Ali\n", "utf-8"),
            ("id,name\n1,Ali\n".encode("utf-8-sig"), "utf-8-sig"),
            ("id,name\n1,Alí\n".encode("cp1252"), "cp1252"),
        ],
    )
    def test_encoding_detection(self, tmp_path: Path, content: bytes, encoding: str) -> None:
        path = tmp_path / "encoded.csv"
        path.write_bytes(content)
        assert detect_encoding(path) == encoding

    @pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
    def test_delimiter_detection(self, tmp_path: Path, delimiter: str) -> None:
        path = tmp_path / "delim.csv"
        path.write_text(
            f"id{delimiter}name{delimiter}city\n1{delimiter}ali{delimiter}tehran\n"
            f"2{delimiter}sara{delimiter}berlin\n",
            encoding="utf-8",
        )
        assert detect_delimiter(path, "utf-8") == delimiter

    def test_empty_file_delimiter_defaults_to_comma(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        assert detect_delimiter(path, "utf-8") == ","

    def test_inspect_file(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        path = tmp_path / "data.csv"
        frame.to_csv(path, index=False, sep=";")
        signature = inspect_file(path)
        assert signature.format is FileFormat.CSV
        assert signature.delimiter == ";"
        assert signature.size_bytes > 0
        assert "size_human" in signature.to_dict()

    @pytest.mark.parametrize(
        ("size", "expected"), [(512, "512 B"), (2048, "2.0 KB"), (5 * 1024**2, "5.0 MB")]
    )
    def test_human_size(self, size: int, expected: str) -> None:
        assert human_size(size) == expected


class TestFileReaders:
    @pytest.mark.parametrize(
        ("suffix", "writer"),
        [
            (".csv", lambda f, p: f.to_csv(p, index=False)),
            (".tsv", lambda f, p: f.to_csv(p, index=False, sep="\t")),
            (".json", lambda f, p: f.to_json(p, orient="records")),
            (".jsonl", lambda f, p: f.to_json(p, orient="records", lines=True)),
            (".parquet", lambda f, p: f.to_parquet(p, index=False)),
            (".feather", lambda f, p: FeatherWriter().write(f, p)),
            (".xlsx", lambda f, p: f.to_excel(p, index=False)),
        ],
    )
    def test_round_trip(self, tmp_path: Path, frame: pd.DataFrame, suffix: str, writer: object) -> None:
        path = tmp_path / f"data{suffix}"
        writer(frame, path)  # type: ignore[operator]
        loaded = readers.create(detect_format(path)).read(path)
        assert list(loaded.columns) == list(frame.columns)
        assert len(loaded) == len(frame)

    def test_csv_detects_semicolon_and_encoding(self, tmp_path: Path) -> None:
        path = tmp_path / "euro.csv"
        path.write_bytes("id;name\n1;Alí\n2;Bob\n".encode("cp1252"))
        loaded = readers.create(FileFormat.CSV).read(path)
        assert list(loaded.columns) == ["id", "name"]
        assert loaded.loc[0, "name"] == "Alí"

    def test_ragged_rows_fail_loudly_with_a_hint(self, tmp_path: Path) -> None:
        path = tmp_path / "ragged.csv"
        path.write_text("a,b,c\n1,2,3\n4,5\n6,7,8,9\n", encoding="utf-8")
        with pytest.raises(ExtractionError) as info:
            readers.create(FileFormat.CSV).read(path)
        assert "on_bad_lines" in str(info.value)

    def test_ragged_rows_can_be_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "ragged.csv"
        path.write_text("a,b,c\n1,2,3\n4,5\n6,7,8,9\n", encoding="utf-8")
        frame = readers.create(FileFormat.CSV).read(path, on_bad_lines="skip")
        assert len(frame) == 2

    def test_embedded_newlines_and_quoted_delimiters(self, tmp_path: Path) -> None:
        path = tmp_path / "quoted.csv"
        path.write_text('id;name\n1;"Smith; John"\n2;"two\nlines"\n', encoding="utf-8")
        frame = readers.create(FileFormat.CSV).read(path)
        assert frame.shape == (2, 2)
        assert frame.loc[0, "name"] == "Smith; John"

    def test_empty_csv_returns_empty_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        assert readers.create(FileFormat.CSV).read(path).empty

    def test_broken_json_reports_position(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text('{"a": 1,}', encoding="utf-8")
        with pytest.raises(ExtractionError, match="Invalid JSON"):
            readers.create(FileFormat.JSON).read(path)

    def test_broken_jsonl_reports_line(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.jsonl"
        path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
        with pytest.raises(ExtractionError, match="line 2"):
            readers.create(FileFormat.JSONL).read(path)

    def test_json_with_wrapper_key(self, tmp_path: Path) -> None:
        path = tmp_path / "wrapped.json"
        path.write_text('{"data": [{"a": 1}, {"a": 2}], "total": 2}', encoding="utf-8")
        loaded = readers.create(FileFormat.JSON).read(path)
        assert list(loaded["a"]) == [1, 2]

    def test_json_record_path(self, tmp_path: Path) -> None:
        path = tmp_path / "nested.json"
        path.write_text('{"payload": {"rows": [{"a": 1}]}}', encoding="utf-8")
        loaded = readers.create(FileFormat.JSON).read(path, record_path="payload.rows")
        assert loaded.to_dict("records") == [{"a": 1}]

    def test_json_bad_record_path(self, tmp_path: Path) -> None:
        path = tmp_path / "nested.json"
        path.write_text('{"payload": {"rows": []}}', encoding="utf-8")
        with pytest.raises(ExtractionError, match="record_path"):
            readers.create(FileFormat.JSON).read(path, record_path="nope.rows")

    def test_yaml_reader(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.yaml"
        path.write_text("- id: 1\n  name: a\n- id: 2\n  name: b\n", encoding="utf-8")
        loaded = readers.create(FileFormat.YAML).read(path)
        assert len(loaded) == 2

    def test_yaml_empty_document(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert readers.create(FileFormat.YAML).read(path).empty

    def test_xml_reader(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.xml"
        path.write_text(
            "<rows><row><id>1</id><name>a</name></row>"
            "<row><id>2</id><name>b</name></row></rows>",
            encoding="utf-8",
        )
        loaded = readers.create(FileFormat.XML).read(path)
        assert list(loaded.columns) == ["id", "name"]

    def test_xml_invalid_document(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.xml"
        path.write_text("<rows><row>", encoding="utf-8")
        with pytest.raises(ExtractionError):
            readers.create(FileFormat.XML).read(path)

    def test_pickle_requires_explicit_opt_in(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        path = tmp_path / "data.pkl"
        frame.to_pickle(path)
        reader = readers.create(FileFormat.PICKLE)
        with pytest.raises(SecurityError, match="arbitrary code"):
            reader.read(path)
        assert len(reader.read(path, allow_pickle=True)) == 3

    def test_pickle_env_opt_in(self, tmp_path: Path, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "data.pkl"
        frame.to_pickle(path)
        monkeypatch.setenv("UD_ALLOW_PICKLE", "1")
        assert len(readers.create(FileFormat.PICKLE).read(path)) == 3


class TestSQLiteReader:
    @pytest.fixture
    def db(self, tmp_path: Path, frame: pd.DataFrame) -> Path:
        import sqlite3

        path = tmp_path / "store.db"
        with sqlite3.connect(path) as connection:
            frame.to_sql("items", connection, index=False)
            frame.to_sql("other", connection, index=False)
        return path

    def test_lists_tables(self, db: Path) -> None:
        assert SQLiteReader().tables(db) == ["items", "other"]

    def test_reads_named_table(self, db: Path) -> None:
        assert len(SQLiteReader().read(db, table="items")) == 3

    def test_defaults_to_first_table(self, db: Path) -> None:
        assert len(SQLiteReader().read(db)) == 3

    def test_rejects_injected_table_name(self, db: Path) -> None:
        with pytest.raises(SecurityError, match="Invalid table name"):
            SQLiteReader().read(db, table="items; DROP TABLE items")

    def test_reports_unknown_table(self, db: Path) -> None:
        with pytest.raises(ExtractionError, match="Table not found"):
            SQLiteReader().read(db, table="missing")

    def test_chunked_read(self, db: Path) -> None:
        chunks = list(SQLiteReader().read_chunks(db, 2, table="items"))
        assert [len(chunk) for chunk in chunks] == [2, 1]


class TestChunking:
    def test_csv_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "big.csv"
        pd.DataFrame({"a": range(25)}).to_csv(path, index=False)
        chunks = list(readers.create(FileFormat.CSV).read_chunks(path, 10))
        assert [len(chunk) for chunk in chunks] == [10, 10, 5]

    def test_jsonl_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "big.jsonl"
        pd.DataFrame({"a": range(7)}).to_json(path, orient="records", lines=True)
        chunks = list(readers.create(FileFormat.JSONL).read_chunks(path, 3))
        assert [len(chunk) for chunk in chunks] == [3, 3, 1]

    def test_parquet_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "big.parquet"
        pd.DataFrame({"a": range(9)}).to_parquet(path, index=False)
        chunks = list(readers.create(FileFormat.PARQUET).read_chunks(path, 4))
        assert sum(len(chunk) for chunk in chunks) == 9

    def test_non_streaming_format_falls_back(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        path = tmp_path / "rows.xlsx"
        frame.to_excel(path, index=False)
        chunks = list(readers.create(FileFormat.EXCEL).read_chunks(path, 2))
        assert [len(chunk) for chunk in chunks] == [2, 1]

    def test_invalid_chunk_size(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        path = tmp_path / "rows.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="positive"):
            list(readers.create(FileFormat.CSV).read_chunks(path, 0))


class TestRegistry:
    def test_unknown_format_raises(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            readers.get("does-not-exist")

    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ReaderRegistry()

        class Dummy(Reader):
            format = FileFormat.CSV
            extensions = (".dummy",)

            def read(self, source: object, **options: object) -> pd.DataFrame:
                return pd.DataFrame()

        registry.register(Dummy)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(Dummy)
        registry.register(Dummy, replace=True)
        assert registry.for_extension(".dummy") is Dummy
        assert registry.formats() == ["csv"]
        assert ".dummy" in registry.extensions()
