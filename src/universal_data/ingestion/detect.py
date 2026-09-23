"""Format, encoding and delimiter detection.

Detection is deliberately conservative: the file signature wins over the
extension, and when nothing is certain the caller gets an explicit error instead
of a guess that silently corrupts data.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from universal_data.core.exceptions import UnsupportedFormatError
from universal_data.core.types import FileFormat, PathLike
from universal_data.security.paths import PathPolicy

SUFFIX_FORMATS: dict[str, FileFormat] = {
    ".csv": FileFormat.CSV,
    ".txt": FileFormat.CSV,
    ".tsv": FileFormat.TSV,
    ".tab": FileFormat.TSV,
    ".xlsx": FileFormat.EXCEL,
    ".xlsm": FileFormat.EXCEL,
    ".xls": FileFormat.EXCEL,
    ".json": FileFormat.JSON,
    ".jsonl": FileFormat.JSONL,
    ".ndjson": FileFormat.JSONL,
    ".xml": FileFormat.XML,
    ".yaml": FileFormat.YAML,
    ".yml": FileFormat.YAML,
    ".parquet": FileFormat.PARQUET,
    ".pq": FileFormat.PARQUET,
    ".feather": FileFormat.FEATHER,
    ".pkl": FileFormat.PICKLE,
    ".pickle": FileFormat.PICKLE,
    ".db": FileFormat.SQLITE,
    ".sqlite": FileFormat.SQLITE,
    ".sqlite3": FileFormat.SQLITE,
}

# Magic bytes are checked before the extension: a ".csv" that is really a parquet
# file should not be parsed as text.
_MAGIC: tuple[tuple[bytes, FileFormat], ...] = (
    (b"PAR1", FileFormat.PARQUET),
    (b"SQLite format 3\x00", FileFormat.SQLITE),
    (b"ARROW1", FileFormat.FEATHER),
    (b"PK\x03\x04", FileFormat.EXCEL),  # xlsx is a zip container
    (b"\xd0\xcf\x11\xe0", FileFormat.EXCEL),  # legacy xls (OLE2)
)

_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)

_TEXT_FALLBACKS = ("utf-8", "cp1252", "latin-1")
_SAMPLE_BYTES = 64 * 1024


@dataclass
class FileSignature:
    """What we could work out about a file before parsing it."""

    path: Path
    format: FileFormat
    size_bytes: int
    encoding: str | None = None
    delimiter: str | None = None
    detected_from: str = "extension"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format": str(self.format),
            "size_bytes": self.size_bytes,
            "size_human": human_size(self.size_bytes),
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "detected_from": self.detected_from,
        }


def human_size(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"  # pragma: no cover - unreachable


def detect_encoding(path: Path, *, sample_size: int = _SAMPLE_BYTES) -> str:
    """Best-effort text encoding detection.

    BOM first, then a strict UTF-8 decode, then the two encodings that account
    for nearly every remaining CSV in the wild.  ``latin-1`` never fails, so the
    function always returns something usable.
    """
    with path.open("rb") as handle:
        sample = handle.read(sample_size)
    for bom, encoding in _BOMS:
        if sample.startswith(bom):
            return encoding
    for encoding in _TEXT_FALLBACKS:
        try:
            sample.decode(encoding)
        except UnicodeDecodeError:
            continue
        else:
            return encoding
    return "latin-1"  # pragma: no cover - latin-1 decodes every byte string


def detect_delimiter(path: Path, encoding: str, *, sample_size: int = _SAMPLE_BYTES) -> str:
    """Sniff a CSV delimiter, falling back to the most frequent candidate."""
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        sample = handle.read(sample_size)
    if not sample.strip():
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {candidate: first_line.count(candidate) for candidate in ",;\t|"}
        best = max(counts, key=lambda key: counts[key])
        return best if counts[best] else ","


def _sniff_text_format(path: Path, encoding: str) -> FileFormat | None:
    """Distinguish JSON, JSON Lines, XML and YAML from delimited text."""
    with path.open("r", encoding=encoding, errors="replace") as handle:
        head = handle.read(4096).lstrip()
    if not head:
        return None
    if head.startswith("<?xml") or head.startswith("<"):
        return FileFormat.XML
    if head.startswith("{") or head.startswith("["):
        first_line = head.splitlines()[0]
        try:
            json.loads(first_line)
        except json.JSONDecodeError:
            return FileFormat.JSON
        # A complete JSON value on the first line means one object per line.
        return FileFormat.JSONL if head.lstrip().startswith("{") else FileFormat.JSON
    if head.startswith("---"):
        return FileFormat.YAML
    return None


def detect_format(path: PathLike, *, policy: PathPolicy | None = None) -> FileFormat:
    """Return the format of *path*, preferring content over extension."""
    resolved = (policy or PathPolicy()).resolve_input(path)
    with resolved.open("rb") as handle:
        header = handle.read(32)
    for magic, fmt in _MAGIC:
        if header.startswith(magic):
            return fmt
    if header[:2] in (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        return FileFormat.PICKLE

    suffix = resolved.suffix.lower()
    if suffix in SUFFIX_FORMATS:
        fmt = SUFFIX_FORMATS[suffix]
        if fmt in (FileFormat.CSV, FileFormat.TSV):
            sniffed = _sniff_text_format(resolved, detect_encoding(resolved))
            if sniffed is not None:
                return sniffed
        return fmt

    sniffed = _sniff_text_format(resolved, detect_encoding(resolved))
    if sniffed is not None:
        return sniffed
    raise UnsupportedFormatError(
        "Could not determine the file format",
        path=str(resolved),
        suffix=suffix or "<none>",
        supported=sorted({str(v) for v in SUFFIX_FORMATS.values()}),
    )


def inspect_file(path: PathLike, *, policy: PathPolicy | None = None) -> FileSignature:
    """Collect everything cheap we can learn about a file without parsing it."""
    resolved = (policy or PathPolicy()).resolve_input(path)
    fmt = detect_format(resolved, policy=policy)
    signature = FileSignature(
        path=resolved,
        format=fmt,
        size_bytes=resolved.stat().st_size,
        detected_from="content" if resolved.suffix.lower() not in SUFFIX_FORMATS else "extension",
    )
    if fmt in (FileFormat.CSV, FileFormat.TSV, FileFormat.JSON, FileFormat.JSONL,
               FileFormat.XML, FileFormat.YAML):
        signature.encoding = detect_encoding(resolved)
        if fmt in (FileFormat.CSV, FileFormat.TSV):
            signature.delimiter = detect_delimiter(resolved, signature.encoding)
    return signature
