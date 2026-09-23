"""Reader interface and registry.

Readers are looked up by :class:`~universal_data.core.types.FileFormat` through a
registry rather than an ``if`` chain, so a plugin can add a format without the
core package knowing about it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from universal_data.core.exceptions import UnsupportedFormatError
from universal_data.core.types import FileFormat, PathLike
from universal_data.observability.logging import get_logger
from universal_data.security.paths import PathPolicy

logger = get_logger(__name__)


class Reader(ABC):
    """Reads one file format into a DataFrame."""

    format: ClassVar[FileFormat]
    extensions: ClassVar[tuple[str, ...]] = ()
    supports_chunking: ClassVar[bool] = False
    binary: ClassVar[bool] = False

    def __init__(self, policy: PathPolicy | None = None) -> None:
        self.policy = policy or PathPolicy()

    def resolve(self, source: PathLike) -> Path:
        return self.policy.resolve_input(source)

    @abstractmethod
    def read(self, source: PathLike, **options: Any) -> pd.DataFrame:
        """Read the whole file into memory."""

    def read_chunks(
        self, source: PathLike, chunk_size: int, **options: Any
    ) -> Iterator[pd.DataFrame]:
        """Yield the file in chunks.

        Formats that cannot stream fall back to a single in-memory read and are
        sliced afterwards; that keeps the pipeline API uniform, but it does not
        save memory, so a warning is emitted.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        logger.warning(
            "Format %s cannot stream; the file is read fully before slicing", self.format
        )
        frame = self.read(source, **options)
        for start in range(0, len(frame), chunk_size):
            yield frame.iloc[start : start + chunk_size]


class ReaderRegistry:
    """Maps formats and file extensions to reader classes."""

    def __init__(self) -> None:
        self._by_format: dict[FileFormat, type[Reader]] = {}
        self._by_extension: dict[str, type[Reader]] = {}

    def register(self, reader_cls: type[Reader], *, replace: bool = False) -> type[Reader]:
        fmt = reader_cls.format
        if fmt in self._by_format and not replace:
            raise ValueError(f"A reader for {fmt} is already registered")
        self._by_format[fmt] = reader_cls
        for extension in reader_cls.extensions:
            self._by_extension.setdefault(extension.lower(), reader_cls)
        return reader_cls

    def get(self, fmt: FileFormat | str) -> type[Reader]:
        try:
            key = FileFormat(fmt)
        except ValueError as exc:
            raise UnsupportedFormatError(
                "Unknown input format", format=str(fmt), supported=self.formats()
            ) from exc
        if key not in self._by_format:
            raise UnsupportedFormatError(
                "No reader is registered for this format",
                format=str(key),
                supported=self.formats(),
            )
        return self._by_format[key]

    def for_extension(self, extension: str) -> type[Reader] | None:
        return self._by_extension.get(extension.lower())

    def create(self, fmt: FileFormat | str, policy: PathPolicy | None = None) -> Reader:
        return self.get(fmt)(policy=policy)

    def formats(self) -> list[str]:
        return sorted(str(fmt) for fmt in self._by_format)

    def extensions(self) -> list[str]:
        return sorted(self._by_extension)


readers = ReaderRegistry()


def register_reader(reader_cls: type[Reader]) -> type[Reader]:
    """Class decorator used by the built-in readers and by plugins."""
    return readers.register(reader_cls)
