"""Writer interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from universal_data.core.exceptions import UnsupportedFormatError
from universal_data.core.types import FileFormat, PathLike
from universal_data.observability.logging import get_logger
from universal_data.security.paths import PathPolicy

logger = get_logger(__name__)


class Writer(ABC):
    """Writes a DataFrame to one destination format."""

    format: ClassVar[FileFormat]
    extensions: ClassVar[tuple[str, ...]] = ()
    supports_streaming: ClassVar[bool] = False

    def __init__(self, policy: PathPolicy | None = None) -> None:
        self.policy = policy or PathPolicy()

    def resolve(self, target: PathLike) -> Path:
        return self.policy.resolve_output(target)

    @abstractmethod
    def write(self, frame: pd.DataFrame, target: PathLike, **options: Any) -> int:
        """Write the frame and return the number of rows written."""

    def write_chunks(
        self, chunks: Iterable[pd.DataFrame], target: PathLike, **options: Any
    ) -> int:
        """Write an iterable of frames.

        Formats that cannot append are materialised in memory first; the warning
        makes that memory cost visible instead of silent.
        """
        logger.warning(
            "Format %s cannot stream; chunks are concatenated before writing", self.format
        )
        frames = list(chunks)
        if not frames:
            return self.write(pd.DataFrame(), target, **options)
        return self.write(pd.concat(frames, ignore_index=True), target, **options)


class WriterRegistry:
    def __init__(self) -> None:
        self._by_format: dict[FileFormat, type[Writer]] = {}
        self._by_extension: dict[str, type[Writer]] = {}

    def register(self, writer_cls: type[Writer], *, replace: bool = False) -> type[Writer]:
        fmt = writer_cls.format
        if fmt in self._by_format and not replace:
            raise ValueError(f"A writer for {fmt} is already registered")
        self._by_format[fmt] = writer_cls
        for extension in writer_cls.extensions:
            self._by_extension.setdefault(extension.lower(), writer_cls)
        return writer_cls

    def get(self, fmt: FileFormat | str) -> type[Writer]:
        try:
            key = FileFormat(fmt)
        except ValueError as exc:
            raise UnsupportedFormatError(
                "Unknown output format", format=str(fmt), supported=self.formats()
            ) from exc
        if key not in self._by_format:
            raise UnsupportedFormatError(
                "No writer is registered for this format",
                format=str(key),
                supported=self.formats(),
            )
        return self._by_format[key]

    def for_path(self, path: PathLike) -> type[Writer]:
        suffix = Path(path).suffix.lower()
        writer = self._by_extension.get(suffix)
        if writer is None:
            raise UnsupportedFormatError(
                "Cannot infer the output format from the file extension",
                suffix=suffix or "<none>",
                supported=sorted(self._by_extension),
            )
        return writer

    def create(self, fmt: FileFormat | str, policy: PathPolicy | None = None) -> Writer:
        return self.get(fmt)(policy=policy)

    def formats(self) -> list[str]:
        return sorted(str(fmt) for fmt in self._by_format)

    def extensions(self) -> list[str]:
        return sorted(self._by_extension)


writers = WriterRegistry()


def register_writer(writer_cls: type[Writer]) -> type[Writer]:
    return writers.register(writer_cls)
