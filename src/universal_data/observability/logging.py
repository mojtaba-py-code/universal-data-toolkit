"""Structured logging.

Log records carry the execution context (execution id, pipeline, step) through a
:mod:`contextvars` variable, so library code can simply call ``logger.info(...)``
without threading identifiers through every function signature.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from universal_data.core.types import PathLike
from universal_data.security.redaction import RedactingFilter

_EMPTY_CONTEXT: Mapping[str, Any] = MappingProxyType({})
_CONTEXT: ContextVar[Mapping[str, Any]] = ContextVar("ud_log_context", default=_EMPTY_CONTEXT)

_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


def new_execution_id() -> str:
    """Short, sortable-enough identifier used for one pipeline run."""
    return uuid.uuid4().hex[:12]


@contextmanager
def log_context(**fields: Any) -> Iterator[dict[str, Any]]:
    """Bind extra fields onto every log record emitted inside the block."""
    current = dict(_CONTEXT.get())
    current.update({k: v for k, v in fields.items() if v is not None})
    token = _CONTEXT.set(current)
    try:
        yield current
    finally:
        _CONTEXT.reset(token)


def current_context() -> dict[str, Any]:
    return dict(_CONTEXT.get())


class ContextFilter(logging.Filter):
    """Copies the contextvar payload onto the record."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _CONTEXT.get().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line - suitable for shipping to a log collector."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = _json_safe(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human readable output with the execution id appended when present."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s - %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        execution_id = getattr(record, "execution_id", None)
        step = getattr(record, "step", None)
        parts = [
            f"[{execution_id}]" if execution_id else "",
            f"({step})" if step else "",
        ]
        return f"{base} {' '.join(part for part in parts if part)}".rstrip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def configure_logging(
    level: str | int = "INFO",
    *,
    json_output: bool = False,
    log_file: PathLike | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    quiet: bool = False,
) -> logging.Logger:
    """Configure the ``universal_data`` logger tree.

    Called once by the CLI and by :class:`~universal_data.pipeline.pipeline.Pipeline`
    when it is asked to log to a file.  Importing the library never configures
    logging on its own - that decision belongs to the application.
    """
    logger = logging.getLogger("universal_data")
    logger.setLevel(level if isinstance(level, int) else level.upper())
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    filters: list[logging.Filter] = [ContextFilter(), RedactingFilter()]

    if not quiet:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(JsonFormatter() if json_output else ConsoleFormatter())
        for flt in filters:
            console.addFilter(flt)
        logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        rotating.setFormatter(JsonFormatter())
        for flt in (ContextFilter(), RedactingFilter()):
            rotating.addFilter(flt)
        logger.addHandler(rotating)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the package logger."""
    if name.startswith("universal_data"):
        return logging.getLogger(name)
    return logging.getLogger(f"universal_data.{name}")
