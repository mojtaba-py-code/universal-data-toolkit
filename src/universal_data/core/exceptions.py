"""Exception hierarchy for the toolkit.

Every error carries an optional ``context`` mapping so that callers (and the CLI)
can show *why* something failed without having to parse the message string.
"""

from __future__ import annotations

from typing import Any


class DataToolkitError(Exception):
    """Base class for all errors raised by this package."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = {k: v for k, v in context.items() if v is not None}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
        return f"{self.message} [{details}]"


class ConfigurationError(DataToolkitError):
    """Raised when user supplied configuration is missing or malformed."""


class SecurityError(DataToolkitError):
    """Raised when an operation is refused for security reasons."""


class ExtractionError(DataToolkitError):
    """Raised when data cannot be read from a source."""


class UnsupportedFormatError(ExtractionError):
    """Raised when a file format is unknown or not supported."""


class DatabaseError(DataToolkitError):
    """Raised for connection, query or transaction failures."""


class AuthenticationError(DataToolkitError):
    """Raised when credentials are missing, rejected or malformed."""


class SchemaError(DataToolkitError):
    """Raised when a schema definition itself is invalid."""


class DataValidationError(DataToolkitError):
    """Raised when a dataset does not satisfy a schema in strict mode."""

    def __init__(self, message: str, errors: list[Any] | None = None, **context: Any) -> None:
        super().__init__(message, **context)
        self.errors = errors or []


class TransformationError(DataToolkitError):
    """Raised when a transformation cannot be applied."""


class RuleError(DataToolkitError):
    """Raised for invalid or unsafe business-rule expressions."""


class ExportError(DataToolkitError):
    """Raised when a dataset cannot be written to a destination."""


class PipelineError(DataToolkitError):
    """Raised when a pipeline step fails."""

    def __init__(self, message: str, step: str | None = None, **context: Any) -> None:
        super().__init__(message, step=step, **context)
        self.step = step


class PluginError(DataToolkitError):
    """Raised when a plugin cannot be loaded or registered."""


class DataProcessingError(DataToolkitError):
    """Generic processing failure that does not fit a more specific class."""
