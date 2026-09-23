"""Data ingestion: files, databases and HTTP APIs."""

from universal_data.ingestion import files as _files  # noqa: F401  (registers readers)
from universal_data.ingestion.api import (
    APIClient,
    APIConfig,
    ApiKeyAuth,
    AuthStrategy,
    BasicAuth,
    BearerTokenAuth,
    CursorPaginator,
    LinkHeaderPaginator,
    NoAuth,
    NoPagination,
    OAuth2ClientCredentials,
    OffsetPaginator,
    PageNumberPaginator,
    Paginator,
    RateLimiter,
    RetryPolicy,
)
from universal_data.ingestion.base import Reader, ReaderRegistry, readers, register_reader
from universal_data.ingestion.database import DatabaseClient, DatabaseConfig
from universal_data.ingestion.detect import (
    FileSignature,
    detect_delimiter,
    detect_encoding,
    detect_format,
    inspect_file,
)

__all__ = [
    "APIClient",
    "APIConfig",
    "ApiKeyAuth",
    "AuthStrategy",
    "BasicAuth",
    "BearerTokenAuth",
    "CursorPaginator",
    "DatabaseClient",
    "DatabaseConfig",
    "FileSignature",
    "LinkHeaderPaginator",
    "NoAuth",
    "NoPagination",
    "OAuth2ClientCredentials",
    "OffsetPaginator",
    "PageNumberPaginator",
    "Paginator",
    "RateLimiter",
    "Reader",
    "ReaderRegistry",
    "RetryPolicy",
    "detect_delimiter",
    "detect_encoding",
    "detect_format",
    "inspect_file",
    "readers",
    "register_reader",
]
