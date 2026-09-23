"""Database ingestion and export through SQLAlchemy.

Credentials are never part of a connection string written in a config file:
:class:`DatabaseConfig` builds a :class:`sqlalchemy.engine.URL` from separate
fields, and the password comes from the environment as a
:class:`~universal_data.security.secrets.SecretStr`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine import URL, Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from universal_data.core.exceptions import ConfigurationError, DatabaseError, SecurityError
from universal_data.observability.logging import get_logger
from universal_data.security.secrets import SecretResolver, SecretStr

logger = get_logger(__name__)

DRIVERS: dict[str, str] = {
    "sqlite": "sqlite",
    "postgresql": "postgresql+psycopg",
    "postgres": "postgresql+psycopg",
    "mysql": "mysql+pymysql",
    "mariadb": "mysql+pymysql",
}

DEFAULT_PORTS = {"postgresql": 5432, "postgres": 5432, "mysql": 3306, "mariadb": 3306}

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_READ_ONLY_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_STATEMENT_SEPARATOR = re.compile(r";\s*\S")

# String literals and comments are removed before a statement is inspected, so
# that a semicolon or a keyword inside a value cannot be mistaken for syntax.
_SQL_LITERAL = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`")
_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)

# Starting with SELECT or WITH is not enough: `WITH x AS (...) DELETE FROM t` is
# a valid data-modifying statement in SQLite, PostgreSQL and MySQL, and it would
# otherwise sail straight through a "starts with WITH" check.
_WRITING_KEYWORD = re.compile(
    r"(?i)\b(insert|update|delete|drop|alter|create|truncate|replace|merge|grant|"
    r"revoke|attach|detach|pragma|vacuum|reindex|analyze|call|exec|execute|into|"
    r"outfile|dumpfile|copy)\b"
)

# Rows per INSERT are capped so a wide frame cannot exceed the driver's bind
# parameter limit; SQLite's default is the tightest of the three.
PARAMETER_LIMITS = {"sqlite": 999, "postgresql": 65_535, "postgres": 65_535,
                    "mysql": 65_535, "mariadb": 65_535}


def _strip_literals(statement: str) -> str:
    """Return *statement* with comments and string literals blanked out."""
    return _SQL_LITERAL.sub("''", _SQL_COMMENT.sub(" ", statement))


def _sqlite_parameter_limit() -> int:
    """Ask SQLite for its real bind-parameter limit, falling back to 999."""
    import sqlite3

    try:
        with sqlite3.connect(":memory:") as connection:
            return int(connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))
    except (AttributeError, sqlite3.Error):  # pragma: no cover - very old builds
        return 999


def validate_identifier(name: str, *, kind: str = "identifier") -> str:
    """Reject anything that is not a plain SQL identifier.

    Table and column names cannot be passed as bind parameters, so they are the
    one place where user input reaches the statement text.  They are validated
    here and quoted by SQLAlchemy's dialect preparer afterwards.
    """
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise SecurityError(f"Invalid {kind}", value=str(name))
    return name


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection settings for one database."""

    driver: str
    database: str
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: SecretStr | None = None
    schema: str | None = None
    connect_args: dict[str, Any] = field(default_factory=dict)
    pool_size: int = 5
    max_overflow: int = 5
    pool_timeout: int = 30
    pool_recycle: int = 1_800

    def __post_init__(self) -> None:
        if self.driver not in DRIVERS:
            raise ConfigurationError(
                "Unsupported database driver", driver=self.driver, supported=sorted(DRIVERS)
            )
        if self.driver != "sqlite" and not self.host:
            raise ConfigurationError("A host is required for this driver", driver=self.driver)

    @classmethod
    def sqlite(cls, path: str) -> DatabaseConfig:
        return cls(driver="sqlite", database=str(path))

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], resolver: SecretResolver | None = None
    ) -> DatabaseConfig:
        resolver = resolver or SecretResolver()
        settings = dict(data)
        driver = str(settings.pop("driver", settings.pop("type", ""))).lower()
        password_ref = settings.pop("password", None)
        password_env = settings.pop("password_env", None)
        password: SecretStr | None = None
        if password_env:
            password = resolver.get(str(password_env), required=True)
        elif password_ref:
            resolved = resolver.resolve(password_ref)
            password = SecretStr(str(resolved)) if resolved else None
        username = settings.pop("username", settings.pop("user", None))
        if username is not None:
            username = str(resolver.resolve(username))
        database = settings.pop("database", settings.pop("path", None))
        if not database:
            raise ConfigurationError("Database name (or sqlite path) is required")
        port = settings.pop("port", None)
        return cls(
            driver=driver,
            database=str(resolver.resolve(database)),
            host=str(resolver.resolve(settings.pop("host"))) if "host" in settings else None,
            port=int(port) if port else DEFAULT_PORTS.get(driver),
            username=username,
            password=password,
            schema=settings.pop("schema", None),
            connect_args=dict(settings.pop("connect_args", {}) or {}),
            pool_size=int(settings.pop("pool_size", 5)),
            max_overflow=int(settings.pop("max_overflow", 5)),
        )

    @classmethod
    def from_env(cls, prefix: str = "DB", resolver: SecretResolver | None = None) -> DatabaseConfig:
        """Build a config from ``<PREFIX>_DRIVER``, ``<PREFIX>_HOST`` ... variables."""
        resolver = resolver or SecretResolver()
        import os

        def value(name: str, default: str | None = None) -> str | None:
            return os.environ.get(f"{prefix}_{name}", default)

        driver = (value("DRIVER") or "").lower()
        if not driver:
            raise ConfigurationError("Database driver is not configured", variable=f"{prefix}_DRIVER")
        port = value("PORT")
        return cls(
            driver=driver,
            database=value("NAME") or value("DATABASE") or "",
            host=value("HOST"),
            port=int(port) if port else DEFAULT_PORTS.get(driver),
            username=value("USER"),
            password=resolver.get(f"{prefix}_PASSWORD"),
            schema=value("SCHEMA"),
        )

    def url(self) -> URL:
        if self.driver == "sqlite":
            return URL.create("sqlite", database=self.database)
        return URL.create(
            DRIVERS[self.driver],
            username=self.username,
            password=self.password.get_secret_value() if self.password else None,
            host=self.host,
            port=self.port,
            database=self.database,
        )

    def safe_url(self) -> str:
        """Connection string with the password masked - safe to log."""
        return self.url().render_as_string(hide_password=True)

    def __repr__(self) -> str:
        return f"DatabaseConfig({self.safe_url()})"


class DatabaseClient:
    """Thin, pooled SQLAlchemy wrapper with a read-only query path."""

    def __init__(self, config: DatabaseConfig, *, echo: bool = False) -> None:
        self.config = config
        self._engine: Engine | None = None
        self._echo = echo

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            kwargs: dict[str, Any] = {
                "echo": self._echo,
                "future": True,
                "connect_args": dict(self.config.connect_args),
            }
            if self.config.driver != "sqlite":
                kwargs.update(
                    pool_size=self.config.pool_size,
                    max_overflow=self.config.max_overflow,
                    pool_timeout=self.config.pool_timeout,
                    pool_recycle=self.config.pool_recycle,
                    pool_pre_ping=True,
                )
            try:
                self._engine = create_engine(self.config.url(), **kwargs)
            except SQLAlchemyError as exc:
                raise DatabaseError(
                    f"Could not create the engine: {exc}", url=self.config.safe_url()
                ) from exc
            logger.debug("Database engine created for %s", self.config.safe_url())
        return self._engine

    @contextmanager
    def connect(self) -> Iterator[Connection]:
        try:
            with self.engine.connect() as connection:
                yield connection
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Connection failed: {exc}", url=self.config.safe_url()) from exc

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """Run a block inside a transaction, rolling back on any exception."""
        try:
            with self.engine.begin() as connection:
                yield connection
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Transaction failed: {exc}", url=self.config.safe_url()) from exc

    def ping(self) -> bool:
        with self.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True

    def list_tables(self, schema: str | None = None) -> list[str]:
        try:
            return list(inspect(self.engine).get_table_names(schema=schema or self.config.schema))
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Could not list tables: {exc}") from exc

    def describe_table(self, table: str, schema: str | None = None) -> list[dict[str, Any]]:
        validate_identifier(table, kind="table name")
        try:
            columns = inspect(self.engine).get_columns(table, schema=schema or self.config.schema)
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Could not describe table: {exc}", table=table) from exc
        return [
            {
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": bool(column.get("nullable", True)),
                "default": column.get("default"),
            }
            for column in columns
        ]

    def _prepared_name(self, table: str, schema: str | None) -> str:
        preparer = self.engine.dialect.identifier_preparer
        validate_identifier(table, kind="table name")
        if schema:
            validate_identifier(schema, kind="schema name")
            return f"{preparer.quote(schema)}.{preparer.quote(table)}"
        return preparer.quote(table)

    def read_table(
        self,
        table: str,
        *,
        schema: str | None = None,
        columns: Sequence[str] | None = None,
        limit: int | None = None,
        chunk_size: int | None = None,
    ) -> pd.DataFrame | Iterator[pd.DataFrame]:
        schema = schema or self.config.schema
        qualified = self._prepared_name(table, schema)
        preparer = self.engine.dialect.identifier_preparer
        if columns:
            selected = ", ".join(
                preparer.quote(validate_identifier(column, kind="column name")) for column in columns
            )
        else:
            selected = "*"
        # Identifiers cannot be bound as parameters; they are validated against
        # _IDENTIFIER and quoted by the dialect preparer before reaching the text.
        statement = f"SELECT {selected} FROM {qualified}"  # noqa: S608  # nosec B608
        params: dict[str, Any] = {}
        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            statement += " LIMIT :_row_limit"
            params["_row_limit"] = limit
        return self.read_query(statement, params=params, chunk_size=chunk_size)

    def read_query(
        self,
        query: str,
        *,
        params: Mapping[str, Any] | None = None,
        chunk_size: int | None = None,
    ) -> pd.DataFrame | Iterator[pd.DataFrame]:
        """Run a read-only statement with bound parameters.

        The statement must start with ``SELECT`` or ``WITH``, must be a single
        statement, and must not contain a writing keyword anywhere - a
        ``WITH ... DELETE`` is valid SQL and would otherwise pass a check that
        only looked at the first word.

        This is a guard against a careless or hostile configuration file, not a
        sandbox: the authoritative control is the permissions of the database
        role the toolkit connects with, which should be read-only.
        """
        sanitized = _strip_literals(query)
        if not _READ_ONLY_START.match(sanitized):
            raise SecurityError("Only SELECT/WITH statements are allowed here")
        if _STATEMENT_SEPARATOR.search(sanitized):
            raise SecurityError("Multiple SQL statements are not allowed")
        writing = _WRITING_KEYWORD.search(sanitized)
        if writing:
            raise SecurityError(
                "Statement contains a writing keyword; use execute() for writes",
                keyword=writing.group(1).lower(),
            )
        statement = text(query)
        try:
            if chunk_size:
                return self._iter_query(statement, dict(params or {}), chunk_size)
            with self.connect() as connection:
                return pd.read_sql_query(statement, connection, params=dict(params or {}))
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Query failed: {exc}") from exc

    def _iter_query(
        self, statement: Any, params: dict[str, Any], chunk_size: int
    ) -> Iterator[pd.DataFrame]:
        with self.connect() as connection:
            yield from pd.read_sql_query(
                statement, connection, params=params, chunksize=chunk_size
            )

    def execute(self, statement: str, params: Mapping[str, Any] | None = None) -> int:
        """Run a write statement inside a transaction and return the row count."""
        if _STATEMENT_SEPARATOR.search(_strip_literals(statement)):
            raise SecurityError("Multiple SQL statements are not allowed")
        with self.transaction() as connection:
            result = connection.execute(text(statement), dict(params or {}))
            return result.rowcount if result.rowcount is not None else 0

    def write_frame(
        self,
        frame: pd.DataFrame,
        table: str,
        *,
        schema: str | None = None,
        if_exists: str = "fail",
        index: bool = False,
        chunk_size: int = 10_000,
    ) -> int:
        validate_identifier(table, kind="table name")
        if if_exists not in ("fail", "replace", "append"):
            raise ValueError("if_exists must be one of: fail, replace, append")
        rows_per_insert = self._rows_per_insert(frame, chunk_size, index=index)
        try:
            with self.transaction() as connection:
                frame.to_sql(
                    table,
                    connection,
                    schema=schema or self.config.schema,
                    if_exists=if_exists,
                    index=index,
                    chunksize=rows_per_insert,
                    method="multi",
                )
        except (SQLAlchemyError, ValueError) as exc:
            raise DatabaseError(f"Could not write to the table: {exc}", table=table) from exc
        return len(frame)

    def _rows_per_insert(self, frame: pd.DataFrame, requested: int, *, index: bool) -> int:
        """Rows per multi-row INSERT that stay under the driver's bind limit.

        ``method="multi"`` sends one placeholder per cell, so a wide frame with
        the default chunk size overflows the limit long before the data is big:
        13 columns x 10,000 rows is 130,000 parameters against SQLite's 32,766.
        """
        columns = max(len(frame.columns) + (1 if index else 0), 1)
        limit = (
            _sqlite_parameter_limit()
            if self.config.driver == "sqlite"
            else PARAMETER_LIMITS.get(self.config.driver, 999)
        )
        return max(1, min(requested, limit // columns))

    def reflect(self, schema: str | None = None) -> MetaData:
        metadata = MetaData()
        metadata.reflect(bind=self.engine, schema=schema or self.config.schema)
        return metadata

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def __enter__(self) -> DatabaseClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
