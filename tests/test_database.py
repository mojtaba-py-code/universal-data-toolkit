"""Database ingestion and export, including SQL injection guards."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from universal_data.core.dataset import Dataset
from universal_data.core.exceptions import ConfigurationError, DatabaseError, SecurityError
from universal_data.export.file_writers import DatabaseExporter
from universal_data.ingestion.database import (
    DatabaseClient,
    DatabaseConfig,
    _sqlite_parameter_limit,
    validate_identifier,
)
from universal_data.security.secrets import SecretResolver, SecretStr


@pytest.fixture
def client(tmp_path: Path) -> DatabaseClient:
    config = DatabaseConfig.sqlite(str(tmp_path / "test.db"))
    client = DatabaseClient(config)
    frame = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["ali", "sara", "reza"], "amount": [10.0, 20.0, 30.0]}
    )
    client.write_frame(frame, "customers", if_exists="replace")
    return client


class TestDatabaseConfig:
    def test_sqlite_url(self, tmp_path: Path) -> None:
        config = DatabaseConfig.sqlite(str(tmp_path / "a.db"))
        assert config.url().drivername == "sqlite"

    def test_unsupported_driver(self) -> None:
        with pytest.raises(ConfigurationError, match="Unsupported database driver"):
            DatabaseConfig(driver="oracle", database="x")

    def test_host_required_for_server_drivers(self) -> None:
        with pytest.raises(ConfigurationError, match="host is required"):
            DatabaseConfig(driver="postgresql", database="app")

    def test_password_never_appears_in_repr_or_safe_url(self) -> None:
        config = DatabaseConfig(
            driver="postgresql",
            database="app",
            host="db.internal",
            username="app",
            password=SecretStr("hunter2"),
        )
        assert "hunter2" not in repr(config)
        assert "hunter2" not in config.safe_url()
        assert config.url().password == "hunter2"

    def test_from_dict_reads_password_from_environment(self) -> None:
        resolver = SecretResolver({"PG_PASSWORD": "s3cret", "PGHOST": "db.internal"})
        config = DatabaseConfig.from_dict(
            {
                "driver": "postgresql",
                "host": "${PGHOST}",
                "database": "app",
                "username": "app",
                "password_env": "PG_PASSWORD",
            },
            resolver,
        )
        assert config.host == "db.internal"
        assert config.password is not None
        assert config.password.get_secret_value() == "s3cret"
        assert config.port == 5432

    def test_from_dict_requires_database(self) -> None:
        with pytest.raises(ConfigurationError, match="Database name"):
            DatabaseConfig.from_dict({"driver": "sqlite"})

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_DRIVER", "postgresql")
        monkeypatch.setenv("APP_HOST", "localhost")
        monkeypatch.setenv("APP_NAME", "app")
        monkeypatch.setenv("APP_PASSWORD", "pw")
        config = DatabaseConfig.from_env("APP")
        assert config.driver == "postgresql"
        assert config.password is not None

    def test_from_env_requires_driver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NOPE_DRIVER", raising=False)
        with pytest.raises(ConfigurationError, match="driver is not configured"):
            DatabaseConfig.from_env("NOPE")


class TestIdentifierValidation:
    @pytest.mark.parametrize(
        "name",
        [
            "users; DROP TABLE users",
            "users--",
            "1users",
            "users table",
            "users'",
            "",
            "users/*comment*/",
        ],
    )
    def test_rejects_injection_attempts(self, name: str) -> None:
        with pytest.raises(SecurityError):
            validate_identifier(name, kind="table name")

    @pytest.mark.parametrize("name", ["users", "_private", "table_2", "col$x"])
    def test_accepts_plain_identifiers(self, name: str) -> None:
        assert validate_identifier(name) == name


class TestDatabaseClient:
    def test_ping_and_list_tables(self, client: DatabaseClient) -> None:
        assert client.ping() is True
        assert "customers" in client.list_tables()

    def test_describe_table(self, client: DatabaseClient) -> None:
        columns = {column["name"] for column in client.describe_table("customers")}
        assert columns == {"id", "name", "amount"}

    def test_read_table(self, client: DatabaseClient) -> None:
        frame = client.read_table("customers")
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == 3

    def test_read_table_with_columns_and_limit(self, client: DatabaseClient) -> None:
        frame = client.read_table("customers", columns=["id", "name"], limit=2)
        assert list(frame.columns) == ["id", "name"]
        assert len(frame) == 2

    def test_read_table_rejects_bad_column(self, client: DatabaseClient) -> None:
        with pytest.raises(SecurityError):
            client.read_table("customers", columns=["id; DROP TABLE customers"])

    def test_read_table_rejects_bad_limit(self, client: DatabaseClient) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            client.read_table("customers", limit=-1)

    def test_read_query_uses_bound_parameters(self, client: DatabaseClient) -> None:
        frame = client.read_query(
            "SELECT * FROM customers WHERE name = :name", params={"name": "sara"}
        )
        assert len(frame) == 1

    def test_injection_through_parameter_is_harmless(self, client: DatabaseClient) -> None:
        frame = client.read_query(
            "SELECT * FROM customers WHERE name = :name",
            params={"name": "sara'; DROP TABLE customers; --"},
        )
        assert len(frame) == 0
        assert "customers" in client.list_tables()

    def test_read_query_rejects_write_statements(self, client: DatabaseClient) -> None:
        with pytest.raises(SecurityError, match="Only SELECT"):
            client.read_query("DELETE FROM customers")

    def test_read_query_rejects_multiple_statements(self, client: DatabaseClient) -> None:
        with pytest.raises(SecurityError, match="Multiple SQL statements"):
            client.read_query("SELECT 1; DROP TABLE customers")

    def test_read_query_accepts_cte(self, client: DatabaseClient) -> None:
        frame = client.read_query("WITH x AS (SELECT * FROM customers) SELECT * FROM x")
        assert len(frame) == 3

    @pytest.mark.parametrize(
        "statement",
        [
            "WITH x AS (SELECT 1) DELETE FROM customers",
            "WITH x AS (SELECT 1) UPDATE customers SET name = 'x'",
            "WITH x AS (SELECT 1) INSERT INTO customers SELECT 9, 'z', 1.0",
            "SELECT * INTO copied FROM customers",
            "SELECT * FROM customers /* then */ ; DROP TABLE customers",
            "WITH x AS (SELECT 1) DROP TABLE customers",
        ],
    )
    def test_read_query_blocks_writes_behind_a_cte(
        self, client: DatabaseClient, statement: str
    ) -> None:
        """`WITH ... DELETE` is valid SQL; starting with WITH must not be enough."""
        with pytest.raises(SecurityError):
            client.read_query(statement)
        assert len(client.read_query("SELECT * FROM customers")) == 3

    def test_read_query_allows_a_semicolon_inside_a_literal(
        self, client: DatabaseClient
    ) -> None:
        client.write_frame(
            pd.DataFrame({"id": [1], "name": ["a;b"], "amount": [1.0]}),
            "quoted",
            if_exists="replace",
        )
        frame = client.read_query("SELECT * FROM quoted WHERE name = 'a;b'")
        assert len(frame) == 1

    def test_read_query_allows_keywords_inside_identifiers_and_literals(
        self, client: DatabaseClient
    ) -> None:
        frame = client.read_query("SELECT name AS updated_name FROM customers WHERE name != 'delete'")
        assert len(frame) == 3

    def test_read_query_still_allows_unions_and_aggregates(self, client: DatabaseClient) -> None:
        frame = client.read_query(
            "SELECT COUNT(*) AS n FROM customers UNION ALL SELECT 1"
        )
        assert len(frame) == 2

    def test_execute_allows_a_semicolon_inside_a_literal(self, client: DatabaseClient) -> None:
        client.execute("UPDATE customers SET name = 'a;b' WHERE id = 1")
        assert client.read_query("SELECT name FROM customers WHERE id = 1").iloc[0]["name"] == "a;b"

    def test_chunked_query(self, client: DatabaseClient) -> None:
        chunks = list(client.read_query("SELECT * FROM customers", chunk_size=2))
        assert [len(chunk) for chunk in chunks] == [2, 1]

    def test_execute_rejects_multiple_statements(self, client: DatabaseClient) -> None:
        with pytest.raises(SecurityError):
            client.execute("UPDATE customers SET name='x'; DROP TABLE customers")

    def test_execute_updates_rows(self, client: DatabaseClient) -> None:
        client.execute("UPDATE customers SET amount = :value WHERE id = :id", {"value": 99, "id": 1})
        frame = client.read_query("SELECT amount FROM customers WHERE id = 1")
        assert frame.iloc[0]["amount"] == 99

    def test_write_frame_rejects_bad_mode(self, client: DatabaseClient) -> None:
        with pytest.raises(ValueError, match="if_exists"):
            client.write_frame(pd.DataFrame({"a": [1]}), "t", if_exists="upsert")

    def test_write_frame_rejects_bad_table_name(self, client: DatabaseClient) -> None:
        with pytest.raises(SecurityError):
            client.write_frame(pd.DataFrame({"a": [1]}), "t; DROP TABLE customers")

    def test_write_frame_handles_a_wide_frame(self, client: DatabaseClient) -> None:
        """method="multi" sends one placeholder per cell; the chunk must respect
        the driver's bind-parameter limit or a wide frame fails to insert."""
        wide = pd.DataFrame({f"c{i}": range(5_000) for i in range(13)})
        assert client.write_frame(wide, "wide", if_exists="replace") == 5_000
        assert len(client.read_table("wide")) == 5_000

    def test_rows_per_insert_is_capped_by_the_column_count(self, client: DatabaseClient) -> None:
        # The ceiling is whatever this SQLite build reports: 32,766 on many
        # builds, 250,000 on the Ubuntu CI runners. Ask for more rows than fit.
        limit = _sqlite_parameter_limit()
        wide = pd.DataFrame({f"c{i}": [1] for i in range(50)})
        rows = client._rows_per_insert(wide, limit, index=False)
        assert rows * 50 <= limit
        assert rows == limit // 50
        narrow = pd.DataFrame({"a": [1]})
        assert client._rows_per_insert(narrow, 100, index=False) == 100

    def test_write_existing_table_fails_by_default(self, client: DatabaseClient) -> None:
        with pytest.raises(DatabaseError):
            client.write_frame(pd.DataFrame({"a": [1]}), "customers", if_exists="fail")

    def test_reflect(self, client: DatabaseClient) -> None:
        assert "customers" in client.reflect().tables

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        with DatabaseClient(DatabaseConfig.sqlite(str(tmp_path / "ctx.db"))) as client:
            client.ping()
        assert client._engine is None


class TestDatasetIntegration:
    def test_dataset_from_database_table(self, client: DatabaseClient) -> None:
        dataset = Dataset.from_database(client, table="customers")
        assert dataset.n_rows == 3
        assert "sqlite" in dataset.source

    def test_dataset_from_database_query(self, client: DatabaseClient) -> None:
        dataset = Dataset.from_database(
            client, query="SELECT * FROM customers WHERE amount > :low", params={"low": 15}
        )
        assert dataset.n_rows == 2

    def test_dataset_requires_exactly_one_source(self, client: DatabaseClient) -> None:
        with pytest.raises(ConfigurationError, match="exactly one"):
            Dataset.from_database(client, table="customers", query="SELECT 1")

    def test_dataset_to_database(self, client: DatabaseClient) -> None:
        dataset = Dataset(pd.DataFrame({"id": [9], "name": ["new"], "amount": [1.0]}))
        assert dataset.to_database(client, "customers", if_exists="append") == 1
        assert len(client.read_table("customers")) == 4

    def test_database_exporter_chunks(self, client: DatabaseClient) -> None:
        exporter = DatabaseExporter(client)
        chunks = [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3]})]
        assert exporter.write_chunks(chunks, "chunked", if_exists="replace") == 3
        assert len(client.read_table("chunked")) == 3
