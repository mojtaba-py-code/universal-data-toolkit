"""PostgreSQL integration tests.

Skipped unless a server is reachable. Everything else in the suite exercises the
database layer through SQLite, which shares the SQLAlchemy code path but not the
dialect - these tests cover what SQLite cannot: a real schema, server-side
parameter limits, and the identifier quoting of another dialect.

    docker compose up -d postgres
    PG_PASSWORD=... UD_TEST_POSTGRES=1 pytest tests/test_postgres_integration.py
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from universal_data.core.dataset import Dataset
from universal_data.core.exceptions import SecurityError
from universal_data.ingestion.database import DatabaseClient, DatabaseConfig
from universal_data.security.secrets import SecretResolver

pytestmark = pytest.mark.integration

ENABLED = os.environ.get("UD_TEST_POSTGRES") == "1"


def build_config() -> DatabaseConfig:
    return DatabaseConfig.from_dict(
        {
            "driver": "postgresql",
            "host": os.environ.get("PGHOST", "localhost"),
            "port": os.environ.get("PGPORT", 5432),
            "database": os.environ.get("PGDATABASE", "datatoolkit"),
            "username": os.environ.get("PGUSER", "datatoolkit"),
            "password_env": "PG_PASSWORD",
        },
        SecretResolver(),
    )


@pytest.fixture(scope="module")
def client() -> DatabaseClient:
    if not ENABLED:
        pytest.skip("set UD_TEST_POSTGRES=1 and start PostgreSQL to run these tests")
    try:
        instance = DatabaseClient(build_config())
        instance.ping()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL is not reachable: {exc}")
    yield instance
    instance.close()


def test_round_trip(client: DatabaseClient) -> None:
    frame = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "amount": [1.0, 2.0, 3.0]})
    assert client.write_frame(frame, "ud_round_trip", if_exists="replace") == 3
    assert len(client.read_table("ud_round_trip")) == 3
    assert "ud_round_trip" in client.list_tables()


def test_wide_frame_respects_the_parameter_limit(client: DatabaseClient) -> None:
    wide = pd.DataFrame({f"c{i}": range(3_000) for i in range(30)})
    assert client.write_frame(wide, "ud_wide", if_exists="replace") == 3_000
    assert len(client.read_table("ud_wide")) == 3_000


def test_bound_parameters_defeat_injection(client: DatabaseClient) -> None:
    client.write_frame(
        pd.DataFrame({"name": ["ali", "sara"]}), "ud_inject", if_exists="replace"
    )
    frame = client.read_query(
        "SELECT * FROM ud_inject WHERE name = :name",
        params={"name": "ali'; DROP TABLE ud_inject; --"},
    )
    assert len(frame) == 0
    assert "ud_inject" in client.list_tables()


def test_cte_prefixed_dml_is_blocked(client: DatabaseClient) -> None:
    client.write_frame(pd.DataFrame({"a": [1, 2]}), "ud_cte", if_exists="replace")
    with pytest.raises(SecurityError):
        client.read_query("WITH x AS (SELECT 1) DELETE FROM ud_cte")
    assert len(client.read_table("ud_cte")) == 2


def test_dataset_end_to_end(client: DatabaseClient) -> None:
    source = Dataset(
        pd.DataFrame({"id": [1, 2, 2], "email": ["a@b.com", "bad", "c@d.org"]}),
        name="people",
    )
    # Rows 2 and 3 share an id but differ in email: a duplicate by key, not by row.
    cleaned = source.clean(remove_duplicates=True, duplicate_subset=["id"])
    assert cleaned.to_database(client, "ud_people", if_exists="replace") == 2
    loaded = Dataset.from_database(client, table="ud_people")
    assert loaded.n_rows == 2
