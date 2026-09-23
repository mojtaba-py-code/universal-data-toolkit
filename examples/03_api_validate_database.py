"""Example 3 - API, validation, database load.

    python examples/03_api_validate_database.py

Set ``EXAMPLE_API_URL`` to hit a real endpoint (``API_TOKEN`` is used as a bearer
token when present).  Without it the example falls back to the bundled JSON
sample so it still runs offline.

The destination is PostgreSQL when ``PG_PASSWORD`` and friends are set (see
``docker compose up -d postgres``) and a local SQLite file otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

from universal_data import DatabaseClient, DatabaseConfig, Dataset, configure_logging
from universal_data.core.types import FieldType
from universal_data.ingestion.api import APIClient, APIConfig, BearerTokenAuth, NoAuth
from universal_data.schema.model import ColumnSchema, Schema
from universal_data.security.network import URLPolicy
from universal_data.security.secrets import SecretResolver

ROOT = Path(__file__).resolve().parent.parent
FALLBACK = ROOT / "sample_data" / "products.json"
SQLITE_TARGET = ROOT / "output" / "products.db"

PRODUCT_SCHEMA = Schema.from_columns(
    [
        ColumnSchema(name="product_id", type=FieldType.INTEGER, nullable=False, unique=True),
        ColumnSchema(name="sku", type=FieldType.STRING, min_length=3),
        ColumnSchema(name="name", type=FieldType.STRING),
        ColumnSchema(name="category", type=FieldType.CATEGORICAL),
        ColumnSchema(name="unit_price", type=FieldType.FLOAT, min=0),
        ColumnSchema(name="stock", type=FieldType.INTEGER, min=0),
    ],
    name="product",
    primary_key=["product_id"],
)


def load_source() -> Dataset:
    url = os.environ.get("EXAMPLE_API_URL")
    if not url:
        print(f"EXAMPLE_API_URL is not set; reading {FALLBACK.name} instead")
        return Dataset.read(FALLBACK)

    resolver = SecretResolver()
    token = resolver.get("API_TOKEN")
    auth = BearerTokenAuth(token) if token else NoAuth()
    config = APIConfig(url=url, records_path=os.environ.get("EXAMPLE_API_RECORDS_PATH"))
    policy = URLPolicy(allow_http=os.environ.get("EXAMPLE_ALLOW_HTTP") == "1")
    with APIClient(config, auth=auth, url_policy=policy) as client:
        return Dataset.from_api(client, name="products")


def build_client() -> tuple[DatabaseClient | None, str]:
    password = os.environ.get("PG_PASSWORD")
    if not password:
        return None, "sqlite"
    config = DatabaseConfig.from_dict(
        {
            "driver": "postgresql",
            "host": os.environ.get("PGHOST", "localhost"),
            "port": os.environ.get("PGPORT", 5432),
            "database": os.environ.get("PGDATABASE", "datatoolkit"),
            "username": os.environ.get("PGUSER", "datatoolkit"),
            "password_env": "PG_PASSWORD",
        }
    )
    return DatabaseClient(config), "postgresql"


def main() -> None:
    configure_logging("INFO")

    dataset = load_source().clean(
        normalize_columns=True,
        convert_types={"unit_price": "float", "stock": "int"},
    )
    valid, invalid = dataset.keep_valid(PRODUCT_SCHEMA)
    print(f"{valid.n_rows:,} valid rows, {invalid.n_rows:,} rejected")
    if invalid.n_rows:
        invalid.to_csv(ROOT / "output" / "products_rejected.csv")

    client, kind = build_client()
    if client is None:
        rows = valid.to_sqlite(SQLITE_TARGET, table="products")
        print(f"Loaded {rows:,} rows into {SQLITE_TARGET.name} (set PG_PASSWORD to use PostgreSQL)")
        return

    with client:
        rows = valid.to_database(client, "products", if_exists="replace")
    print(f"Loaded {rows:,} rows into {kind}")


if __name__ == "__main__":
    main()
