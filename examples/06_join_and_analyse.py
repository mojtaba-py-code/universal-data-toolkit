"""Example 6 - join several datasets and produce an analytical table.

    python examples/06_join_and_analyse.py

Customers (CSV), orders (CSV) and products (JSON) are cleaned separately, joined,
enriched with a region lookup and aggregated into a per-region report.
"""

from __future__ import annotations

from pathlib import Path

from universal_data import Dataset
from universal_data.enrichment import LookupEnricher
from universal_data.transformation import Aggregate, CreateColumn, RenameColumns, SortRows

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_data"
TARGET = ROOT / "output" / "region_report.csv"


def main() -> None:
    customers = Dataset.read(SAMPLE / "customers_raw.csv").clean(
        normalize_columns=True,
        trim_whitespace=True,
        convert_types={"lifetime_value": "float"},
        remove_duplicates=True,
        duplicate_subset=["customer_id"],
    ).select(["customer_id", "country_code", "segment", "lifetime_value"])

    orders = Dataset.read(SAMPLE / "orders_raw.csv").clean(
        normalize_columns=True,
        convert_types={"quantity": "int", "unit_price": "float", "discount_pct": "float"},
        remove_duplicates=True,
        duplicate_subset=["order_id"],
    )

    products = Dataset.read(SAMPLE / "products.json").select(
        ["product_id", "category", "unit_cost"]
    )

    joined = (
        orders.join(customers, on="customer_id", how="inner")
        .join(products, on="product_id", how="left")
        .enrich(
            LookupEnricher.from_file(
                SAMPLE / "country_lookup.csv", on="country_code", columns=["region"]
            )
        )
    )
    print(f"Joined table: {joined.n_rows:,} rows x {joined.n_columns} columns")

    report = joined.transform(
        CreateColumn("net_amount", "quantity * unit_price * (1 - discount_pct / 100)"),
        CreateColumn("margin", "net_amount - quantity * unit_cost"),
        Aggregate(
            group_by=["region", "segment"],
            aggregations={
                "net_amount": ["sum"],
                "margin": ["sum"],
                "order_id": "count",
                "customer_id": "nunique",
            },
        ),
        RenameColumns(
            {
                "net_amount_sum": "revenue",
                "margin_sum": "margin",
                "order_id_count": "orders",
                "customer_id_nunique": "customers",
            }
        ),
        SortRows("revenue", ascending=False),
    )

    print(report.frame.head(15).to_string(index=False))
    report.to_csv(TARGET)
    print(f"\nWrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
