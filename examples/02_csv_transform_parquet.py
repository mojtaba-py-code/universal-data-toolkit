"""Example 2 - CSV, transformation chain, Parquet.

    python examples/02_csv_transform_parquet.py

Shows the transformation API: derived columns from a safe expression, filtering,
aggregation and a columnar output format.
"""

from __future__ import annotations

from pathlib import Path

from universal_data import Dataset
from universal_data.transformation import (
    Aggregate,
    CreateColumn,
    FilterRows,
    RenameColumns,
    SortRows,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "sample_data" / "orders_raw.csv"
TARGET = ROOT / "output" / "revenue_by_channel.parquet"


def main() -> None:
    # Lower-casing the text columns first matters: the raw export contains
    # "web", "Web" and "WEB", which would otherwise become three channels.
    orders = Dataset.read(SOURCE).clean(
        case="lower",
        convert_types={"quantity": "int", "unit_price": "float", "discount_pct": "float"},
        remove_duplicates=True,
        duplicate_subset=["order_id"],
    )

    summary = orders.transform(
        FilterRows("channel IS NOT NULL AND status != 'cancelled' AND quantity > 0"),
        CreateColumn("net_amount", "quantity * unit_price * (1 - discount_pct / 100)"),
        Aggregate(
            group_by=["channel"],
            aggregations={"net_amount": ["sum", "mean"], "order_id": "count"},
        ),
        RenameColumns(
            {
                "net_amount_sum": "revenue",
                "net_amount_mean": "average_order_value",
                "order_id_count": "orders",
            }
        ),
        SortRows("revenue", ascending=False),
    )

    print(summary.frame.to_string(index=False))
    summary.to_parquet(TARGET)
    print(f"\nWrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
