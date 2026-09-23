"""Regenerate the files in ``sample_data/``.

    python scripts/generate_sample_data.py

The output is deterministic for a given seed, so the repository stays stable
between runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from universal_data import Dataset
from universal_data.generator.synthetic import QualityIssues, SyntheticDataGenerator

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--customers", type=int, default=800)
    parser.add_argument("--orders", type=int, default=3_000)
    parser.add_argument("--products", type=int, default=150)
    parser.add_argument("--employees", type=int, default=200)
    args = parser.parse_args()

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    generator = SyntheticDataGenerator(seed=args.seed)
    issues = QualityIssues.realistic()

    customers = generator.customers(args.customers, issues=issues)
    products = generator.products(args.products, issues=QualityIssues.clean())
    orders = generator.orders(
        args.orders,
        customer_ids=list(range(1, args.customers + 1)),
        product_ids=list(range(1, args.products + 1)),
        issues=issues,
    )
    employees = generator.employees(args.employees, issues=issues)

    Dataset(customers, name="customers").to_csv(SAMPLE_DIR / "customers_raw.csv")
    Dataset(products, name="products").to_json(SAMPLE_DIR / "products.json")
    Dataset(orders, name="orders").to_csv(SAMPLE_DIR / "orders_raw.csv")
    Dataset(employees, name="employees").to_excel(SAMPLE_DIR / "employees.xlsx")
    Dataset(generator.country_lookup(), name="countries").to_csv(
        SAMPLE_DIR / "country_lookup.csv"
    )

    for path in sorted(SAMPLE_DIR.iterdir()):
        print(f"{path.name:<24} {path.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
