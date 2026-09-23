"""Example 1 - CSV in, cleaned CSV out.

    python examples/01_csv_clean_csv.py

The smallest useful pipeline: read a messy export, normalise it, drop the
duplicates and write it back out.
"""

from __future__ import annotations

from pathlib import Path

from universal_data import Dataset, configure_logging

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "sample_data" / "customers_raw.csv"
TARGET = ROOT / "output" / "customers_clean.csv"


def main() -> None:
    configure_logging("INFO")

    dataset = Dataset.read(SOURCE)
    print(f"Loaded {dataset.n_rows:,} rows x {dataset.n_columns} columns")

    cleaned = dataset.clean(
        normalize_columns=True,
        trim_whitespace=True,
        convert_types={"age": "int", "lifetime_value": "float"},
        date_columns=["signup_date"],
        missing_values="median",
        remove_duplicates=True,
    )

    report = cleaned.last_cleaning_report()
    if report:
        print(report.summary())

    rows = cleaned.to_csv(TARGET)
    print(f"\nWrote {rows:,} rows to {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
