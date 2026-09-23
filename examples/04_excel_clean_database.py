"""Example 4 - Excel workbook to a database table.

    python examples/04_excel_clean_database.py

Reads the employees workbook, masks the personal columns, validates the result
and writes it to SQLite through the same interface used for PostgreSQL/MySQL.
"""

from __future__ import annotations

from pathlib import Path

from universal_data import DatabaseClient, DatabaseConfig, Dataset
from universal_data.masking import MaskingPolicy

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "sample_data" / "employees.xlsx"
TARGET = ROOT / "output" / "hr.db"


def main() -> None:
    employees = Dataset.read(SOURCE).clean(
        normalize_columns=True,
        trim_whitespace=True,
        string_columns=["department"],
        case="lower",
        convert_types={"salary": "float"},
        date_columns=["hire_date"],
        missing_values="median",
        remove_duplicates=True,
    )
    print(f"Cleaned {employees.n_rows:,} employee rows")

    schema = employees.detect_schema()
    print("Detected personal data in:", ", ".join(schema.pii_columns) or "nothing")

    masked = employees.mask(
        MaskingPolicy.from_dict(
            {
                "full_name": "name",
                "email": {"strategy": "hash", "length": 12},
            }
        )
    )

    config = DatabaseConfig.sqlite(str(TARGET))
    with DatabaseClient(config) as client:
        rows = masked.to_database(client, "employees", if_exists="replace")
        print(f"Wrote {rows:,} rows to {TARGET.name}")
        print("Tables:", client.list_tables())
        by_department = client.read_query(
            "SELECT department, COUNT(*) AS headcount, ROUND(AVG(salary), 2) AS avg_salary "
            "FROM employees GROUP BY department ORDER BY headcount DESC"
        )
        print(by_department.to_string(index=False))


if __name__ == "__main__":
    main()
