"""The full demonstration: raw customer data to a loaded, documented dataset.

    python examples/08_end_to_end_demo.py

    Raw CSV -> inspect -> detect schema -> validate -> clean -> deduplicate
            -> normalise -> mask PII -> business rules -> quality analysis
            -> HTML/JSON report -> Parquet -> database

The database step uses PostgreSQL when ``PG_PASSWORD`` is set and SQLite
otherwise, so the whole script runs without any external service.
"""

from __future__ import annotations

import os
from pathlib import Path

from universal_data import DatabaseClient, DatabaseConfig, Dataset, configure_logging
from universal_data.masking import MaskingPolicy
from universal_data.quality.report import QualityDocument
from universal_data.rules.engine import RuleEngine
from universal_data.schema.loader import load_schema, save_schema

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
SOURCE = ROOT / "sample_data" / "customers_raw.csv"


def rule_engine() -> RuleEngine:
    import yaml

    definitions = yaml.safe_load(
        (ROOT / "configs" / "customer_rules.yaml").read_text(encoding="utf-8")
    )["rules"]
    return RuleEngine.from_config(definitions)


def step(number: int, title: str) -> None:
    print(f"\n{'=' * 70}\n{number}. {title}\n{'=' * 70}")


def main() -> None:
    configure_logging("WARNING")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    step(1, "Load the raw export")
    raw = Dataset.read(SOURCE)
    print(f"{raw.n_rows:,} rows x {raw.n_columns} columns, {raw.memory_mb:.2f} MB")

    step(2, "Inspect")
    profile = raw.profile()
    print(profile.summary())

    step(3, "Detect the schema")
    detected = raw.detect_schema(infer_constraints=True)
    save_schema(detected, OUTPUT / "customer_detected_schema.yaml")
    print(detected.describe())
    print(f"\nPossible personal data: {', '.join(detected.pii_columns)}")

    step(4, "Validate against the declared schema")
    schema = load_schema(ROOT / "schemas" / "customer.yaml")
    before = raw.validate(schema)
    print(before.summary())

    step(5, "Clean, deduplicate and normalise")
    cleaned = raw.clean(
        normalize_columns=True,
        trim_whitespace=True,
        normalize_spaces=True,
        convert_types={"age": "int", "lifetime_value": "float"},
        date_columns=["signup_date"],
        missing_values="median",
        remove_duplicates=True,
    )
    cleaning_report = cleaned.last_cleaning_report()
    if cleaning_report:
        print(cleaning_report.summary())

    step(6, "Cap the outliers instead of deleting them")
    capped = cleaned.handle_outliers(["lifetime_value"], method="iqr", action="cap")
    print(capped.detect_outliers(["lifetime_value"]))

    step(7, "Apply the business rules")
    rules = rule_engine()
    flagged, evaluation = capped.apply_rules(rules, action="flag")
    for outcome in evaluation.outcomes:
        print(f"  {outcome.name:<22} {outcome.passed:>6,} pass / {outcome.failed:>5,} fail")
    print(f"\n{evaluation.rows_failed:,} rows violate at least one blocking rule")

    step(8, "Score the data quality")
    # Scored before masking: a masked email is deliberately not a valid email,
    # so measuring quality afterwards would report the masking, not the data.
    quality = flagged.quality(schema, rules)
    print(quality.summary())

    step(9, "Write the report")
    document = QualityDocument(
        dataset="customers",
        source=str(SOURCE),
        profile=flagged.profile(),
        validation=flagged.validate(schema),
        rules=evaluation,
        quality=quality,
        cleaning=cleaning_report,
    )
    document.to_html(OUTPUT / "customers_report.html")
    document.to_json(OUTPUT / "customers_report.json")
    print(f"Report written to {OUTPUT / 'customers_report.html'}")

    step(10, "Mask the personal data before it leaves the pipeline")
    masked = flagged.mask(
        MaskingPolicy.from_dict(
            {
                "email": "email",
                "phone": {"strategy": "phone", "keep_last": 4},
                "first_name": "name",
                "last_name": "name",
            }
        )
    )
    print(masked.frame[["email", "phone", "first_name"]].head(5).to_string(index=False))

    step(11, "Export to Parquet")
    parquet_rows = masked.to_parquet(OUTPUT / "customers_clean.parquet")
    print(f"{parquet_rows:,} rows written")

    step(12, "Load into the database")
    if os.environ.get("PG_PASSWORD"):
        config = DatabaseConfig.from_dict(
            {
                "driver": "postgresql",
                "host": os.environ.get("PGHOST", "localhost"),
                "database": os.environ.get("PGDATABASE", "datatoolkit"),
                "username": os.environ.get("PGUSER", "datatoolkit"),
                "password_env": "PG_PASSWORD",
            }
        )
        target = "PostgreSQL"
    else:
        config = DatabaseConfig.sqlite(str(OUTPUT / "customers.db"))
        target = "SQLite (set PG_PASSWORD to use PostgreSQL)"

    with DatabaseClient(config) as client:
        rows = masked.to_database(client, "customers", if_exists="replace")
    print(f"{rows:,} rows loaded into {target}")

    print("\nDone. Artefacts are in the output/ directory.")


if __name__ == "__main__":
    main()
