"""The Dataset façade."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from universal_data.core.dataset import Dataset
from universal_data.core.exceptions import ConfigurationError, DataProcessingError
from universal_data.masking.maskers import MaskingPolicy
from universal_data.rules.engine import FAILURE_COLUMN, BusinessRule, RuleEngine
from universal_data.schema.loader import load_schema
from universal_data.security.paths import PathPolicy


class TestConstruction:
    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(DataProcessingError, match="expects a pandas DataFrame"):
            Dataset([1, 2, 3])  # type: ignore[arg-type]

    def test_from_records(self) -> None:
        dataset = Dataset.from_records([{"a": 1}, {"a": 2}])
        assert dataset.shape == (2, 1)

    def test_read_detects_the_format(self, customers_csv: Path) -> None:
        dataset = Dataset.read(customers_csv)
        assert dataset.name == "customers"
        assert dataset.metadata["format"] == "csv"
        assert dataset.history[0]["operation"] == "read"

    def test_typed_constructors(self, tmp_path: Path, customers_frame: pd.DataFrame) -> None:
        parquet = tmp_path / "c.parquet"
        customers_frame.to_parquet(parquet, index=False)
        assert Dataset.from_parquet(parquet).n_rows == len(customers_frame)

    def test_read_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "big.csv"
        pd.DataFrame({"a": range(10)}).to_csv(path, index=False)
        chunks = list(Dataset.read_chunks(path, 4))
        assert [chunk.n_rows for chunk in chunks] == [4, 4, 2]
        assert chunks[0].metadata["chunk_index"] == 0

    def test_path_policy_is_honoured(self, tmp_path: Path) -> None:
        policy = PathPolicy.confined_to(tmp_path / "inside")
        (tmp_path / "outside.csv").write_text("a\n1\n", encoding="utf-8")
        from universal_data.core.exceptions import SecurityError

        with pytest.raises(SecurityError):
            Dataset.read(tmp_path / "outside.csv", policy=policy)


class TestProperties:
    def test_basic_properties(self, customers_dataset: Dataset) -> None:
        assert customers_dataset.n_rows == 6
        assert customers_dataset.n_columns == 7
        assert "email" in customers_dataset.columns
        assert customers_dataset.memory_mb > 0
        assert not customers_dataset.empty
        assert len(customers_dataset) == 6
        assert "Dataset(" in repr(customers_dataset)
        assert "customer_id" in customers_dataset.dtypes

    def test_preview_helpers(self, customers_dataset: Dataset) -> None:
        assert len(customers_dataset.head(2)) == 2
        assert len(customers_dataset.tail(2)) == 2
        assert len(customers_dataset.sample(3, random_state=1)) == 3

    def test_to_records(self, customers_dataset: Dataset) -> None:
        assert len(customers_dataset.to_records()) == 6


class TestOperations:
    def test_operations_are_immutable(self, customers_dataset: Dataset) -> None:
        original_rows = customers_dataset.n_rows
        deduplicated = customers_dataset.remove_duplicates()
        assert customers_dataset.n_rows == original_rows
        assert deduplicated.n_rows == 5

    def test_history_accumulates(self, customers_dataset: Dataset) -> None:
        result = customers_dataset.remove_duplicates().filter("age > 18")
        operations = [entry["operation"] for entry in result.history]
        assert operations == ["remove_duplicates", "transform"]

    def test_clean_with_keyword_arguments(self, customers_dataset: Dataset) -> None:
        cleaned = customers_dataset.clean(missing_values="mode", remove_duplicates=True)
        assert cleaned.n_rows == 5
        assert cleaned.last_cleaning_report() is not None

    def test_clean_with_dict(self, customers_dataset: Dataset) -> None:
        assert customers_dataset.clean({"remove_duplicates": True}).n_rows == 5

    def test_clean_with_config_object(self, customers_dataset: Dataset) -> None:
        from universal_data.cleaning.engine import CleaningConfig

        assert customers_dataset.clean(CleaningConfig(remove_duplicates=True)).n_rows == 5

    def test_fill_and_drop_missing(self, customers_dataset: Dataset) -> None:
        assert customers_dataset.fill_missing("median", columns=["salary"]).frame["salary"].notna().all()
        assert customers_dataset.drop_missing(["email"]).n_rows == 4

    def test_column_operations(self, customers_dataset: Dataset) -> None:
        result = (
            customers_dataset.select(["customer_id", "age", "salary"])
            .rename({"age": "years"})
            .drop(["salary"])
            .sort("years", ascending=False)
        )
        assert result.columns == ["customer_id", "years"]
        assert result.frame["years"].iloc[0] == 51

    def test_add_column_and_filter(self, customers_dataset: Dataset) -> None:
        result = customers_dataset.add_column("adult", "age >= 18").filter("adult")
        assert result.n_rows == 5

    def test_join(self, customers_dataset: Dataset) -> None:
        lookup = pd.DataFrame({"country": ["IR", "DE", "FR"], "region": ["ME", "EU", "EU"]})
        result = customers_dataset.join(lookup, on="country")
        assert "region" in result.columns

    def test_aggregate(self, customers_dataset: Dataset) -> None:
        result = customers_dataset.aggregate("country", {"salary": "mean"})
        assert result.n_rows == 3

    def test_outliers(self, customers_dataset: Dataset) -> None:
        # Two identical extremes inflate the IQR, so this small sample needs the
        # percentile method to flag them - which is exactly why the method is
        # configurable.
        report = customers_dataset.detect_outliers(
            ["salary"], method="percentile", threshold=0.3
        )
        assert report["total_outliers"] >= 1
        capped = customers_dataset.handle_outliers(
            ["salary"], method="percentile", threshold=0.3, action="cap"
        )
        assert capped.frame["salary"].max() < 99999.0

    def test_copy(self, customers_dataset: Dataset) -> None:
        copy = customers_dataset.copy()
        assert copy.frame is not customers_dataset.frame
        assert copy.history[-1]["operation"] == "copy"


class TestSchemaAndRules:
    def test_detect_schema(self, customers_dataset: Dataset) -> None:
        schema = customers_dataset.detect_schema()
        assert "email" in schema.pii_columns

    def test_validate_requires_a_schema(self, customers_dataset: Dataset) -> None:
        with pytest.raises(ConfigurationError, match="No schema"):
            customers_dataset.validate()

    def test_with_schema_then_validate(self, customers_dataset: Dataset, schema_file: Path) -> None:
        schema = load_schema(schema_file)
        attached = customers_dataset.with_schema(schema)
        assert attached.schema is schema
        assert attached.validate().invalid_rows > 0

    def test_keep_valid_splits_the_dataset(self, customers_dataset: Dataset, schema_file: Path) -> None:
        valid, invalid = customers_dataset.keep_valid(load_schema(schema_file))
        assert valid.n_rows + invalid.n_rows == customers_dataset.n_rows
        assert invalid.name.endswith("_invalid")

    def test_apply_rules_flag(self, customers_dataset: Dataset) -> None:
        engine = RuleEngine([BusinessRule(name="adult", condition="age >= 18")])
        flagged, evaluation = customers_dataset.apply_rules(engine)
        assert FAILURE_COLUMN in flagged.columns
        assert evaluation.rows_failed == 1

    def test_apply_rules_drop(self, customers_dataset: Dataset) -> None:
        engine = RuleEngine([BusinessRule(name="adult", condition="age >= 18")])
        dropped, _ = customers_dataset.apply_rules(engine, action="drop")
        assert dropped.n_rows == 5

    def test_apply_rules_validates_action(self, customers_dataset: Dataset) -> None:
        engine = RuleEngine([BusinessRule(name="adult", condition="age >= 18")])
        with pytest.raises(ConfigurationError, match="flag"):
            customers_dataset.apply_rules(engine, action="explode")


class TestMaskingAndEnrichment:
    def test_mask_with_a_dict(self, customers_dataset: Dataset) -> None:
        masked = customers_dataset.mask({"email": "email"})
        assert masked.frame["email"].dropna().iloc[0].startswith("a***@")

    def test_mask_with_a_policy(self, customers_dataset: Dataset) -> None:
        masked = customers_dataset.mask(MaskingPolicy.for_columns(["email"], "redact"))
        assert masked.frame["email"].dropna().iloc[0] == "[REDACTED]"

    def test_mask_detected_pii(self, customers_dataset: Dataset) -> None:
        masked = customers_dataset.mask_detected_pii("redact")
        assert masked.frame["email"].dropna().iloc[0] == "[REDACTED]"

    def test_mask_detected_pii_without_pii(self) -> None:
        dataset = Dataset(pd.DataFrame({"total": [1.0, 2.0]}))
        result = dataset.mask_detected_pii()
        assert result is not dataset                 # every operation returns a new dataset
        assert result.frame.equals(dataset.frame)
        assert result.history[-1]["operation"] == "mask"

    def test_enrich(self, customers_dataset: Dataset) -> None:
        from universal_data.enrichment.enrichers import DerivedColumnEnricher

        result = customers_dataset.enrich(DerivedColumnEnricher("double", "salary * 2"))
        assert "double" in result.columns
        assert result.history[-1]["enricher"] == "DerivedColumnEnricher"


class TestExport:
    @pytest.mark.parametrize(
        "method", ["to_csv", "to_json", "to_jsonl", "to_excel", "to_parquet"]
    )
    def test_export_methods(self, customers_dataset: Dataset, tmp_path: Path, method: str) -> None:
        target = tmp_path / f"out_{method}"
        assert getattr(customers_dataset, method)(target) == 6

    def test_to_sqlite(self, customers_dataset: Dataset, tmp_path: Path) -> None:
        target = tmp_path / "out.db"
        assert customers_dataset.to_sqlite(target, table="customers") == 6

    def test_write_infers_the_format(self, customers_dataset: Dataset, tmp_path: Path) -> None:
        assert customers_dataset.write(tmp_path / "out.parquet") == 6


class TestReporting:
    def test_profile_and_quality(self, customers_dataset: Dataset) -> None:
        profile = customers_dataset.profile()
        assert profile.rows == 6
        report = customers_dataset.quality()
        assert 0 <= report.overall_score <= 1

    def test_report_document(self, customers_dataset: Dataset, schema_file: Path) -> None:
        engine = RuleEngine([BusinessRule(name="adult", condition="age >= 18")])
        document = customers_dataset.report(schema=load_schema(schema_file), rules=engine)
        payload = document.to_dict()
        assert payload["profile"]["rows"] == 6
        assert payload["validation"] is not None
        assert payload["business_rules"]["rows_failed"] >= 1
