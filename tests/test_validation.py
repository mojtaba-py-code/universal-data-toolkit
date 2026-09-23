"""Schema validation."""

from __future__ import annotations

import pandas as pd
import pytest

from universal_data.core.exceptions import DataValidationError
from universal_data.core.types import FieldType, Severity
from universal_data.schema.model import Schema
from universal_data.validation import checks
from universal_data.validation.engine import SchemaValidator, validate_frame
from universal_data.validation.results import ValidationIssue, ValidationResult


@pytest.fixture
def schema() -> Schema:
    return Schema.from_dict(
        {
            "name": "people",
            "primary_key": ["id"],
            "schema": {
                "id": {"type": "integer", "nullable": False, "unique": True},
                "email": {"type": "email", "required": True},
                "age": {"type": "integer", "min": 18, "max": 100},
                "country": {"type": "categorical", "allowed": ["IR", "DE"]},
                "code": {"type": "string", "pattern": r"[A-Z]{3}"},
            },
        }
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "email": ["a@b.com", "bad-email", "c@d.org", None],
            "age": [30, 15, 200, 40],
            "country": ["IR", "DE", "FR", "IR"],
            "code": ["ABC", "abc", "ABC", "ABCD"],
        }
    )


class TestChecks:
    def test_nulls_never_count_as_type_errors(self) -> None:
        series = pd.Series([1, None, 3])
        assert checks.check_integer(series).all()

    def test_integer_check_rejects_fractions(self) -> None:
        assert checks.check_integer(pd.Series(["1", "2.5"])).tolist() == [True, False]

    def test_float_check(self) -> None:
        assert checks.check_float(pd.Series(["1.5", "x"])).tolist() == [True, False]

    def test_boolean_check_accepts_tokens(self) -> None:
        assert checks.check_boolean(pd.Series(["yes", "no", "maybe"])).tolist() == [
            True, True, False
        ]

    def test_datetime_check(self) -> None:
        assert checks.check_datetime(pd.Series(["2024-01-01", "nope"])).tolist() == [True, False]

    def test_string_check_rejects_containers(self) -> None:
        assert checks.check_string(pd.Series([["a"], "b"])).tolist() == [False, True]

    def test_length_check(self) -> None:
        result = checks.check_length(pd.Series(["ab", "abcd"]), minimum=3)
        assert result.tolist() == [False, True]

    def test_allowed_compares_as_text_too(self) -> None:
        assert checks.check_allowed(pd.Series([1, 2]), ["1", "3"]).tolist() == [True, False]

    def test_unique_check(self) -> None:
        assert checks.check_unique(pd.Series([1, 1, 2])).tolist() == [False, False, True]

    def test_bounds_on_dates(self) -> None:
        series = pd.Series(["2020-01-01", "2024-01-01"])
        result = checks.check_min(series, "2022-01-01", FieldType.DATETIME)
        assert result.tolist() == [False, True]


class TestValidator:
    def test_reports_every_failing_check(self, frame: pd.DataFrame, schema: Schema) -> None:
        result = SchemaValidator(schema).validate(frame)
        checks_found = {issue.check for issue in result.issues}
        assert {"type", "min", "max", "allowed", "pattern"} <= checks_found
        assert not result.is_valid

    def test_row_mask_marks_bad_rows(self, frame: pd.DataFrame, schema: Schema) -> None:
        result = SchemaValidator(schema).validate(frame)
        assert result.row_mask.tolist() == [True, False, False, False]
        assert result.valid_rows == 1
        assert result.invalid_rows == 3

    def test_valid_and_invalid_frames(self, frame: pd.DataFrame, schema: Schema) -> None:
        result = SchemaValidator(schema).validate(frame)
        assert len(result.valid_frame(frame)) == 1
        assert len(result.invalid_frame(frame)) == 3

    def test_missing_required_column_is_an_error(self, schema: Schema) -> None:
        result = SchemaValidator(schema).validate(pd.DataFrame({"id": [1]}))
        issues = {issue.check: issue for issue in result.issues}
        assert issues["missing_column"].severity is Severity.ERROR

    def test_missing_optional_column_is_a_warning(self) -> None:
        schema = Schema.from_dict({"schema": {"a": {"type": "integer", "required": False}}})
        result = SchemaValidator(schema).validate(pd.DataFrame({"b": [1]}))
        assert result.is_valid
        assert result.warnings

    def test_unexpected_columns_warn_unless_strict(self) -> None:
        schema = Schema.from_dict({"schema": {"a": "integer"}})
        result = SchemaValidator(schema).validate(pd.DataFrame({"a": [1], "b": [2]}))
        assert result.is_valid
        strict = Schema.from_dict({"schema": {"a": "integer"}, "strict": True})
        assert not SchemaValidator(strict).validate(pd.DataFrame({"a": [1], "b": [2]})).is_valid

    def test_not_null_constraint(self) -> None:
        schema = Schema.from_dict({"schema": {"a": {"type": "integer", "nullable": False}}})
        result = SchemaValidator(schema).validate(pd.DataFrame({"a": [1, None]}))
        assert any(issue.check == "not_null" for issue in result.issues)

    def test_primary_key_duplicates(self) -> None:
        schema = Schema.from_dict(
            {"schema": {"id": {"type": "integer"}}, "primary_key": ["id"]}
        )
        result = SchemaValidator(schema).validate(pd.DataFrame({"id": [1, 1]}))
        assert any(issue.check == "primary_key" for issue in result.issues)
        assert result.invalid_rows == 2

    def test_raise_on_error(self, frame: pd.DataFrame, schema: Schema) -> None:
        with pytest.raises(DataValidationError) as info:
            SchemaValidator(schema).validate(frame, raise_on_error=True)
        assert info.value.errors

    def test_clean_dataset_passes(self, schema: Schema) -> None:
        frame = pd.DataFrame(
            {
                "id": [1, 2],
                "email": ["a@b.com", "c@d.com"],
                "age": [20, 30],
                "country": ["IR", "DE"],
                "code": ["ABC", "DEF"],
            }
        )
        result = validate_frame(frame, schema)
        assert result.is_valid
        assert "Validation passed" in result.summary()

    def test_samples_are_truncated(self) -> None:
        schema = Schema.from_dict({"schema": {"a": {"type": "integer"}}})
        frame = pd.DataFrame({"a": ["x" * 200]})
        result = SchemaValidator(schema).validate(frame)
        assert result.issues[0].samples[0].endswith("...")

    def test_empty_frame_is_valid(self, schema: Schema) -> None:
        empty = pd.DataFrame(columns=["id", "email", "age", "country", "code"])
        result = SchemaValidator(schema).validate(empty)
        assert result.total_rows == 0


class TestResults:
    def test_issue_serialisation(self) -> None:
        issue = ValidationIssue(
            column="a", check="type", message="bad", failed_rows=2, total_rows=4
        )
        assert issue.failure_rate == 0.5
        assert issue.to_dict()["failure_rate"] == 0.5
        assert "bad" in str(issue)

    def test_result_summary_and_dict(self) -> None:
        result = ValidationResult(total_rows=2, schema_name="s")
        result.add(ValidationIssue(column=None, check="x", message="m", severity=Severity.WARNING))
        assert result.is_valid
        assert result.to_dict()["warning_count"] == 1
        assert "warning" in result.summary()

    def test_result_without_mask(self) -> None:
        frame = pd.DataFrame({"a": [1]})
        result = ValidationResult(total_rows=1)
        assert result.invalid_rows == 0
        assert len(result.valid_frame(frame)) == 1
        assert len(result.invalid_frame(frame)) == 0
