"""Cleaning operations and the cleaning engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from universal_data.cleaning.engine import CleaningConfig, DataCleaner
from universal_data.cleaning.operations import (
    clean_numeric,
    clean_strings,
    convert_types,
    drop_empty,
    find_duplicates,
    handle_missing,
    normalize_column_names,
    parse_dates,
    remove_duplicates,
)
from universal_data.core.exceptions import TransformationError
from universal_data.core.types import DuplicateKeep, MissingStrategy


class TestMissingValues:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "x"], "c": [None, None, None]})

    def test_none_strategy_changes_nothing(self, frame: pd.DataFrame) -> None:
        result, report = handle_missing(frame, MissingStrategy.NONE)
        assert result.equals(frame)
        assert report["filled"] == 0

    def test_drop_rows(self, frame: pd.DataFrame) -> None:
        result, report = handle_missing(frame, "drop_rows", columns=["a"])
        assert len(result) == 2
        assert report["rows_dropped"] == 1

    def test_drop_columns_above_threshold(self, frame: pd.DataFrame) -> None:
        result, report = handle_missing(frame, "drop_columns", threshold=0.5)
        assert "c" not in result.columns
        assert report["columns_dropped"] == ["c"]

    def test_mean_and_median(self, frame: pd.DataFrame) -> None:
        mean_result, _ = handle_missing(frame, "mean", columns=["a"])
        assert mean_result["a"].tolist() == [1.0, 2.0, 3.0]
        median_result, _ = handle_missing(frame, "median", columns=["a"])
        assert median_result["a"].tolist() == [1.0, 2.0, 3.0]

    def test_mode_for_text(self, frame: pd.DataFrame) -> None:
        result, _ = handle_missing(frame, "mode", columns=["b"])
        assert result["b"].tolist() == ["x", "x", "x"]

    def test_mean_on_text_falls_back_to_mode(self, frame: pd.DataFrame) -> None:
        result, _ = handle_missing(frame, "mean", columns=["b"])
        assert result["b"].isna().sum() == 0

    def test_constant_requires_a_value(self, frame: pd.DataFrame) -> None:
        with pytest.raises(TransformationError, match="needs a value"):
            handle_missing(frame, "constant", columns=["a"])

    def test_constant_fill(self, frame: pd.DataFrame) -> None:
        result, _ = handle_missing(frame, "constant", columns=["a"], value=0)
        assert result["a"].tolist() == [1.0, 0.0, 3.0]

    def test_forward_and_backward_fill(self, frame: pd.DataFrame) -> None:
        forward, _ = handle_missing(frame, "ffill", columns=["a"])
        assert forward["a"].tolist() == [1.0, 1.0, 3.0]
        backward, _ = handle_missing(frame, "bfill", columns=["a"])
        assert backward["a"].tolist() == [1.0, 3.0, 3.0]

    def test_integer_column_gets_an_integer_fill(self) -> None:
        frame = pd.DataFrame({"a": pd.array([1, None, 4], dtype="Int64")})
        result, _ = handle_missing(frame, "median", columns=["a"])
        assert result["a"].tolist() == [1, 2, 4]

    def test_all_null_column_is_skipped(self) -> None:
        frame = pd.DataFrame({"a": [None, None]})
        result, _ = handle_missing(frame, "mode", columns=["a"])
        assert result["a"].isna().all()

    def test_unknown_column_is_reported(self, frame: pd.DataFrame) -> None:
        with pytest.raises(TransformationError, match="Unknown column"):
            handle_missing(frame, "mean", columns=["zzz"])


class TestDuplicates:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"id": [1, 1, 2, 3, 3], "v": ["a", "a", "b", "c", "d"]})

    def test_remove_keeps_first(self, frame: pd.DataFrame) -> None:
        result, report = remove_duplicates(frame)
        assert len(result) == 4
        assert report["rows_removed"] == 1

    def test_remove_on_subset(self, frame: pd.DataFrame) -> None:
        result, report = remove_duplicates(frame, subset=["id"])
        assert len(result) == 3
        assert report["duplicate_rows"] == 4

    def test_keep_none_drops_all_copies(self, frame: pd.DataFrame) -> None:
        result, _ = remove_duplicates(frame, subset=["id"], keep=DuplicateKeep.NONE)
        assert result["id"].tolist() == [2]

    def test_find_duplicates(self, frame: pd.DataFrame) -> None:
        assert len(find_duplicates(frame, subset=["id"])) == 4


class TestStringCleaning:
    def test_trim_and_collapse_spaces(self) -> None:
        frame = pd.DataFrame({"name": ["  Ali  Reza ", "Sara"]})
        result, report = clean_strings(frame)
        assert result["name"].tolist() == ["Ali Reza", "Sara"]
        assert report["cells_changed"] == 1

    @pytest.mark.parametrize(
        ("case", "expected"), [("lower", "ali"), ("upper", "ALI"), ("title", "Ali")]
    )
    def test_case_normalisation(self, case: str, expected: str) -> None:
        result, _ = clean_strings(pd.DataFrame({"n": ["aLi"]}), case=case)
        assert result["n"].iloc[0] == expected

    def test_invalid_case_is_rejected(self) -> None:
        with pytest.raises(TransformationError, match="case must be"):
            clean_strings(pd.DataFrame({"n": ["a"]}), case="sentence")

    def test_remove_special_characters(self) -> None:
        result, _ = clean_strings(pd.DataFrame({"n": ["a*b#c"]}), remove_special=True)
        assert result["n"].iloc[0] == "abc"

    def test_regex_replacements(self) -> None:
        result, _ = clean_strings(
            pd.DataFrame({"n": ["tel: 123"]}), replacements=[(r"tel:\s*", "")]
        )
        assert result["n"].iloc[0] == "123"

    def test_empty_strings_become_null(self) -> None:
        result, _ = clean_strings(pd.DataFrame({"n": ["   ", "x"]}))
        assert result["n"].isna().tolist() == [True, False]

    def test_unicode_normalisation(self) -> None:
        result, _ = clean_strings(pd.DataFrame({"n": ["ｆｕｌｌ"]}), normalize_unicode=True)
        assert result["n"].iloc[0] == "full"

    def test_frame_without_text_columns(self) -> None:
        frame = pd.DataFrame({"a": [1, 2]})
        result, report = clean_strings(frame)
        assert report["columns"] == []
        assert result.equals(frame)


class TestColumnNames:
    def test_snake_case_conversion(self) -> None:
        frame = pd.DataFrame(columns=["Customer Name", "totalAmount", "  weird!!name  ", "ID"])
        result, report = normalize_column_names(frame)
        assert list(result.columns) == ["customer_name", "total_amount", "weird_name", "id"]
        assert report["renamed"]["Customer Name"] == "customer_name"

    def test_collisions_get_a_suffix(self) -> None:
        frame = pd.DataFrame(columns=["a b", "a_b"])
        result, _ = normalize_column_names(frame)
        assert list(result.columns) == ["a_b", "a_b_2"]


class TestTypeConversion:
    def test_converts_numeric_and_dates(self) -> None:
        frame = pd.DataFrame({"n": ["1", "2", "x"], "d": ["2024-01-01", "nope", None]})
        result, report = convert_types(frame, {"n": "int", "d": "datetime"})
        assert result["n"].tolist()[:2] == [1, 2]
        assert report["coerced_to_null"]["n"] == 1
        assert pd.api.types.is_datetime64_any_dtype(result["d"])

    def test_boolean_conversion(self) -> None:
        frame = pd.DataFrame({"b": ["yes", "NO", "maybe"]})
        result, _ = convert_types(frame, {"b": "bool"})
        assert result["b"].tolist()[:2] == [True, False]

    def test_boolean_conversion_keeps_bool_dtype(self) -> None:
        frame = pd.DataFrame({"b": [True, False]})
        result, _ = convert_types(frame, {"b": "boolean"})
        assert result["b"].tolist() == [True, False]

    def test_string_and_category(self) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result, _ = convert_types(frame, {"a": "string", "b": "category"})
        assert str(result["a"].dtype) == "string"
        assert str(result["b"].dtype) == "category"

    def test_date_is_normalised(self) -> None:
        frame = pd.DataFrame({"d": ["2024-01-01 13:45:00"]})
        result, _ = convert_types(frame, {"d": "date"})
        assert result["d"].iloc[0].hour == 0

    def test_unknown_target_type(self) -> None:
        with pytest.raises(TransformationError, match="Unsupported target type"):
            convert_types(pd.DataFrame({"a": [1]}), {"a": "complex"})


class TestDatesAndNumbers:
    def test_parse_dates_reports_failures(self) -> None:
        frame = pd.DataFrame({"d": ["2024-01-01", "not a date"]})
        result, report = parse_dates(frame, ["d"])
        assert report["unparseable"]["d"] == 1
        assert result["d"].isna().sum() == 1

    def test_parse_dates_with_timezone(self) -> None:
        frame = pd.DataFrame({"d": ["2024-01-01 10:00:00"]})
        result, _ = parse_dates(frame, ["d"], timezone="UTC")
        assert result["d"].dt.tz is not None

    def test_clean_numeric_strips_symbols(self) -> None:
        frame = pd.DataFrame({"amount": ["$1,200", "2 500", "n/a"]})
        result, report = clean_numeric(frame, ["amount"])
        assert result["amount"].tolist()[:2] == [1200.0, 2500.0]
        assert report["coerced_to_null"]["amount"] == 1

    def test_clean_numeric_range(self) -> None:
        frame = pd.DataFrame({"amount": [1.0, -5.0, 1000.0]})
        result, report = clean_numeric(frame, ["amount"], minimum=0, maximum=100)
        assert report["out_of_range"]["amount"] == 2
        assert np.isnan(result["amount"].iloc[1])

    def test_drop_empty(self) -> None:
        frame = pd.DataFrame({"a": [1, None], "b": [None, None]})
        result, report = drop_empty(frame)
        assert "b" not in result.columns
        assert report["rows_dropped"] == 1


class TestCleaningEngine:
    def test_full_pipeline(self) -> None:
        frame = pd.DataFrame(
            {
                "Customer Name": ["  Ali ", "ALI", "Sara", None],
                "Age": ["30", "30", None, "40"],
                "Empty": [None, None, None, None],
            }
        )
        config = CleaningConfig(
            normalize_columns=True,
            drop_empty_columns=True,
            case="lower",
            convert_types={"age": "int"},
            missing_values=MissingStrategy.MEDIAN,
            remove_duplicates=True,
        )
        result, report = DataCleaner(config).clean(frame)
        assert list(result.columns) == ["customer_name", "age"]
        assert len(result) == 3
        assert report.rows_before == 4
        assert "Cleaning removed" in report.summary()
        assert report.to_dict()["columns_after"] == 2

    def test_config_from_dict_with_aliases(self) -> None:
        config = CleaningConfig.from_dict({"duplicates": True, "lowercase": "lower"})
        assert config.remove_duplicates is True
        assert config.case == "lower"

    def test_config_rejects_unknown_options(self) -> None:
        with pytest.raises(ValueError, match="Unknown cleaning option"):
            CleaningConfig.from_dict({"nope": 1})

    def test_config_accepts_replacement_mapping(self) -> None:
        config = CleaningConfig.from_dict({"replacements": {"a": "b"}})
        assert config.replacements == (("a", "b"),)

    def test_default_config_is_conservative(self) -> None:
        frame = pd.DataFrame({"a": [1, 1], "b": [" x ", " x "]})
        result, _ = DataCleaner().clean(frame)
        assert len(result) == 2  # duplicates are kept unless asked for
        assert result["b"].tolist() == ["x", "x"]
