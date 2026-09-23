"""Tests for the safe expression engine."""

from __future__ import annotations

import pandas as pd
import pytest

from universal_data.core.exceptions import RuleError
from universal_data.rules.expression import SafeExpression, available_functions


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [17, 25, 40, None],
            "salary": [1000.0, -20.0, 5000.0, 2000.0],
            "email": ["a@b.com", "bad", None, "c@d.org"],
            "country": ["IR", "DE", "FR", "IR"],
            "joined": ["2024-01-01", "2023-06-15", "2020-01-01", None],
            "first name": ["Ali", "Sara", "Reza", "Nina"],
        }
    )


class TestSandbox:
    @pytest.mark.parametrize(
        "source",
        [
            "__import__('os').system('echo pwned')",
            "().__class__.__bases__",
            "open('/etc/passwd').read()",
            "[x for x in range(10)]",
            "lambda: 1",
            "(a := 3)",
            "exec('x=1')",
            "globals()",
            "age.__class__",
            "{'a': 1}['a']",
            "f'{age}'",
        ],
    )
    def test_dangerous_expressions_are_rejected(self, source: str) -> None:
        with pytest.raises(RuleError):
            SafeExpression(source)

    def test_unknown_function_is_rejected(self) -> None:
        with pytest.raises(RuleError, match="Unknown function"):
            SafeExpression("eval(age)")

    def test_keyword_arguments_are_rejected(self) -> None:
        with pytest.raises(RuleError, match="keyword arguments"):
            SafeExpression("is_null(x=age)")

    def test_wrong_arity_is_rejected(self) -> None:
        with pytest.raises(RuleError, match="expects between"):
            SafeExpression("matches(email)")

    def test_empty_expression_is_rejected(self) -> None:
        with pytest.raises(RuleError, match="non-empty"):
            SafeExpression("   ")

    def test_overlong_expression_is_rejected(self) -> None:
        with pytest.raises(RuleError, match="too long"):
            SafeExpression("age + " * 400 + "1")

    def test_complex_expression_is_rejected(self) -> None:
        with pytest.raises(RuleError, match="too complex|too long"):
            SafeExpression(" + ".join(["age"] * 300))

    def test_huge_exponent_is_rejected(self, frame: pd.DataFrame) -> None:
        expression = SafeExpression("age ** 99")
        with pytest.raises(RuleError, match="Exponent"):
            expression.evaluate(frame)

    def test_invalid_regex_is_rejected(self) -> None:
        with pytest.raises(RuleError, match="Invalid regular expression"):
            SafeExpression("matches(email, '([')").evaluate(pd.DataFrame({"email": ["a"]}))

    def test_syntax_error_is_wrapped(self) -> None:
        with pytest.raises(RuleError, match="Could not parse"):
            SafeExpression("age >=")


class TestEvaluation:
    def test_comparison_mask(self, frame: pd.DataFrame) -> None:
        mask = SafeExpression("age >= 18").evaluate_mask(frame)
        assert mask.tolist() == [False, True, True, False]

    def test_boolean_operators(self, frame: pd.DataFrame) -> None:
        mask = SafeExpression("age >= 18 AND salary >= 0").evaluate_mask(frame)
        assert mask.tolist() == [False, False, True, False]

    def test_or_and_not(self, frame: pd.DataFrame) -> None:
        mask = SafeExpression("NOT (country == 'IR') OR salary > 4000").evaluate_mask(frame)
        assert mask.tolist() == [False, True, True, False]

    def test_is_null_sugar(self, frame: pd.DataFrame) -> None:
        assert SafeExpression("email IS NULL").evaluate_mask(frame).tolist() == [
            False, False, True, False
        ]
        assert SafeExpression("email IS NOT NULL").evaluate_mask(frame).tolist() == [
            True, True, False, True
        ]

    def test_in_operator(self, frame: pd.DataFrame) -> None:
        mask = SafeExpression("country in ['IR', 'FR']").evaluate_mask(frame)
        assert mask.tolist() == [True, False, True, True]

    def test_not_in_operator(self, frame: pd.DataFrame) -> None:
        mask = SafeExpression("country not in ['IR']").evaluate_mask(frame)
        assert mask.tolist() == [False, True, True, False]

    def test_chained_comparison(self, frame: pd.DataFrame) -> None:
        mask = SafeExpression("18 <= age <= 30").evaluate_mask(frame)
        assert mask.tolist() == [False, True, False, False]

    def test_arithmetic_expression(self, frame: pd.DataFrame) -> None:
        values = SafeExpression("salary * 2 + 1").evaluate(frame)
        assert values.tolist() == [2001.0, -39.0, 10001.0, 4001.0]

    def test_conditional_expression(self, frame: pd.DataFrame) -> None:
        values = SafeExpression("'senior' if age >= 30 else 'junior'").evaluate(frame)
        assert values.tolist() == ["junior", "junior", "senior", "junior"]

    def test_is_email_function(self, frame: pd.DataFrame) -> None:
        mask = SafeExpression("is_email(email)").evaluate_mask(frame)
        assert mask.tolist() == [True, False, False, True]

    def test_string_functions(self, frame: pd.DataFrame) -> None:
        assert SafeExpression("lower(country)").evaluate(frame).tolist() == [
            "ir", "de", "fr", "ir"
        ]
        assert SafeExpression("len(country) == 2").evaluate_mask(frame).all()
        assert SafeExpression("startswith(email, 'a')").evaluate_mask(frame).tolist() == [
            True, False, False, False
        ]

    def test_between_and_coalesce(self, frame: pd.DataFrame) -> None:
        assert SafeExpression("between(age, 18, 30)").evaluate_mask(frame).tolist() == [
            False, True, False, False
        ]
        assert SafeExpression("coalesce(age, 0) >= 0").evaluate_mask(frame).all()

    def test_date_functions(self, frame: pd.DataFrame) -> None:
        years = SafeExpression("year(joined)").evaluate(frame)
        assert years.dropna().astype(int).tolist() == [2024, 2023, 2020]

    def test_round_with_default_argument(self, frame: pd.DataFrame) -> None:
        assert SafeExpression("round(salary)").evaluate(frame).tolist() == [
            1000.0, -20.0, 5000.0, 2000.0
        ]

    def test_backtick_column_names(self, frame: pd.DataFrame) -> None:
        mask = SafeExpression("`first name` == 'Ali'").evaluate_mask(frame)
        assert mask.tolist() == [True, False, False, False]

    def test_numeric_text_column_is_coerced(self) -> None:
        frame = pd.DataFrame({"age": ["21", "N/A", "17"]})
        mask = SafeExpression("age >= 18").evaluate_mask(frame)
        assert mask.tolist() == [True, False, False]

    def test_missing_column_is_reported(self, frame: pd.DataFrame) -> None:
        expression = SafeExpression("unknown_column > 1")
        assert expression.missing_columns(frame) == ["unknown_column"]
        with pytest.raises(RuleError, match="columns that are not in the dataset"):
            expression.evaluate(frame)

    def test_columns_property(self) -> None:
        expression = SafeExpression("age >= 18 AND salary > 0")
        assert expression.columns == frozenset({"age", "salary"})

    def test_repr_shows_source(self) -> None:
        assert "age" in repr(SafeExpression("age > 1"))

    def test_available_functions_are_listed(self) -> None:
        names = available_functions()
        assert "is_email" in names
        assert "is_null" in names
