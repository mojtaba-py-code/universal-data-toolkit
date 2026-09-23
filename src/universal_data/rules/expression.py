"""A safe, vectorised expression evaluator.

Business rules, row filters and calculated columns all need to evaluate a short
user supplied expression against a DataFrame.  ``eval``/``pandas.query`` are not
an option: both execute arbitrary Python, and ``query`` still reaches attributes
and builtins through ``@`` locals.

This module parses the expression with :mod:`ast` and walks an explicit
allow-list of node types.  Anything else - attribute access, subscripting,
lambdas, comprehensions, imports, walrus - is rejected before evaluation.

Evaluation is vectorised: names resolve to pandas Series, so a rule over a
million rows costs one pass per operator instead of a Python loop.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from universal_data.core.exceptions import RuleError

MAX_EXPRESSION_LENGTH = 2_000
MAX_NODES = 400
MAX_REGEX_LENGTH = 500
MAX_POW_EXPONENT = 8

_BACKTICK = re.compile(r"`([^`]+)`")
_SQL_IS_NOT_NULL = re.compile(r"(?i)\bIS\s+NOT\s+NULL\b")
_SQL_IS_NULL = re.compile(r"(?i)\bIS\s+NULL\b")
_SQL_AND = re.compile(r"\bAND\b")
_SQL_OR = re.compile(r"\bOR\b")
_SQL_NOT = re.compile(r"\bNOT\b(?!\s*=)")
_SQL_NEQ = re.compile(r"<>")

_EMAIL_PATTERN = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
_URL_PATTERN = r"https?://[^\s/$.?#].[^\s]*"

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.IfExp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
)


def _to_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series([value] * len(index), index=index)


def _text(value: Any, index: pd.Index) -> pd.Series:
    series = _to_series(value, index)
    return series.astype("string")


def _as_numeric(value: Any) -> Any:
    """Coerce a text column to numbers so it can be compared with a number."""
    if (
        isinstance(value, pd.Series)
        and not pd.api.types.is_numeric_dtype(value)
        and not pd.api.types.is_datetime64_any_dtype(value)
    ):
        return pd.to_numeric(value, errors="coerce")
    return value


def _align_operands(left: Any, right: Any) -> tuple[Any, Any]:
    """Make a numeric/text mismatch comparable instead of a TypeError.

    Real datasets carry numbers stored as text ("N/A" in an age column).  A rule
    like ``age >= 18`` must evaluate to False for those rows, not crash the run.
    """
    numeric_literal = (int, float)
    if isinstance(right, numeric_literal) and not isinstance(right, bool):
        left = _as_numeric(left)
    if isinstance(left, numeric_literal) and not isinstance(left, bool):
        right = _as_numeric(right)
    if (
        isinstance(left, pd.Series)
        and isinstance(right, pd.Series)
        and pd.api.types.is_numeric_dtype(left) != pd.api.types.is_numeric_dtype(right)
    ):
        left, right = _as_numeric(left), _as_numeric(right)
    return left, right


def _compile_pattern(pattern: Any) -> str:
    if not isinstance(pattern, str):
        raise RuleError("Regular expression pattern must be a literal string")
    if len(pattern) > MAX_REGEX_LENGTH:
        raise RuleError("Regular expression is too long", length=len(pattern))
    try:
        re.compile(pattern)
    except re.error as exc:
        raise RuleError(f"Invalid regular expression: {exc}", pattern=pattern) from exc
    return pattern


def _fn_is_null(value: Any, index: pd.Index) -> pd.Series:
    return _to_series(value, index).isna()


def _fn_len(value: Any, index: pd.Index) -> pd.Series:
    return _text(value, index).str.len()


def _fn_matches(value: Any, pattern: Any, index: pd.Index) -> pd.Series:
    return _text(value, index).str.fullmatch(_compile_pattern(pattern), na=False).fillna(False)


def _fn_contains(value: Any, needle: Any, index: pd.Index) -> pd.Series:
    if not isinstance(needle, str):
        raise RuleError("contains() expects a literal string")
    return _text(value, index).str.contains(needle, regex=False, na=False).fillna(False)


def _fn_between(value: Any, low: Any, high: Any, index: pd.Index) -> pd.Series:
    series = pd.to_numeric(_to_series(value, index), errors="coerce")
    return (series >= low) & (series <= high)


def _fn_coalesce(first: Any, second: Any, index: pd.Index) -> pd.Series:
    series = _to_series(first, index)
    return series.where(series.notna(), _to_series(second, index))


def _fn_to_datetime(value: Any, index: pd.Index) -> pd.Series:
    series = _to_series(value, index)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _fn_days_since(value: Any, index: pd.Index) -> pd.Series:
    parsed = _fn_to_datetime(value, index)
    now = pd.Timestamp.now(tz=parsed.dt.tz) if getattr(parsed.dt, "tz", None) else pd.Timestamp.now()
    return (now - parsed).dt.total_seconds() / 86_400


# Every function receives the frame index as the last positional argument so it
# can broadcast scalars to the right length.
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "is_null": _fn_is_null,
    "not_null": lambda v, index: ~_fn_is_null(v, index),
    "len": _fn_len,
    "lower": lambda v, index: _text(v, index).str.lower(),
    "upper": lambda v, index: _text(v, index).str.upper(),
    "strip": lambda v, index: _text(v, index).str.strip(),
    "abs": lambda v, index: np.abs(_to_series(v, index)),
    "round": lambda v, digits, index: _to_series(v, index).round(int(digits)),
    "matches": _fn_matches,
    "contains": _fn_contains,
    "startswith": lambda v, p, index: _text(v, index).str.startswith(str(p), na=False),
    "endswith": lambda v, p, index: _text(v, index).str.endswith(str(p), na=False),
    "between": _fn_between,
    "coalesce": _fn_coalesce,
    "is_email": lambda v, index: _fn_matches(v, _EMAIL_PATTERN, index),
    "is_url": lambda v, index: _text(v, index).str.match(_URL_PATTERN, na=False).fillna(False),
    "to_number": lambda v, index: pd.to_numeric(_to_series(v, index), errors="coerce"),
    "to_datetime": _fn_to_datetime,
    "year": lambda v, index: _fn_to_datetime(v, index).dt.year,
    "month": lambda v, index: _fn_to_datetime(v, index).dt.month,
    "day": lambda v, index: _fn_to_datetime(v, index).dt.day,
    "days_since": _fn_days_since,
    "least": lambda a, b, index: np.minimum(_to_series(a, index), _to_series(b, index)),
    "greatest": lambda a, b, index: np.maximum(_to_series(a, index), _to_series(b, index)),
}

_FUNCTION_ARITY: dict[str, tuple[int, int]] = {
    "round": (1, 2),
    "matches": (2, 2),
    "contains": (2, 2),
    "startswith": (2, 2),
    "endswith": (2, 2),
    "between": (3, 3),
    "coalesce": (2, 2),
    "least": (2, 2),
    "greatest": (2, 2),
}

_FUNCTION_DEFAULTS: dict[str, list[Any]] = {"round": [0]}

_BINARY_OPS: dict[type[ast.AST], Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}

_COMPARE_OPS: dict[type[ast.AST], Callable[[Any, Any], Any]] = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}


def _normalize_sql_syntax(source: str) -> str:
    """Accept the small SQL dialect people naturally write in YAML rules."""
    result = _SQL_IS_NOT_NULL.sub("!= @@NULL@@", source)
    result = _SQL_IS_NULL.sub("== @@NULL@@", result)
    result = _SQL_NEQ.sub("!=", result)
    result = _SQL_AND.sub("and", result)
    result = _SQL_OR.sub("or", result)
    return _SQL_NOT.sub("not", result)


class SafeExpression:
    """A parsed, validated expression that can be evaluated against a frame."""

    __slots__ = ("source", "_tree", "_aliases", "_names")

    def __init__(self, source: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise RuleError("Expression must be a non-empty string")
        if len(source) > MAX_EXPRESSION_LENGTH:
            raise RuleError("Expression is too long", length=len(source))

        self.source = source
        prepared, self._aliases = self._prepare(source)
        try:
            tree = ast.parse(prepared, mode="eval")
        except SyntaxError as exc:
            raise RuleError(f"Could not parse expression: {exc.msg}", expression=source) from exc

        for node_count, node in enumerate(ast.walk(tree), start=1):
            if node_count > MAX_NODES:
                raise RuleError("Expression is too complex", expression=source)
            if not isinstance(node, _ALLOWED_NODES):
                raise RuleError(
                    f"Expression uses a construct that is not allowed: {type(node).__name__}",
                    expression=source,
                )
            if isinstance(node, ast.Call):
                self._validate_call(node)

        self._tree = tree
        self._names = frozenset(
            self._aliases.get(n.id, n.id)
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id not in _FUNCTIONS and n.id != "__NULL__"
        )

    @staticmethod
    def _prepare(source: str) -> tuple[str, dict[str, str]]:
        """Replace backtick-quoted column names with safe identifiers."""
        aliases: dict[str, str] = {}

        def substitute(match: re.Match[str]) -> str:
            name = match.group(1)
            alias = f"_col_{len(aliases)}"
            aliases[alias] = name
            return alias

        prepared = _BACKTICK.sub(substitute, source)
        prepared = _normalize_sql_syntax(prepared)
        return prepared.replace("@@NULL@@", "__NULL__"), aliases

    @staticmethod
    def _validate_call(node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise RuleError("Only direct calls to built-in rule functions are allowed")
        name = node.func.id
        if name not in _FUNCTIONS:
            raise RuleError(
                f"Unknown function '{name}'",
                available=sorted(_FUNCTIONS),
            )
        if node.keywords:
            raise RuleError(f"Function '{name}' does not accept keyword arguments")
        minimum, maximum = _FUNCTION_ARITY.get(name, (1, 1))
        if not minimum <= len(node.args) <= maximum:
            raise RuleError(
                f"Function '{name}' expects between {minimum} and {maximum} arguments",
                given=len(node.args),
            )

    @property
    def columns(self) -> frozenset[str]:
        """Column names referenced by the expression."""
        return self._names

    def missing_columns(self, frame: pd.DataFrame) -> list[str]:
        return sorted(name for name in self._names if name not in frame.columns)

    def evaluate(self, frame: pd.DataFrame) -> Any:
        """Evaluate the expression; returns a Series for column expressions."""
        missing = self.missing_columns(frame)
        if missing:
            raise RuleError(
                "Expression references columns that are not in the dataset",
                expression=self.source,
                missing=missing,
            )
        return self._eval(self._tree.body, frame)

    def evaluate_mask(self, frame: pd.DataFrame) -> pd.Series:
        """Evaluate to a boolean mask; nulls count as ``False``."""
        result = self.evaluate(frame)
        series = _to_series(result, frame.index)
        if series.dtype == object or str(series.dtype) == "string":
            series = series.map(lambda v: bool(v) if v is not None and v is not pd.NA else False)
        return series.fillna(False).astype(bool)

    def _eval(self, node: ast.AST, frame: pd.DataFrame) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id == "__NULL__":
                return np.nan
            column = self._aliases.get(node.id, node.id)
            if column in frame.columns:
                return frame[column]
            raise RuleError(
                "Expression references an unknown column",
                column=column,
                expression=self.source,
            )

        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return [self._eval(element, frame) for element in node.elts]

        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand, frame)
            if isinstance(node.op, ast.Not):
                if isinstance(operand, pd.Series):
                    return ~operand.fillna(False).astype(bool)
                return not operand
            if isinstance(node.op, ast.USub):
                return -operand
            return +operand

        if isinstance(node, ast.BoolOp):
            values = [self._eval(value, frame) for value in node.values]
            combine = (lambda a, b: a & b) if isinstance(node.op, ast.And) else (lambda a, b: a | b)
            result = self._as_bool(values[0], frame)
            for value in values[1:]:
                result = combine(result, self._as_bool(value, frame))
            return result

        if isinstance(node, ast.BinOp):
            left, right = _align_operands(
                self._eval(node.left, frame), self._eval(node.right, frame)
            )
            if isinstance(node.op, ast.Pow):
                if isinstance(right, (int, float)) and abs(right) > MAX_POW_EXPONENT:
                    raise RuleError("Exponent is too large", exponent=right)
                return left**right
            handler = _BINARY_OPS.get(type(node.op))
            if handler is None:  # pragma: no cover - guarded by the node allow-list
                raise RuleError(f"Unsupported operator: {type(node.op).__name__}")
            return handler(left, right)

        if isinstance(node, ast.Compare):
            return self._eval_compare(node, frame)

        if isinstance(node, ast.IfExp):
            test = self._as_bool(self._eval(node.test, frame), frame)
            body = self._eval(node.body, frame)
            orelse = self._eval(node.orelse, frame)
            if isinstance(test, pd.Series):
                return pd.Series(
                    np.where(test, _to_series(body, frame.index), _to_series(orelse, frame.index)),
                    index=frame.index,
                )
            return body if test else orelse

        if isinstance(node, ast.Call):
            # _validate_call already proved the callee is a plain Name.
            assert isinstance(node.func, ast.Name)
            name = node.func.id
            args = [self._eval(arg, frame) for arg in node.args]
            args.extend(_FUNCTION_DEFAULTS.get(name, [])[len(args) - 1 :])
            return _FUNCTIONS[name](*args, frame.index)

        raise RuleError(f"Unsupported expression node: {type(node).__name__}")

    def _eval_compare(self, node: ast.Compare, frame: pd.DataFrame) -> Any:
        left = self._eval(node.left, frame)
        result: Any = None
        for operator, comparator_node in zip(node.ops, node.comparators, strict=True):
            right = self._eval(comparator_node, frame)
            if isinstance(operator, (ast.In, ast.NotIn)):
                if not isinstance(right, list):
                    raise RuleError("'in' expects a list of literal values")
                series = _to_series(left, frame.index)
                comparison = series.isin(right)
                if isinstance(operator, ast.NotIn):
                    comparison = ~comparison
            elif isinstance(right, float) and np.isnan(right):
                # `col IS NULL` and `col IS NOT NULL` normalise to == / != NULL.
                series = _to_series(left, frame.index)
                comparison = series.isna() if isinstance(operator, ast.Eq) else series.notna()
            else:
                handler = _COMPARE_OPS.get(type(operator))
                if handler is None:  # pragma: no cover - guarded by the allow-list
                    raise RuleError(f"Unsupported comparison: {type(operator).__name__}")
                aligned_left, aligned_right = _align_operands(left, right)
                comparison = handler(aligned_left, aligned_right)
            result = comparison if result is None else (result & comparison)
            left = right
        return result

    @staticmethod
    def _as_bool(value: Any, frame: pd.DataFrame) -> Any:
        if isinstance(value, pd.Series):
            return value.fillna(False).astype(bool)
        return bool(value)

    def __repr__(self) -> str:
        return f"SafeExpression({self.source!r})"


def compile_expression(source: str | SafeExpression) -> SafeExpression:
    """Convenience wrapper so callers can accept either form."""
    return source if isinstance(source, SafeExpression) else SafeExpression(source)


def available_functions() -> list[str]:
    return sorted(_FUNCTIONS)
