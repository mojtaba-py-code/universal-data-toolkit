"""Built-in transformations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import pandas as pd

from universal_data.core.exceptions import TransformationError
from universal_data.rules.expression import SafeExpression, compile_expression
from universal_data.transformation.base import Transformation, register_transformation

# Aggregations are restricted to this set: a name coming from YAML must never
# reach ``DataFrame.agg`` unchecked, because pandas happily accepts callables.
ALLOWED_AGGREGATIONS = frozenset(
    {
        "sum", "mean", "median", "min", "max", "count", "size",
        "nunique", "std", "var", "first", "last", "prod",
    }
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, operation: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TransformationError(
            "Column(s) not found in the dataset", operation=operation, missing=missing
        )


@register_transformation
class RenameColumns(Transformation):
    name: ClassVar[str] = "rename"

    def __init__(self, mapping: dict[str, str] | None = None, **extra: str) -> None:
        self.mapping = {**(mapping or {}), **extra}
        if not self.mapping:
            raise TransformationError("rename needs a mapping of old -> new names")

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, list(self.mapping), operation=self.name)
        return frame.rename(columns=self.mapping)

    @classmethod
    def from_config(cls, config: Any) -> RenameColumns:
        if isinstance(config, dict) and "mapping" not in config:
            return cls(mapping=dict(config))
        return super().from_config(config)  # type: ignore[return-value]


@register_transformation
class DropColumns(Transformation):
    name: ClassVar[str] = "drop_columns"

    def __init__(self, columns: Sequence[str] | str, ignore_missing: bool = False) -> None:
        self.columns = [columns] if isinstance(columns, str) else list(columns)
        self.ignore_missing = ignore_missing

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.ignore_missing:
            _require_columns(frame, self.columns, operation=self.name)
        return frame.drop(columns=self.columns, errors="ignore")

    @classmethod
    def from_config(cls, config: Any) -> DropColumns:
        if isinstance(config, (list, str)):
            return cls(columns=config)
        return super().from_config(config)  # type: ignore[return-value]


@register_transformation
class SelectColumns(Transformation):
    name: ClassVar[str] = "select"

    def __init__(self, columns: Sequence[str] | str) -> None:
        self.columns = [columns] if isinstance(columns, str) else list(columns)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, self.columns, operation=self.name)
        return frame[self.columns]

    @classmethod
    def from_config(cls, config: Any) -> SelectColumns:
        if isinstance(config, (list, str)):
            return cls(columns=config)
        return super().from_config(config)  # type: ignore[return-value]


@register_transformation
class FilterRows(Transformation):
    """Keeps rows for which *condition* evaluates truthy."""

    name: ClassVar[str] = "filter"

    def __init__(self, condition: str | SafeExpression) -> None:
        self.condition = compile_expression(condition)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return frame.loc[self.condition.evaluate_mask(frame)]

    @classmethod
    def from_config(cls, config: Any) -> FilterRows:
        if isinstance(config, str):
            return cls(condition=config)
        return super().from_config(config)  # type: ignore[return-value]

    def describe(self) -> str:
        return f"filter({self.condition.source!r})"


@register_transformation
class SortRows(Transformation):
    name: ClassVar[str] = "sort"

    def __init__(
        self,
        by: Sequence[str] | str,
        ascending: bool | Sequence[bool] = True,
        na_position: str = "last",
    ) -> None:
        self.by = [by] if isinstance(by, str) else list(by)
        self.ascending = ascending
        self.na_position = na_position

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, self.by, operation=self.name)
        return frame.sort_values(
            by=self.by, ascending=self.ascending, na_position=self.na_position
        )

    @classmethod
    def from_config(cls, config: Any) -> SortRows:
        if isinstance(config, (list, str)):
            return cls(by=config)
        return super().from_config(config)  # type: ignore[return-value]


@register_transformation
class Aggregate(Transformation):
    """Group rows and aggregate: ``group_by: [country], aggregations: {salary: [mean, max]}``."""

    name: ClassVar[str] = "aggregate"

    def __init__(
        self,
        group_by: Sequence[str] | str,
        aggregations: dict[str, Any],
        flatten_names: bool = True,
    ) -> None:
        self.group_by = [group_by] if isinstance(group_by, str) else list(group_by)
        self.aggregations = self._validate(aggregations)
        self.flatten_names = flatten_names

    @staticmethod
    def _validate(aggregations: dict[str, Any]) -> dict[str, list[str]]:
        if not aggregations:
            raise TransformationError("aggregate needs at least one aggregation")
        cleaned: dict[str, list[str]] = {}
        for column, functions in aggregations.items():
            names = [functions] if isinstance(functions, str) else list(functions)
            unknown = [f for f in names if f not in ALLOWED_AGGREGATIONS]
            if unknown:
                raise TransformationError(
                    "Unsupported aggregation function",
                    functions=unknown,
                    allowed=sorted(ALLOWED_AGGREGATIONS),
                )
            cleaned[column] = names
        return cleaned

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, [*self.group_by, *self.aggregations], operation=self.name)
        grouped = frame.groupby(self.group_by, dropna=False, observed=True).agg(self.aggregations)
        if self.flatten_names and isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [f"{column}_{function}" for column, function in grouped.columns]
        return grouped.reset_index()


@register_transformation
class Pivot(Transformation):
    name: ClassVar[str] = "pivot"

    def __init__(
        self,
        index: Sequence[str] | str,
        columns: str,
        values: str,
        aggfunc: str = "sum",
    ) -> None:
        if aggfunc not in ALLOWED_AGGREGATIONS:
            raise TransformationError(
                "Unsupported aggregation function",
                function=aggfunc,
                allowed=sorted(ALLOWED_AGGREGATIONS),
            )
        self.index = [index] if isinstance(index, str) else list(index)
        self.columns = columns
        self.values = values
        self.aggfunc = aggfunc

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(
            frame, [*self.index, self.columns, self.values], operation=self.name
        )
        pivoted = frame.pivot_table(
            index=self.index,
            columns=self.columns,
            values=self.values,
            aggfunc=self.aggfunc,
            observed=True,
        )
        pivoted.columns = [str(column) for column in pivoted.columns]
        return pivoted.reset_index()


@register_transformation
class Melt(Transformation):
    name: ClassVar[str] = "melt"

    def __init__(
        self,
        id_vars: Sequence[str] | str,
        value_vars: Sequence[str] | None = None,
        var_name: str = "variable",
        value_name: str = "value",
    ) -> None:
        self.id_vars = [id_vars] if isinstance(id_vars, str) else list(id_vars)
        self.value_vars = list(value_vars) if value_vars else None
        self.var_name = var_name
        self.value_name = value_name

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, self.id_vars, operation=self.name)
        return frame.melt(
            id_vars=self.id_vars,
            value_vars=self.value_vars,
            var_name=self.var_name,
            value_name=self.value_name,
        )


@register_transformation
class Join(Transformation):
    """Joins against another frame supplied by the caller (not by config)."""

    name: ClassVar[str] = "join"

    def __init__(
        self,
        right: pd.DataFrame,
        on: Sequence[str] | str | None = None,
        how: str = "left",
        left_on: Sequence[str] | str | None = None,
        right_on: Sequence[str] | str | None = None,
        suffixes: tuple[str, str] = ("", "_right"),
    ) -> None:
        if how not in ("left", "right", "inner", "outer", "cross"):
            raise TransformationError("Unsupported join type", how=how)
        if hasattr(right, "frame"):  # accept a Dataset without importing it
            right = right.frame
        if not isinstance(right, pd.DataFrame):
            raise TransformationError("join needs a DataFrame or Dataset on the right side")
        self.right = right
        self.on = on
        self.how = how
        self.left_on = left_on
        self.right_on = right_on
        self.suffixes = suffixes

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.merge(
            self.right,
            how=self.how,
            on=self.on,
            left_on=self.left_on,
            right_on=self.right_on,
            suffixes=self.suffixes,
        )

    def describe(self) -> str:
        return f"join(how={self.how!r}, on={self.on!r}, right_rows={len(self.right)})"


@register_transformation
class MapValues(Transformation):
    name: ClassVar[str] = "map_values"

    def __init__(
        self, column: str, mapping: dict[Any, Any], default: Any = None, target: str | None = None
    ) -> None:
        self.column = column
        self.mapping = dict(mapping)
        self.default = default
        self.target = target or column

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, [self.column], operation=self.name)
        result = frame.copy()
        mapped = result[self.column].map(self.mapping)
        if self.default is not None:
            mapped = mapped.fillna(self.default)
        else:
            mapped = mapped.where(mapped.notna(), result[self.column])
        result[self.target] = mapped
        return result


@register_transformation
class CreateColumn(Transformation):
    """Adds a column computed from a safe expression."""

    name: ClassVar[str] = "create_column"

    def __init__(self, column: str, expression: str | SafeExpression) -> None:
        if not column:
            raise TransformationError("create_column needs a column name")
        self.column = column
        self.expression = compile_expression(expression)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if frame.empty:
            result[self.column] = pd.Series(dtype="object")
            return result
        value = self.expression.evaluate(frame)
        result[self.column] = value
        return result

    def describe(self) -> str:
        return f"create_column({self.column!r}, {self.expression.source!r})"


@register_transformation
class ConvertTypes(Transformation):
    name: ClassVar[str] = "convert_types"

    def __init__(self, mapping: dict[str, str] | None = None, **extra: str) -> None:
        self.mapping = {**(mapping or {}), **extra}
        if not self.mapping:
            raise TransformationError("convert_types needs a column -> type mapping")

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        from universal_data.cleaning.operations import convert_types

        converted, _ = convert_types(frame, self.mapping)
        return converted

    @classmethod
    def from_config(cls, config: Any) -> ConvertTypes:
        if isinstance(config, dict) and "mapping" not in config:
            return cls(mapping=dict(config))
        return super().from_config(config)  # type: ignore[return-value]


@register_transformation
class ReplaceValues(Transformation):
    name: ClassVar[str] = "replace"

    def __init__(
        self,
        column: str,
        pattern: str,
        replacement: str = "",
        regex: bool = True,
    ) -> None:
        self.column = column
        self.pattern = pattern
        self.replacement = replacement
        self.regex = regex

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, [self.column], operation=self.name)
        result = frame.copy()
        result[self.column] = (
            result[self.column]
            .astype("string")
            .str.replace(self.pattern, self.replacement, regex=self.regex)
        )
        return result


@register_transformation
class LimitRows(Transformation):
    name: ClassVar[str] = "limit"

    def __init__(self, rows: int) -> None:
        if rows <= 0:
            raise TransformationError("limit must be positive", rows=rows)
        self.rows = int(rows)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.head(self.rows)

    @classmethod
    def from_config(cls, config: Any) -> LimitRows:
        if isinstance(config, int):
            return cls(rows=config)
        return super().from_config(config)  # type: ignore[return-value]
