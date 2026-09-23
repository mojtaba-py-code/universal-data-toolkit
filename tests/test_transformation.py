"""Transformations and the transformation chain."""

from __future__ import annotations

import pandas as pd
import pytest

from universal_data.core.exceptions import TransformationError
from universal_data.transformation.base import (
    Transformation,
    TransformationChain,
    TransformationRegistry,
    transformations,
)
from universal_data.transformation.operations import (
    Aggregate,
    ConvertTypes,
    CreateColumn,
    DropColumns,
    FilterRows,
    Join,
    LimitRows,
    MapValues,
    Melt,
    Pivot,
    RenameColumns,
    ReplaceValues,
    SelectColumns,
    SortRows,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "country": ["IR", "DE", "IR", "FR"],
            "quantity": [2, 1, 5, 3],
            "price": [10.0, 20.0, 5.0, 7.5],
        }
    )


class TestColumnOperations:
    def test_rename(self, frame: pd.DataFrame) -> None:
        result = RenameColumns({"country": "country_code"}).apply(frame)
        assert "country_code" in result.columns

    def test_rename_requires_existing_columns(self, frame: pd.DataFrame) -> None:
        with pytest.raises(TransformationError, match="not found"):
            RenameColumns({"nope": "x"}).apply(frame)

    def test_rename_needs_a_mapping(self) -> None:
        with pytest.raises(TransformationError, match="mapping"):
            RenameColumns()

    def test_rename_from_flat_config(self, frame: pd.DataFrame) -> None:
        transformation = RenameColumns.from_config({"country": "c"})
        assert "c" in transformation.apply(frame).columns

    def test_drop_columns(self, frame: pd.DataFrame) -> None:
        assert "price" not in DropColumns("price").apply(frame).columns

    def test_drop_missing_column_can_be_ignored(self, frame: pd.DataFrame) -> None:
        assert len(DropColumns(["nope"], ignore_missing=True).apply(frame).columns) == 4
        with pytest.raises(TransformationError):
            DropColumns(["nope"]).apply(frame)

    def test_select(self, frame: pd.DataFrame) -> None:
        assert list(SelectColumns(["id", "price"]).apply(frame).columns) == ["id", "price"]

    def test_select_from_list_config(self, frame: pd.DataFrame) -> None:
        assert list(SelectColumns.from_config(["id"]).apply(frame).columns) == ["id"]


class TestRowOperations:
    def test_filter(self, frame: pd.DataFrame) -> None:
        assert len(FilterRows("quantity > 2").apply(frame)) == 2

    def test_filter_on_empty_frame(self) -> None:
        empty = pd.DataFrame(columns=["quantity"])
        assert FilterRows("quantity > 2").apply(empty).empty

    def test_filter_describe(self) -> None:
        assert "quantity" in FilterRows("quantity > 2").describe()

    def test_sort(self, frame: pd.DataFrame) -> None:
        result = SortRows("price", ascending=False).apply(frame)
        assert result["price"].tolist() == [20.0, 10.0, 7.5, 5.0]

    def test_sort_requires_existing_column(self, frame: pd.DataFrame) -> None:
        with pytest.raises(TransformationError):
            SortRows("nope").apply(frame)

    def test_limit(self, frame: pd.DataFrame) -> None:
        assert len(LimitRows(2).apply(frame)) == 2

    def test_limit_rejects_zero(self) -> None:
        with pytest.raises(TransformationError, match="positive"):
            LimitRows(0)

    def test_limit_from_int_config(self, frame: pd.DataFrame) -> None:
        assert len(LimitRows.from_config(1).apply(frame)) == 1


class TestAggregation:
    def test_aggregate_flattens_names(self, frame: pd.DataFrame) -> None:
        result = Aggregate("country", {"price": ["sum", "mean"], "id": "count"}).apply(frame)
        assert set(result.columns) == {"country", "price_sum", "price_mean", "id_count"}
        assert len(result) == 3

    def test_aggregate_rejects_unknown_function(self) -> None:
        with pytest.raises(TransformationError, match="Unsupported aggregation"):
            Aggregate("country", {"price": ["eval"]})

    def test_aggregate_needs_aggregations(self) -> None:
        with pytest.raises(TransformationError, match="at least one"):
            Aggregate("country", {})

    def test_pivot(self, frame: pd.DataFrame) -> None:
        result = Pivot(index="country", columns="quantity", values="price").apply(frame)
        assert "country" in result.columns

    def test_pivot_rejects_unknown_aggfunc(self) -> None:
        with pytest.raises(TransformationError, match="Unsupported aggregation"):
            Pivot(index="a", columns="b", values="c", aggfunc="exec")

    def test_melt(self, frame: pd.DataFrame) -> None:
        result = Melt(id_vars="id", value_vars=["quantity", "price"]).apply(frame)
        assert set(result.columns) == {"id", "variable", "value"}
        assert len(result) == 8


class TestJoinAndMapping:
    def test_join(self, frame: pd.DataFrame) -> None:
        lookup = pd.DataFrame({"country": ["IR", "DE", "FR"], "region": ["ME", "EU", "EU"]})
        result = Join(lookup, on="country").apply(frame)
        assert "region" in result.columns
        assert result["region"].tolist() == ["ME", "EU", "ME", "EU"]

    def test_join_rejects_unknown_how(self, frame: pd.DataFrame) -> None:
        with pytest.raises(TransformationError, match="Unsupported join"):
            Join(frame, on="id", how="sideways")

    def test_join_rejects_non_frame(self) -> None:
        with pytest.raises(TransformationError, match="DataFrame or Dataset"):
            Join("not a frame")  # type: ignore[arg-type]

    def test_join_describe(self, frame: pd.DataFrame) -> None:
        assert "right_rows" in Join(frame, on="id").describe()

    def test_map_values_keeps_unmapped(self, frame: pd.DataFrame) -> None:
        result = MapValues("country", {"IR": "Iran"}).apply(frame)
        assert result["country"].tolist() == ["Iran", "DE", "Iran", "FR"]

    def test_map_values_with_default_and_target(self, frame: pd.DataFrame) -> None:
        result = MapValues("country", {"IR": "Iran"}, default="Other", target="name").apply(frame)
        assert result["name"].tolist() == ["Iran", "Other", "Iran", "Other"]

    def test_map_values_requires_the_column(self, frame: pd.DataFrame) -> None:
        with pytest.raises(TransformationError):
            MapValues("nope", {}).apply(frame)


class TestDerivedColumns:
    def test_create_column(self, frame: pd.DataFrame) -> None:
        result = CreateColumn("total", "quantity * price").apply(frame)
        assert result["total"].tolist() == [20.0, 20.0, 25.0, 22.5]

    def test_create_column_on_empty_frame(self) -> None:
        empty = pd.DataFrame(columns=["quantity", "price"])
        assert "total" in CreateColumn("total", "quantity * price").apply(empty).columns

    def test_create_column_needs_a_name(self) -> None:
        with pytest.raises(TransformationError, match="needs a column name"):
            CreateColumn("", "1")

    def test_create_column_describe(self) -> None:
        assert "total" in CreateColumn("total", "quantity").describe()

    def test_convert_types(self, frame: pd.DataFrame) -> None:
        result = ConvertTypes({"quantity": "float"}).apply(frame)
        assert str(result["quantity"].dtype) == "Float64"

    def test_convert_types_needs_a_mapping(self) -> None:
        with pytest.raises(TransformationError, match="mapping"):
            ConvertTypes()

    def test_replace_values(self, frame: pd.DataFrame) -> None:
        result = ReplaceValues("country", r"^IR$", "Iran").apply(frame)
        assert result["country"].tolist() == ["Iran", "DE", "Iran", "FR"]


class TestChainAndRegistry:
    def test_chain_applies_in_order(self, frame: pd.DataFrame) -> None:
        chain = TransformationChain(
            [FilterRows("quantity > 1"), SelectColumns(["id", "quantity"]), SortRows("quantity")]
        )
        result = chain.apply(frame)
        assert list(result.columns) == ["id", "quantity"]
        assert result["quantity"].tolist() == [2, 3, 5]
        assert len(chain) == 3
        assert "filter" in repr(chain)

    def test_chain_wraps_unexpected_errors(self, frame: pd.DataFrame) -> None:
        class Broken(Transformation):
            name = "broken"

            def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
                raise RuntimeError("kaboom")

        with pytest.raises(TransformationError, match="kaboom"):
            TransformationChain([Broken()]).apply(frame)

    def test_chain_from_list_config(self, frame: pd.DataFrame) -> None:
        chain = TransformationChain.from_config(
            [{"filter": "quantity > 1"}, {"select": ["id"]}]
        )
        assert list(chain.apply(frame).columns) == ["id"]

    def test_chain_from_mapping_config(self, frame: pd.DataFrame) -> None:
        chain = TransformationChain.from_config({"select": ["id"]})
        assert list(chain.apply(frame).columns) == ["id"]

    def test_chain_rejects_bad_entries(self) -> None:
        with pytest.raises(TransformationError, match="single-key mapping"):
            TransformationChain.from_config([{"a": 1, "b": 2}])
        with pytest.raises(TransformationError, match="list or a mapping"):
            TransformationChain.from_config("filter")

    def test_registry_lookup(self) -> None:
        assert "filter" in transformations
        assert transformations.get("filter") is FilterRows
        with pytest.raises(TransformationError, match="Unknown transformation"):
            transformations.get("nope")

    def test_registry_rejects_duplicates(self) -> None:
        registry = TransformationRegistry()

        class Custom(Transformation):
            name = "custom"

            def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
                return frame

        registry.register(Custom)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(Custom)
        assert registry.names() == ["custom"]

    def test_registry_rejects_unnamed(self) -> None:
        registry = TransformationRegistry()

        class Unnamed(Transformation):
            name = ""

            def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
                return frame

        with pytest.raises(ValueError, match="needs a name"):
            registry.register(Unnamed)

    def test_from_config_rejects_non_mapping(self) -> None:
        with pytest.raises(TransformationError, match="expects a mapping"):
            Aggregate.from_config("nope")
