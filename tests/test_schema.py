"""Schema model, detection and persistence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from universal_data.core.exceptions import SchemaError
from universal_data.core.types import FieldType
from universal_data.schema.detect import detect_column_type, detect_schema, looks_like_pii
from universal_data.schema.loader import load_schema, save_schema
from universal_data.schema.model import ColumnSchema, Schema


class TestColumnSchema:
    def test_requires_a_name(self) -> None:
        with pytest.raises(SchemaError, match="needs a name"):
            ColumnSchema(name="")

    def test_accepts_type_as_string(self) -> None:
        assert ColumnSchema(name="a", type="email").type is FieldType.EMAIL  # type: ignore[arg-type]

    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(SchemaError, match="Unknown column type"):
            ColumnSchema(name="a", type="quaternion")  # type: ignore[arg-type]

    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(SchemaError, match="min is greater than max"):
            ColumnSchema(name="a", type=FieldType.INTEGER, min=10, max=1)

    def test_from_dict_shorthand(self) -> None:
        assert ColumnSchema.from_dict("age", "integer").type is FieldType.INTEGER

    def test_from_dict_rejects_unknown_keys(self) -> None:
        with pytest.raises(SchemaError, match="Unknown schema keys"):
            ColumnSchema.from_dict("age", {"type": "integer", "maximum": 10})

    def test_to_dict_only_keeps_non_defaults(self) -> None:
        data = ColumnSchema(name="a", type=FieldType.INTEGER, min=1, pii=True).to_dict()
        assert data == {"type": "integer", "min": 1, "pii": True}


class TestSchema:
    def test_from_dict_with_schema_key(self) -> None:
        schema = Schema.from_dict(
            {"name": "c", "schema": {"id": {"type": "integer"}, "email": "email"}}
        )
        assert schema.column_names == ["id", "email"]
        assert schema.get("email").type is FieldType.EMAIL

    def test_from_dict_with_bare_columns(self) -> None:
        schema = Schema.from_dict({"id": "integer"})
        assert "id" in schema

    def test_from_dict_requires_columns(self) -> None:
        with pytest.raises(SchemaError, match="does not declare any column"):
            Schema.from_dict({"name": "empty", "schema": {}})

    def test_primary_key_must_exist(self) -> None:
        with pytest.raises(SchemaError, match="Primary key column"):
            Schema.from_dict({"schema": {"id": "integer"}, "primary_key": "missing"})

    def test_primary_key_accepts_a_string(self) -> None:
        schema = Schema.from_dict({"schema": {"id": "integer"}, "primary_key": "id"})
        assert schema.primary_key == ["id"]

    def test_round_trip_dict(self) -> None:
        original = Schema.from_dict(
            {
                "name": "c",
                "strict": True,
                "primary_key": ["id"],
                "description": "test",
                "schema": {"id": {"type": "integer", "unique": True}},
            }
        )
        restored = Schema.from_dict(original.to_dict())
        assert restored.name == "c"
        assert restored.strict is True
        assert restored.primary_key == ["id"]

    def test_helpers(self) -> None:
        schema = Schema.from_columns(
            [
                ColumnSchema(name="id", type=FieldType.INTEGER),
                ColumnSchema(name="email", type=FieldType.EMAIL, required=False, pii=True),
            ]
        )
        assert schema.required_columns == ["id"]
        assert schema.pii_columns == ["email"]
        assert len(schema) == 2
        assert [column.name for column in schema] == ["id", "email"]
        assert "email" in schema.describe()

    def test_add_and_get(self) -> None:
        schema = Schema(name="s")
        schema.add(ColumnSchema(name="x"))
        assert schema.get("x") is not None
        assert schema.get("missing") is None


class TestDetection:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([1, 2, 3], FieldType.INTEGER),
            ([1.5, 2.5], FieldType.FLOAT),
            ([True, False], FieldType.BOOLEAN),
            (["a@b.com", "c@d.org"], FieldType.EMAIL),
            (["https://a.test/x", "https://b.test/y"], FieldType.URL),
            (["2024-01-01", "2024-02-01"], FieldType.DATETIME),
            (["yes", "no", "yes"], FieldType.BOOLEAN),
            (["a", "b", "a", "b", "a", "b"], FieldType.CATEGORICAL),
            ([None, None], FieldType.UNKNOWN),
            (
                ["550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440001"],
                FieldType.UUID,
            ),
        ],
    )
    def test_detect_column_type(self, values: list[object], expected: FieldType) -> None:
        assert detect_column_type(pd.Series(values), "col") is expected

    def test_numeric_text_is_detected(self) -> None:
        assert detect_column_type(pd.Series(["1,000", "2,500"]), "amount") is FieldType.INTEGER

    def test_phone_column_uses_the_name(self) -> None:
        series = pd.Series(["+491701234567", "+989121234567"])
        assert detect_column_type(series, "phone") is FieldType.PHONE
        assert detect_column_type(series, "code") is FieldType.INTEGER

    def test_float_of_whole_numbers_is_integer(self) -> None:
        assert detect_column_type(pd.Series([1.0, 2.0, 3.0]), "n") is FieldType.INTEGER

    def test_high_cardinality_text_is_string(self) -> None:
        series = pd.Series([f"free text value number {i}" for i in range(100)])
        assert detect_column_type(series, "notes") is FieldType.STRING

    @pytest.mark.parametrize(
        "name", ["email", "phone_number", "national_id", "date_of_birth", "street_address"]
    )
    def test_pii_detected_by_name(self, name: str) -> None:
        assert looks_like_pii(name, FieldType.STRING)

    def test_pii_detected_by_type(self) -> None:
        assert looks_like_pii("contact", FieldType.EMAIL)
        assert not looks_like_pii("total", FieldType.FLOAT)

    def test_detect_schema(self, customers_frame: pd.DataFrame) -> None:
        schema = detect_schema(customers_frame, name="customers")
        assert schema.name == "customers"
        assert schema.get("customer_id").type is FieldType.INTEGER
        assert "email" in schema.pii_columns

    def test_detect_schema_with_constraints(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [20, 30, 40, 25, 35, 45],
                "grade": ["a", "b", "a", "b", "a", "b"],
                "note": ["short", "a bit longer", "x", "y", "z", "w"],
            }
        )
        schema = detect_schema(frame, infer_constraints=True)
        assert schema.get("age").min == 20
        assert schema.get("age").max == 45
        assert schema.get("grade").allowed == ["a", "b"]
        assert schema.get("note").max_length == 12

    def test_unique_integer_id_becomes_primary_key(self) -> None:
        frame = pd.DataFrame({"id": [1, 2, 3], "value": [1, 1, 2]})
        schema = detect_schema(frame)
        assert schema.primary_key == ["id"]


class TestPersistence:
    def test_save_and_load_yaml(self, tmp_path: Path) -> None:
        schema = Schema.from_dict({"name": "c", "schema": {"id": {"type": "integer"}}})
        path = tmp_path / "schema.yaml"
        save_schema(schema, path)
        loaded = load_schema(path)
        assert loaded.name == "c"
        assert loaded.get("id").type is FieldType.INTEGER

    def test_save_and_load_json(self, tmp_path: Path) -> None:
        schema = Schema.from_dict({"name": "c", "schema": {"id": {"type": "integer"}}})
        path = tmp_path / "schema.json"
        save_schema(schema, path)
        assert load_schema(path).get("id").type is FieldType.INTEGER

    def test_invalid_yaml_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("name: [unclosed", encoding="utf-8")
        with pytest.raises(SchemaError, match="Could not parse"):
            load_schema(path)

    def test_non_mapping_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(SchemaError, match="must contain a mapping"):
            load_schema(path)

    def test_file_name_becomes_schema_name(self, tmp_path: Path) -> None:
        path = tmp_path / "orders.yaml"
        path.write_text("schema:\n  id:\n    type: integer\n", encoding="utf-8")
        assert load_schema(path).name == "orders"
