"""Schema representation.

A schema is a declarative description of the columns a dataset must have.  It is
serialisable to YAML so that it can live next to the pipeline configuration and
be reviewed like any other artefact.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from universal_data.core.exceptions import SchemaError
from universal_data.core.types import FieldType


@dataclass
class ColumnSchema:
    """Declared expectations for a single column."""

    name: str
    type: FieldType = FieldType.STRING
    required: bool = True
    nullable: bool = True
    unique: bool = False
    min: float | str | None = None
    max: float | str | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    allowed: list[Any] | None = None
    description: str = ""
    pii: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise SchemaError("Column schema needs a name")
        if not isinstance(self.type, FieldType):
            try:
                self.type = FieldType(str(self.type).lower())
            except ValueError as exc:
                raise SchemaError(
                    "Unknown column type",
                    column=self.name,
                    type=str(self.type),
                    supported=[str(t) for t in FieldType],
                ) from exc
        if self.min is not None and self.max is not None:
            try:
                if float(self.min) > float(self.max):
                    raise SchemaError("min is greater than max", column=self.name)
            except (TypeError, ValueError):
                pass  # date bounds are compared as strings during validation

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any] | str) -> ColumnSchema:
        if isinstance(data, str):
            return cls(name=name, type=FieldType(data.lower()))
        payload = dict(data)
        known = {f for f in cls.__dataclass_fields__ if f != "name"}
        unknown = set(payload) - known
        if unknown:
            raise SchemaError(
                "Unknown schema keys", column=name, keys=sorted(unknown), allowed=sorted(known)
            )
        return cls(name=name, **payload)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": str(self.type)}
        defaults = {
            "required": True,
            "nullable": True,
            "unique": False,
            "min": None,
            "max": None,
            "min_length": None,
            "max_length": None,
            "pattern": None,
            "allowed": None,
            "description": "",
            "pii": False,
        }
        for key, default in defaults.items():
            value = getattr(self, key)
            if value != default:
                data[key] = value
        return data


@dataclass
class Schema:
    """An ordered collection of column schemas."""

    name: str = "schema"
    columns: dict[str, ColumnSchema] = field(default_factory=dict)
    strict: bool = False
    primary_key: list[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        for key in self.primary_key:
            if key not in self.columns:
                raise SchemaError("Primary key column is not declared", column=key, schema=self.name)

    @classmethod
    def from_columns(cls, columns: Iterable[ColumnSchema], **kwargs: Any) -> Schema:
        return cls(columns={column.name: column for column in columns}, **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Schema:
        payload = dict(data)
        # Accept both {"schema": {...}} and a bare column mapping.
        if "schema" in payload and isinstance(payload["schema"], Mapping):
            columns_data = payload["schema"]
        elif "columns" in payload and isinstance(payload["columns"], Mapping):
            columns_data = payload["columns"]
        else:
            columns_data = {
                key: value
                for key, value in payload.items()
                if key not in ("name", "strict", "primary_key", "description")
            }
        if not columns_data:
            raise SchemaError("Schema does not declare any column")
        columns = {
            str(name): ColumnSchema.from_dict(str(name), spec)
            for name, spec in columns_data.items()
        }
        primary_key = payload.get("primary_key") or []
        if isinstance(primary_key, str):
            primary_key = [primary_key]
        return cls(
            name=str(payload.get("name", "schema")),
            columns=columns,
            strict=bool(payload.get("strict", False)),
            primary_key=list(primary_key),
            description=str(payload.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "schema": {name: column.to_dict() for name, column in self.columns.items()},
        }
        if self.strict:
            data["strict"] = True
        if self.primary_key:
            data["primary_key"] = list(self.primary_key)
        if self.description:
            data["description"] = self.description
        return data

    @property
    def column_names(self) -> list[str]:
        return list(self.columns)

    @property
    def required_columns(self) -> list[str]:
        return [name for name, column in self.columns.items() if column.required]

    @property
    def pii_columns(self) -> list[str]:
        return [name for name, column in self.columns.items() if column.pii]

    def add(self, column: ColumnSchema) -> Schema:
        self.columns[column.name] = column
        return self

    def get(self, name: str) -> ColumnSchema | None:
        return self.columns.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self.columns

    def __iter__(self) -> Iterator[ColumnSchema]:
        return iter(self.columns.values())

    def __len__(self) -> int:
        return len(self.columns)

    def describe(self) -> str:
        """Readable rendering used by ``data-tool schema``."""
        width = max((len(name) for name in self.columns), default=4)
        lines = [self.name, ""]
        for name, column in self.columns.items():
            flags = []
            if not column.required:
                flags.append("optional")
            if not column.nullable:
                flags.append("not null")
            if column.unique:
                flags.append("unique")
            if column.pii:
                flags.append("pii")
            suffix = f"  ({', '.join(flags)})" if flags else ""
            lines.append(f"{name.ljust(width)} -> {column.type}{suffix}")
        return "\n".join(lines)
