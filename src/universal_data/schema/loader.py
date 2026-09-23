"""Loading and saving schema definitions."""

from __future__ import annotations

import json
from typing import Any

import yaml

from universal_data.core.exceptions import SchemaError
from universal_data.core.types import PathLike
from universal_data.schema.model import Schema
from universal_data.security.paths import PathPolicy


def load_schema(path: PathLike, *, policy: PathPolicy | None = None) -> Schema:
    """Read a schema from a YAML or JSON file."""
    resolved = (policy or PathPolicy()).resolve_input(path)
    text = resolved.read_text(encoding="utf-8")
    try:
        if resolved.suffix.lower() == ".json":
            payload: Any = json.loads(text)
        else:
            payload = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise SchemaError(f"Could not parse the schema file: {exc}", path=str(resolved)) from exc
    if not isinstance(payload, dict):
        raise SchemaError("A schema file must contain a mapping", path=str(resolved))
    schema = Schema.from_dict(payload)
    if schema.name == "schema":
        schema.name = resolved.stem
    return schema


def save_schema(schema: Schema, path: PathLike, *, policy: PathPolicy | None = None) -> None:
    """Write a schema to YAML (or JSON when the path ends in ``.json``)."""
    resolved = (policy or PathPolicy()).resolve_output(path)
    payload = schema.to_dict()
    if resolved.suffix.lower() == ".json":
        resolved.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    else:
        resolved.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
