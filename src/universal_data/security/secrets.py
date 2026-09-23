"""Secret handling.

Credentials never live in configuration files.  YAML and CLI options carry
*references* (``env:PG_PASSWORD`` or ``${PG_PASSWORD}``) which are resolved at
runtime against the process environment, optionally seeded from a ``.env`` file.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from universal_data.core.exceptions import ConfigurationError
from universal_data.core.types import PathLike

_REFERENCE = re.compile(r"^(?:env:(?P<direct>[A-Za-z_][A-Za-z0-9_]*)|\$\{(?P<braced>[^}]+)\})$")
_INLINE_REFERENCE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


class SecretStr:
    """Wrapper that keeps a secret out of logs, reprs and tracebacks."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SecretStr) and other._value == self._value

    def __hash__(self) -> int:
        return hash(("SecretStr", self._value))

    def __str__(self) -> str:
        return "***" if self._value else ""

    def __repr__(self) -> str:
        return "SecretStr('***')"


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines.  Deliberately tiny: no interpolation, no export."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key.isidentifier():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(path: PathLike = ".env", *, override: bool = False) -> dict[str, str]:
    """Load a ``.env`` file into ``os.environ``.  Missing files are ignored."""
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    values = parse_dotenv(file_path.read_text(encoding="utf-8"))
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


class SecretResolver:
    """Resolves secret references against an environment mapping."""

    def __init__(self, environ: MutableMapping[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else os.environ

    def get(self, name: str, *, required: bool = False, default: str | None = None) -> SecretStr | None:
        value = self._environ.get(name, default)
        if value is None or value == "":
            if required:
                raise ConfigurationError(
                    "Required secret is not set in the environment", variable=name
                )
            return None
        return SecretStr(value)

    def resolve(self, value: Any, *, required: bool = True) -> Any:
        """Expand a single value if it is a secret reference.

        ``env:NAME`` and ``${NAME}`` resolve to the raw string.  ``${NAME:-fallback}``
        is also supported for non-sensitive defaults such as hosts and ports.
        """
        if not isinstance(value, str):
            return value
        match = _REFERENCE.match(value.strip())
        if match:
            name = match.group("direct") or match.group("braced")
            if ":-" in name:
                name, _, fallback = name.partition(":-")
                return self._environ.get(name.strip(), fallback)
            resolved = self._environ.get(name)
            if resolved is None:
                if required:
                    raise ConfigurationError(
                        "Configuration references an undefined environment variable",
                        variable=name,
                    )
                return None
            return resolved

        def _expand(m: re.Match[str]) -> str:
            name = m.group("name")
            fallback = m.group("default")
            found = self._environ.get(name)
            if found is None:
                if fallback is not None:
                    return fallback
                if required:
                    raise ConfigurationError(
                        "Configuration references an undefined environment variable",
                        variable=name,
                    )
                return ""
            return found

        return _INLINE_REFERENCE.sub(_expand, value)

    def resolve_mapping(self, data: Mapping[str, Any], *, required: bool = True) -> dict[str, Any]:
        """Recursively resolve references inside a nested configuration mapping."""
        result: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, Mapping):
                result[key] = self.resolve_mapping(value, required=required)
            elif isinstance(value, list):
                result[key] = [
                    self.resolve_mapping(item, required=required)
                    if isinstance(item, Mapping)
                    else self.resolve(item, required=required)
                    for item in value
                ]
            else:
                result[key] = self.resolve(value, required=required)
        return result
