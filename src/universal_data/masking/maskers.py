"""Privacy transformations.

Masking is applied to a copy of the data and the original values are never
logged.  Hashing is salted: an unsalted hash of an email address or a phone
number is trivially reversible with a dictionary attack, so the salt is required
in production and only defaults to a random per-run value for local use.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

from universal_data.core.exceptions import ConfigurationError
from universal_data.observability.logging import get_logger
from universal_data.security.secrets import SecretStr

logger = get_logger(__name__)

SALT_ENV_VAR = "UD_MASKING_SALT"
_EMAIL_SPLIT = re.compile(r"^([^@]+)@(.+)$")
_DIGITS = re.compile(r"\D+")


def _resolve_salt(salt: SecretStr | str | None) -> SecretStr:
    if isinstance(salt, SecretStr):
        return salt
    if isinstance(salt, str) and salt:
        return SecretStr(salt)
    from_env = os.environ.get(SALT_ENV_VAR)
    if from_env:
        return SecretStr(from_env)
    logger.warning(
        "%s is not set; a random per-run salt is used, so hashes will not be stable "
        "between runs",
        SALT_ENV_VAR,
    )
    return SecretStr(secrets.token_hex(16))


class Masker(ABC):
    """Transforms a column of sensitive values into a safe representation."""

    name: ClassVar[str]

    @abstractmethod
    def mask(self, series: pd.Series) -> pd.Series:
        """Return the masked column, preserving nulls."""

    def _text(self, series: pd.Series) -> pd.Series:
        return series.astype("string")


@dataclass
class EmailMasker(Masker):
    """``john.smith@example.com`` -> ``j***@example.com``."""

    name: ClassVar[str] = "email"
    keep: int = 1
    mask_domain: bool = False

    def mask(self, series: pd.Series) -> pd.Series:
        def transform(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            match = _EMAIL_SPLIT.match(value)
            if not match:
                return "***"
            local, domain = match.groups()
            head = local[: self.keep]
            if self.mask_domain:
                parts = domain.rsplit(".", 1)
                domain = f"***.{parts[-1]}" if len(parts) == 2 else "***"
            return f"{head}***@{domain}"

        return self._text(series).map(transform, na_action="ignore")


@dataclass
class PhoneMasker(Masker):
    """Keeps the last few digits so records stay recognisable to support staff."""

    name: ClassVar[str] = "phone"
    keep_last: int = 4

    def mask(self, series: pd.Series) -> pd.Series:
        def transform(value: Any) -> Any:
            if not isinstance(value, str):
                value = str(value)
            digits = _DIGITS.sub("", value)
            if len(digits) <= self.keep_last:
                return "*" * len(digits)
            return "*" * (len(digits) - self.keep_last) + digits[-self.keep_last :]

        return self._text(series).map(transform, na_action="ignore")


@dataclass
class NameMasker(Masker):
    """``Ali Rezaei`` -> ``A. R.``"""

    name: ClassVar[str] = "name"

    def mask(self, series: pd.Series) -> pd.Series:
        def transform(value: Any) -> Any:
            if not isinstance(value, str) or not value.strip():
                return value
            parts = [part for part in value.split() if part]
            return " ".join(f"{part[0].upper()}." for part in parts)

        return self._text(series).map(transform, na_action="ignore")


@dataclass
class PartialMasker(Masker):
    """Generic partial redaction: keep some characters at each end."""

    name: ClassVar[str] = "partial"
    keep_first: int = 2
    keep_last: int = 2
    character: str = "*"

    def mask(self, series: pd.Series) -> pd.Series:
        def transform(value: Any) -> Any:
            text = value if isinstance(value, str) else str(value)
            if len(text) <= self.keep_first + self.keep_last:
                return self.character * len(text)
            middle = self.character * (len(text) - self.keep_first - self.keep_last)
            return f"{text[: self.keep_first]}{middle}{text[len(text) - self.keep_last :]}"

        return self._text(series).map(transform, na_action="ignore")


@dataclass
class HashMasker(Masker):
    """Salted, deterministic pseudonymisation - joins still work after masking."""

    name: ClassVar[str] = "hash"
    salt: SecretStr | str | None = None
    algorithm: str = "sha256"
    length: int = 16
    _salt: SecretStr = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.algorithm not in hashlib.algorithms_available:
            raise ConfigurationError("Unknown hash algorithm", algorithm=self.algorithm)
        if self.algorithm in ("md5", "sha1"):
            raise ConfigurationError(
                "md5 and sha1 are not acceptable for pseudonymisation", algorithm=self.algorithm
            )
        self._salt = _resolve_salt(self.salt)
        self.salt = None  # keep the raw salt out of reprs and pickles

    def mask(self, series: pd.Series) -> pd.Series:
        salt = self._salt.get_secret_value().encode("utf-8")
        length = self.length
        algorithm = self.algorithm

        def transform(value: Any) -> Any:
            digest = hashlib.new(algorithm, salt + str(value).encode("utf-8")).hexdigest()
            return digest[:length] if length else digest

        return self._text(series).map(transform, na_action="ignore")


@dataclass
class TokenMasker(Masker):
    """Replaces each distinct value with a random token, consistent per run."""

    name: ClassVar[str] = "token"
    prefix: str = "tok_"
    _tokens: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def mask(self, series: pd.Series) -> pd.Series:
        def transform(value: Any) -> Any:
            key = str(value)
            if key not in self._tokens:
                self._tokens[key] = f"{self.prefix}{uuid.uuid4().hex[:12]}"
            return self._tokens[key]

        return self._text(series).map(transform, na_action="ignore")


@dataclass
class RedactMasker(Masker):
    """Drops the value entirely."""

    name: ClassVar[str] = "redact"
    replacement: str = "[REDACTED]"

    def mask(self, series: pd.Series) -> pd.Series:
        return pd.Series(
            [self.replacement if pd.notna(value) else value for value in series],
            index=series.index,
            dtype="string",
        )


MASKERS: dict[str, type[Masker]] = {
    masker.name: masker
    for masker in (
        EmailMasker,
        PhoneMasker,
        NameMasker,
        PartialMasker,
        HashMasker,
        TokenMasker,
        RedactMasker,
    )
}


def create_masker(name: str, **options: Any) -> Masker:
    try:
        masker_cls = MASKERS[name]
    except KeyError:
        raise ConfigurationError(
            "Unknown masking strategy", strategy=name, available=sorted(MASKERS)
        ) from None
    return masker_cls(**options)


@dataclass
class MaskingPolicy:
    """Maps columns to masking strategies."""

    rules: dict[str, Masker] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaskingPolicy:
        rules: dict[str, Masker] = {}
        for column, spec in data.items():
            if isinstance(spec, str):
                rules[column] = create_masker(spec)
            elif isinstance(spec, dict):
                options = dict(spec)
                strategy = options.pop("strategy", options.pop("type", None))
                if not strategy:
                    raise ConfigurationError("Masking rule needs a strategy", column=column)
                rules[column] = create_masker(str(strategy), **options)
            else:
                raise ConfigurationError("Invalid masking rule", column=column)
        return cls(rules=rules)

    @classmethod
    def for_columns(cls, columns: Sequence[str], strategy: str = "hash", **options: Any) -> MaskingPolicy:
        return cls(rules={column: create_masker(strategy, **options) for column in columns})

    def add(self, column: str, masker: Masker) -> MaskingPolicy:
        self.rules[column] = masker
        return self

    def apply(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        result = frame.copy()
        applied: dict[str, str] = {}
        skipped: list[str] = []
        for column, masker in self.rules.items():
            if column not in result.columns:
                skipped.append(column)
                continue
            result[column] = masker.mask(result[column])
            applied[column] = masker.name
        if skipped:
            logger.warning("Masking skipped for missing column(s): %s", skipped)
        return result, {"operation": "mask", "applied": applied, "skipped": skipped}

    def __len__(self) -> int:
        return len(self.rules)
