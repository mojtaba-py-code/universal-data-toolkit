"""Data enrichment.

Enrichment adds columns from a second source: a lookup table, a mapping, a
derived expression or an external service.  API enrichment is deduplicated and
cached per distinct key - looking up the same country code 50,000 times is the
usual way an enrichment step becomes the slowest part of a pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import pandas as pd

from universal_data.core.exceptions import ConfigurationError, TransformationError
from universal_data.core.types import PathLike
from universal_data.observability.logging import get_logger
from universal_data.rules.expression import SafeExpression, compile_expression
from universal_data.security.paths import PathPolicy

logger = get_logger(__name__)


class Enricher(ABC):
    """Adds columns to a frame from an external source."""

    @abstractmethod
    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the frame with the extra columns."""

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.enrich(frame)


class LookupEnricher(Enricher):
    """Left-joins a reference table onto the dataset."""

    def __init__(
        self,
        lookup: pd.DataFrame,
        *,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        columns: Sequence[str] | None = None,
        suffix: str = "_lookup",
        validate_one_to_one: bool = True,
    ) -> None:
        if hasattr(lookup, "frame"):
            lookup = lookup.frame
        if not isinstance(lookup, pd.DataFrame):
            raise ConfigurationError("Lookup source must be a DataFrame or Dataset")
        self.left_key = left_on or on
        self.right_key = right_on or on
        if not self.left_key or not self.right_key:
            raise ConfigurationError("A lookup needs 'on' or both 'left_on' and 'right_on'")
        if self.right_key not in lookup.columns:
            raise ConfigurationError(
                "Lookup table has no key column", column=self.right_key,
                available=[str(c) for c in lookup.columns],
            )
        if validate_one_to_one and lookup[self.right_key].duplicated().any():
            raise ConfigurationError(
                "Lookup table has duplicate keys, which would multiply rows",
                column=self.right_key,
            )
        keep = list(columns) if columns else [c for c in lookup.columns if c != self.right_key]
        missing = [column for column in keep if column not in lookup.columns]
        if missing:
            raise ConfigurationError("Lookup table is missing column(s)", columns=missing)
        self.lookup = lookup[[self.right_key, *keep]].copy()
        self.suffix = suffix

    @classmethod
    def from_file(
        cls, path: PathLike, *, policy: PathPolicy | None = None, **kwargs: Any
    ) -> LookupEnricher:
        from universal_data.core.dataset import Dataset

        return cls(Dataset.read(path, policy=policy).frame, **kwargs)

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.left_key not in frame.columns:
            raise TransformationError("Dataset has no lookup key column", column=self.left_key)
        merged = frame.merge(
            self.lookup,
            how="left",
            left_on=self.left_key,
            right_on=self.right_key,
            suffixes=("", self.suffix),
        )
        if self.right_key != self.left_key and self.right_key in merged.columns:
            merged = merged.drop(columns=[self.right_key])
        return merged


class MappingEnricher(Enricher):
    """Creates a column from a plain ``{key: value}`` mapping."""

    def __init__(
        self, source: str, target: str, mapping: dict[Any, Any], default: Any = None
    ) -> None:
        self.source = source
        self.target = target
        self.mapping = dict(mapping)
        self.default = default

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.source not in frame.columns:
            raise TransformationError("Source column not found", column=self.source)
        result = frame.copy()
        mapped = result[self.source].map(self.mapping)
        if self.default is not None:
            mapped = mapped.fillna(self.default)
        result[self.target] = mapped
        return result


class DerivedColumnEnricher(Enricher):
    """Adds a calculated column defined by a safe expression."""

    def __init__(self, target: str, expression: str | SafeExpression) -> None:
        self.target = target
        self.expression = compile_expression(expression)

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result[self.target] = self.expression.evaluate(frame)
        return result


class CallableEnricher(Enricher):
    """Enriches from any callable, evaluated once per distinct key.

    This is the seam used for external services: pass a function that takes a
    key and returns a mapping of new column values.  Results are cached, so a
    dataset with 100,000 rows and 40 distinct countries makes 40 calls.
    """

    def __init__(
        self,
        source: str,
        lookup: Callable[[Any], dict[str, Any] | None],
        *,
        columns: Sequence[str] | None = None,
        prefix: str = "",
        on_error: str = "null",
    ) -> None:
        if on_error not in ("null", "raise"):
            raise ConfigurationError("on_error must be 'null' or 'raise'", given=on_error)
        self.source = source
        self.lookup = lookup
        self.columns = list(columns) if columns else None
        self.prefix = prefix
        self.on_error = on_error
        self._cache: dict[Any, dict[str, Any]] = {}

    def _resolve(self, key: Any) -> dict[str, Any]:
        if key in self._cache:
            return self._cache[key]
        try:
            value = self.lookup(key) or {}
        except Exception as exc:
            if self.on_error == "raise":
                raise TransformationError(f"Enrichment lookup failed: {exc}") from exc
            logger.warning("Enrichment lookup failed for one key: %s", type(exc).__name__)
            value = {}
        self._cache[key] = value
        return value

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.source not in frame.columns:
            raise TransformationError("Source column not found", column=self.source)
        result = frame.copy()
        keys = result[self.source].dropna().unique()
        resolved = {key: self._resolve(key) for key in keys}
        columns = self.columns or sorted({k for value in resolved.values() for k in value})
        for column in columns:
            result[f"{self.prefix}{column}"] = result[self.source].map(
                lambda key, col=column: resolved.get(key, {}).get(col)
            )
        logger.debug("Enriched %s rows from %s distinct keys", len(result), len(keys))
        return result

    @property
    def cache_size(self) -> int:
        return len(self._cache)


class EnrichmentChain:
    """Applies several enrichers in order."""

    def __init__(self, enrichers: Iterable[Enricher] = ()) -> None:
        self.enrichers = list(enrichers)

    def add(self, enricher: Enricher) -> EnrichmentChain:
        self.enrichers.append(enricher)
        return self

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame
        for enricher in self.enrichers:
            result = enricher.enrich(result)
        return result

    def __len__(self) -> int:
        return len(self.enrichers)
