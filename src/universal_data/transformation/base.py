"""Transformation interface, registry and chaining."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any, ClassVar

import pandas as pd

from universal_data.core.exceptions import TransformationError
from universal_data.observability.logging import get_logger

logger = get_logger(__name__)


class Transformation(ABC):
    """A single, composable operation on a DataFrame.

    Implementations must be pure: given the same input they return the same
    output and never mutate the frame they receive.  That is what makes a
    pipeline replayable and a failed step recoverable.
    """

    name: ClassVar[str]

    @abstractmethod
    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a new frame with the transformation applied."""

    @classmethod
    def from_config(cls, config: Any) -> Transformation:
        """Build the transformation from its YAML fragment."""
        if isinstance(config, dict):
            return cls(**config)
        raise TransformationError(
            f"{cls.name} expects a mapping of options", given=type(config).__name__
        )

    def describe(self) -> str:
        options = ", ".join(
            f"{key}={value!r}"
            for key, value in vars(self).items()
            if not key.startswith("_") and value is not None
        )
        return f"{self.name}({options})"

    def __repr__(self) -> str:
        return self.describe()

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.apply(frame)


class TransformationRegistry:
    """Name to class mapping used by the YAML loader and by plugins."""

    def __init__(self) -> None:
        self._items: dict[str, type[Transformation]] = {}

    def register(
        self, transformation: type[Transformation], *, replace: bool = False
    ) -> type[Transformation]:
        key = transformation.name
        if not key:
            raise ValueError("A transformation needs a name")
        if key in self._items and not replace:
            raise ValueError(f"Transformation '{key}' is already registered")
        self._items[key] = transformation
        return transformation

    def get(self, name: str) -> type[Transformation]:
        try:
            return self._items[name]
        except KeyError:
            raise TransformationError(
                "Unknown transformation", name=name, available=self.names()
            ) from None

    def create(self, name: str, config: Any = None) -> Transformation:
        return self.get(name).from_config(config if config is not None else {})

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items


transformations = TransformationRegistry()


def register_transformation(transformation: type[Transformation]) -> type[Transformation]:
    return transformations.register(transformation)


class TransformationChain:
    """An ordered list of transformations applied as one unit."""

    def __init__(self, steps: Iterable[Transformation] = ()) -> None:
        self.steps: list[Transformation] = list(steps)

    def add(self, transformation: Transformation) -> TransformationChain:
        self.steps.append(transformation)
        return self

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame
        for step in self.steps:
            try:
                result = step.apply(result)
            except TransformationError:
                raise
            except Exception as exc:
                raise TransformationError(
                    f"Transformation failed: {exc}", transformation=step.describe()
                ) from exc
            logger.debug("Applied %s -> %s rows", step.name, len(result))
        return result

    @classmethod
    def from_config(cls, config: Any) -> TransformationChain:
        """Build a chain from ``[{rename: {...}}, {filter: {...}}]`` style YAML."""
        steps: list[Transformation] = []
        if isinstance(config, dict):
            entries: list[Any] = [{key: value} for key, value in config.items()]
        elif isinstance(config, list):
            entries = list(config)
        else:
            raise TransformationError(
                "Transformations must be a list or a mapping", given=type(config).__name__
            )
        for entry in entries:
            if isinstance(entry, str):
                steps.append(transformations.create(entry, {}))
                continue
            if not isinstance(entry, dict) or len(entry) != 1:
                raise TransformationError(
                    "Each transformation entry must be a single-key mapping", entry=str(entry)
                )
            (name, options), = entry.items()
            steps.append(transformations.create(name, options))
        return cls(steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Transformation]:
        return iter(self.steps)

    def __repr__(self) -> str:
        return f"TransformationChain({[step.describe() for step in self.steps]})"
