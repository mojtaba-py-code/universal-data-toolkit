"""Plugin discovery.

Third parties extend the toolkit by shipping a package that declares an entry
point in the ``universal_data.plugins`` group::

    [project.entry-points."universal_data.plugins"]
    my_formats = "my_package.plugin:register"

The callable receives a :class:`PluginContext` holding every registry, so a
plugin adds readers, writers, transformations or maskers without importing
private modules or patching the core package.

Loading a plugin executes its code.  Discovery is therefore limited to entry
points of *installed* distributions; loading an arbitrary module by name is
possible but requires an explicit call, never a configuration value.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from universal_data.core.exceptions import PluginError
from universal_data.export.base import WriterRegistry, writers
from universal_data.ingestion.base import ReaderRegistry, readers
from universal_data.masking.maskers import MASKERS, Masker
from universal_data.observability.logging import get_logger
from universal_data.transformation.base import TransformationRegistry, transformations

logger = get_logger(__name__)

ENTRY_POINT_GROUP = "universal_data.plugins"


@dataclass
class PluginContext:
    """Handed to every plugin's ``register`` callable."""

    readers: ReaderRegistry = field(default_factory=lambda: readers)
    writers: WriterRegistry = field(default_factory=lambda: writers)
    transformations: TransformationRegistry = field(default_factory=lambda: transformations)

    def register_masker(self, masker: type[Masker]) -> type[Masker]:
        if masker.name in MASKERS:
            raise PluginError("A masker with this name already exists", name=masker.name)
        MASKERS[masker.name] = masker
        return masker


@dataclass
class LoadedPlugin:
    name: str
    source: str
    error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.error is None


def iter_entry_points(group: str = ENTRY_POINT_GROUP) -> Iterator[EntryPoint]:
    yield from entry_points(group=group)


def load_plugins(
    group: str = ENTRY_POINT_GROUP, *, context: PluginContext | None = None
) -> list[LoadedPlugin]:
    """Load every installed plugin, returning one record per entry point."""
    context = context or PluginContext()
    loaded: list[LoadedPlugin] = []
    for entry_point in iter_entry_points(group):
        record = LoadedPlugin(name=entry_point.name, source=entry_point.value)
        try:
            register = entry_point.load()
            _invoke(register, context)
            logger.info("Loaded plugin '%s' from %s", entry_point.name, entry_point.value)
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            logger.error("Plugin '%s' failed to load: %s", entry_point.name, exc)
        loaded.append(record)
    return loaded


def load_plugin_module(module_path: str, *, context: PluginContext | None = None) -> LoadedPlugin:
    """Load a plugin from ``package.module:callable``.

    Only call this with a value that came from the operator (a CLI flag), never
    from a data file.
    """
    context = context or PluginContext()
    module_name, _, attribute = module_path.partition(":")
    record = LoadedPlugin(name=module_name, source=module_path)
    try:
        module = importlib.import_module(module_name)
        register = getattr(module, attribute or "register", None)
        if register is None:
            raise PluginError(
                "Plugin module has no register() callable", module=module_name
            )
        _invoke(register, context)
    except PluginError as exc:
        record.error = str(exc)
    except Exception as exc:
        record.error = f"{type(exc).__name__}: {exc}"
    return record


def _invoke(register: Callable[..., Any], context: PluginContext) -> None:
    if not callable(register):
        raise PluginError("Plugin entry point is not callable")
    try:
        register(context)
    except TypeError:
        # Allow plugins that register on import and expose a zero-argument hook.
        register()


def installed_plugins() -> list[str]:
    return [entry_point.name for entry_point in iter_entry_points()]
