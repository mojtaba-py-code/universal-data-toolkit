"""Plugin system."""

from universal_data.plugins.loader import (
    ENTRY_POINT_GROUP,
    LoadedPlugin,
    PluginContext,
    installed_plugins,
    load_plugin_module,
    load_plugins,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "LoadedPlugin",
    "PluginContext",
    "installed_plugins",
    "load_plugin_module",
    "load_plugins",
]
