# Plugins

New readers, writers, transformations and maskers can be added from a separate
package, without touching the core.

## Declare an entry point

```toml
# my_package/pyproject.toml
[project.entry-points."universal_data.plugins"]
avro = "my_package.plugin:register"
```

## Implement `register`

The callable receives a `PluginContext` holding every registry.

```python
# my_package/plugin.py
from typing import Any

import pandas as pd
from universal_data.core.types import FileFormat, PathLike
from universal_data.export.base import Writer
from universal_data.ingestion.base import Reader
from universal_data.masking.maskers import Masker
from universal_data.plugins import PluginContext
from universal_data.transformation.base import Transformation


class AvroReader(Reader):
    format = FileFormat.PICKLE      # or a value your package adds to FileFormat
    extensions = (".avro",)

    def read(self, source: PathLike, **options: Any) -> pd.DataFrame:
        path = self.resolve(source)          # goes through the PathPolicy
        import fastavro
        with path.open("rb") as handle:
            return pd.DataFrame(list(fastavro.reader(handle)))


class RoundToNearest(Transformation):
    name = "round_to_nearest"

    def __init__(self, column: str, step: int = 10) -> None:
        self.column = column
        self.step = step

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result[self.column] = (result[self.column] / self.step).round() * self.step
        return result


class IbanMasker(Masker):
    name = "iban"

    def mask(self, series: pd.Series) -> pd.Series:
        text = series.astype("string")
        return text.str[:4] + "****" + text.str[-4:]


def register(context: PluginContext) -> None:
    context.readers.register(AvroReader)
    context.transformations.register(RoundToNearest)
    context.register_masker(IbanMasker)
```

Once the package is installed, the plugin is discovered automatically:

```bash
pip install my-package
data-tool plugins
```

```python
from universal_data.plugins import load_plugins

for plugin in load_plugins():
    print(plugin.name, "ok" if plugin.loaded else plugin.error)
```

The registered names are then usable everywhere the built-ins are, including
YAML:

```yaml
steps:
  - transform:
      - round_to_nearest:
          column: amount
          step: 50
  - mask:
      iban_number: iban
```

## Interfaces

| Extension point | Base class | Required members |
| --- | --- | --- |
| Reader | `universal_data.ingestion.base.Reader` | `format`, `extensions`, `read()`; optionally `read_chunks()` |
| Writer | `universal_data.export.base.Writer` | `format`, `extensions`, `write()`; optionally `write_chunks()` |
| Transformation | `universal_data.transformation.base.Transformation` | `name`, `apply()`; optionally `from_config()` |
| Masker | `universal_data.masking.maskers.Masker` | `name`, `mask()` |
| Enricher | `universal_data.enrichment.enrichers.Enricher` | `enrich()` |
| Metrics sink | `universal_data.observability.metrics.MetricsSink` | `emit()` |
| Validator | a function returning a boolean Series | see `universal_data.validation.checks` |

## Rules

* **Do not mutate the frame you receive.** Every transformation must return a new
  one; the pipeline relies on that to recover from a failed step.
* **Resolve paths through `self.resolve()`** so the caller's `PathPolicy`
  applies.
* **Raise the toolkit's exceptions** (`ExtractionError`, `TransformationError`,
  `ExportError`) so the CLI can report them properly.
* **Registration is not idempotent.** Registering an existing name raises;
  pass `replace=True` if you really mean to override a built-in.

## Loading a plugin explicitly

Loading a plugin executes its code, so discovery is limited to entry points of
installed distributions. A module can be loaded by name, but only from an
operator-supplied value - never from a data file or a pipeline configuration:

```python
from universal_data.plugins import load_plugin_module

record = load_plugin_module("my_package.plugin:register")
```
