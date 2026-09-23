"""The benchmark harness.

Run with a tiny dataset here; ``data-tool benchmark`` is where the real numbers
come from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from universal_data.cli.benchmarks import run_benchmarks
from universal_data.plugins.loader import PluginContext, load_plugins


def test_benchmarks_run_and_report(tmp_path: Path) -> None:
    results = run_benchmarks(rows=300, chunk_size=100, output_dir=tmp_path)
    scenarios = {item["scenario"] for item in results}
    assert scenarios == {
        "row_wise apply",
        "vectorised expression",
        "in_memory pipeline",
        "chunked pipeline",
    }
    for item in results:
        assert item["seconds"] > 0
        assert item["throughput"] > 0
    assert (tmp_path / "bench_memory.parquet").exists()
    assert (tmp_path / "bench_chunked.parquet").exists()


@pytest.mark.slow
def test_vectorised_beats_row_wise(tmp_path: Path) -> None:
    results = {item["scenario"]: item for item in run_benchmarks(rows=20_000, output_dir=tmp_path)}
    assert results["vectorised expression"]["seconds"] < results["row_wise apply"]["seconds"]


def test_plugin_discovery_handles_a_broken_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenEntryPoint:
        name = "broken"
        value = "nowhere:register"

        def load(self) -> object:
            raise ImportError("no such module")

    class GoodEntryPoint:
        name = "good"
        value = "somewhere:register"
        called: list[PluginContext] = []

        def load(self) -> object:
            def register(context: PluginContext) -> None:
                GoodEntryPoint.called.append(context)

            return register

    monkeypatch.setattr(
        "universal_data.plugins.loader.iter_entry_points",
        lambda group="universal_data.plugins": iter([BrokenEntryPoint(), GoodEntryPoint()]),
    )
    loaded = load_plugins()
    assert [(item.name, item.loaded) for item in loaded] == [("broken", False), ("good", True)]
    assert len(GoodEntryPoint.called) == 1


def test_plugin_accepts_a_zero_argument_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class EntryPoint:
        name = "legacy"
        value = "legacy:register"

        def load(self) -> object:
            def register() -> None:
                calls.append("called")

            return register

    monkeypatch.setattr(
        "universal_data.plugins.loader.iter_entry_points",
        lambda group="universal_data.plugins": iter([EntryPoint()]),
    )
    assert load_plugins()[0].loaded
    assert calls == ["called"]
