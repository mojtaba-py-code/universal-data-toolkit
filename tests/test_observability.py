"""Logging, metrics, memory accounting, the generator and the plugin loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from universal_data.generator.synthetic import (
    QualityIssues,
    SyntheticDataGenerator,
    generate_dataset,
)
from universal_data.observability.logging import (
    ConsoleFormatter,
    JsonFormatter,
    configure_logging,
    current_context,
    get_logger,
    log_context,
    new_execution_id,
)
from universal_data.observability.memory import (
    MemoryTracker,
    dataframe_memory_mb,
    estimate_row_size_bytes,
    suggest_chunk_size,
)
from universal_data.observability.metrics import (
    MetricsCollector,
    PrometheusTextSink,
    RunMetrics,
    StepMetrics,
)
from universal_data.plugins.loader import (
    LoadedPlugin,
    PluginContext,
    installed_plugins,
    load_plugin_module,
)


class TestLogging:
    def test_execution_ids_are_unique(self) -> None:
        assert new_execution_id() != new_execution_id()

    def test_context_is_bound_and_restored(self) -> None:
        assert current_context() == {}
        with log_context(execution_id="abc", step="clean"):
            assert current_context()["execution_id"] == "abc"
            with log_context(step="write"):
                assert current_context()["step"] == "write"
            assert current_context()["step"] == "clean"
        assert current_context() == {}

    def test_json_formatter_includes_context(self) -> None:
        record = logging.LogRecord("x", logging.INFO, "f", 1, "hello %s", ("world",), None)
        record.execution_id = "abc"
        payload = json.loads(JsonFormatter().format(record))
        assert payload["message"] == "hello world"
        assert payload["execution_id"] == "abc"
        assert payload["level"] == "INFO"

    def test_json_formatter_serialises_exceptions(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord("x", logging.ERROR, "f", 1, "failed", None, sys.exc_info())
        payload = json.loads(JsonFormatter().format(record))
        assert "ValueError" in payload["exception"]

    def test_console_formatter_appends_context(self) -> None:
        record = logging.LogRecord("x", logging.INFO, "f", 1, "msg", None, None)
        record.execution_id = "abc"
        record.step = "clean"
        rendered = ConsoleFormatter().format(record)
        assert "[abc]" in rendered
        assert "(clean)" in rendered

    def test_configure_logging_writes_a_rotating_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "logs" / "app.log"
        logger = configure_logging("DEBUG", log_file=log_file, quiet=True)
        logger.info("hello")
        for handler in logger.handlers:
            handler.flush()
        assert log_file.exists()
        assert "hello" in log_file.read_text(encoding="utf-8")
        configure_logging("WARNING", quiet=True)

    def test_configure_logging_replaces_handlers(self) -> None:
        first = configure_logging("INFO")
        second = configure_logging("INFO")
        assert first is second
        assert len(second.handlers) == 1
        configure_logging("WARNING", quiet=True)

    def test_quiet_mode_uses_a_null_handler(self) -> None:
        logger = configure_logging("INFO", quiet=True)
        assert isinstance(logger.handlers[0], logging.NullHandler)

    def test_get_logger_namespacing(self) -> None:
        assert get_logger("cleaning").name == "universal_data.cleaning"
        assert get_logger("universal_data.x").name == "universal_data.x"


class TestMetrics:
    def test_step_metrics_throughput(self) -> None:
        step = StepMetrics(name="s", rows_out=100, duration_seconds=2.0)
        assert step.throughput == 50.0
        assert StepMetrics(name="s").throughput == 0.0
        assert step.to_dict()["throughput"] == 50.0

    def test_collector_tracks_success(self) -> None:
        collector = MetricsCollector("id", "pipeline")
        with collector.step("clean", rows_in=10) as step:
            step.rows_out = 8
        run = collector.finish()
        assert run.status == "success"
        assert run.rows_input == 10
        assert run.rows_output == 8
        assert run.duration_seconds >= 0

    def test_collector_records_failures(self) -> None:
        collector = MetricsCollector("id")
        with pytest.raises(RuntimeError), collector.step("boom"):
            raise RuntimeError("nope")
        run = collector.finish()
        assert run.status == "failed"
        assert "RuntimeError" in run.steps[0].error

    def test_empty_run(self) -> None:
        run = MetricsCollector("id").finish()
        assert run.status == "empty"
        assert run.rows_output == 0
        assert run.throughput == 0.0

    def test_totals_override_step_counts(self) -> None:
        run = RunMetrics(execution_id="id")
        run.rows_input_total = 500
        run.rows_output_total = 480
        assert run.rows_input == 500
        assert run.rows_output == 480

    def test_prometheus_sink_is_empty_before_use(self) -> None:
        assert PrometheusTextSink().render() == ""


class TestMemory:
    def test_dataframe_memory(self) -> None:
        frame = pd.DataFrame({"a": range(1000), "b": ["text"] * 1000})
        assert dataframe_memory_mb(frame) > 0
        assert dataframe_memory_mb(pd.DataFrame()) == 0.0

    def test_row_size_estimation(self) -> None:
        frame = pd.DataFrame({"a": range(100)})
        assert estimate_row_size_bytes(frame) > 0
        assert estimate_row_size_bytes(pd.DataFrame()) == 0.0

    def test_chunk_size_suggestion(self) -> None:
        frame = pd.DataFrame({"a": range(1000), "b": ["x" * 50] * 1000})
        chunk = suggest_chunk_size(frame, target_mb=1.0)
        assert 1_000 <= chunk <= 1_000_000
        assert suggest_chunk_size(pd.DataFrame(), target_mb=1.0) == 1_000

    def test_tracker(self) -> None:
        tracker = MemoryTracker()
        assert tracker.process_memory_mb() > 0
        assert tracker.peak_memory_mb() >= tracker.process_memory_mb() - 1
        snapshot = tracker.snapshot()
        assert snapshot.available_mb > 0

    def test_headroom_warning(self) -> None:
        tracker = MemoryTracker()
        assert tracker.check_headroom(1.0) is None
        assert tracker.check_headroom(10_000_000.0) is not None


class TestGenerator:
    def test_datasets_are_reproducible(self) -> None:
        first = SyntheticDataGenerator(seed=5).customers(50)
        second = SyntheticDataGenerator(seed=5).customers(50)
        assert first.equals(second)

    @pytest.mark.parametrize(
        "kind", ["customers", "products", "orders", "transactions", "employees"]
    )
    def test_every_dataset_kind(self, kind: str) -> None:
        frame = generate_dataset(kind, rows=25)
        assert len(frame) == 25
        assert not frame.empty

    def test_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            generate_dataset("aliens")

    def test_clean_profile_has_no_defects(self) -> None:
        frame = SyntheticDataGenerator(seed=1).customers(100, issues=QualityIssues.clean())
        assert frame.isna().sum().sum() == 0
        assert frame.duplicated().sum() == 0

    def test_realistic_profile_injects_defects(self) -> None:
        frame = SyntheticDataGenerator(seed=1).customers(300, issues=QualityIssues.realistic())
        assert frame.isna().sum().sum() > 0
        assert frame.duplicated().sum() > 0
        assert (~frame["email"].dropna().str.contains("@", regex=False)).sum() > 0

    def test_severe_profile_is_worse(self) -> None:
        realistic = SyntheticDataGenerator(seed=2).customers(300, issues=QualityIssues.realistic())
        severe = SyntheticDataGenerator(seed=2).customers(300, issues=QualityIssues.severe())
        assert severe.isna().sum().sum() > realistic.isna().sum().sum()

    def test_country_lookup(self) -> None:
        lookup = SyntheticDataGenerator().country_lookup()
        assert list(lookup.columns) == ["country_code", "country_name", "region"]
        assert not lookup["country_code"].duplicated().any()

    def test_orders_reference_given_ids(self) -> None:
        orders = SyntheticDataGenerator(seed=3).orders(
            50, customer_ids=[1, 2], product_ids=[7]
        )
        assert set(orders["customer_id"]) <= {1, 2}
        assert set(orders["product_id"]) == {7}


class TestPlugins:
    def test_no_plugins_installed_by_default(self) -> None:
        assert isinstance(installed_plugins(), list)

    def test_load_module_that_does_not_exist(self) -> None:
        record = load_plugin_module("universal_data_missing_plugin")
        assert not record.loaded
        assert "ModuleNotFoundError" in record.error

    def test_load_module_without_register(self) -> None:
        record = load_plugin_module("json")
        assert not record.loaded
        assert "register" in record.error

    def test_plugin_context_registers_a_masker(self) -> None:
        from universal_data.masking.maskers import MASKERS, Masker

        class UpperMasker(Masker):
            name = "upper_test"

            def mask(self, series: pd.Series) -> pd.Series:
                return series.astype("string").str.upper()

        context = PluginContext()
        context.register_masker(UpperMasker)
        try:
            assert "upper_test" in MASKERS
            from universal_data.core.exceptions import PluginError

            with pytest.raises(PluginError, match="already exists"):
                context.register_masker(UpperMasker)
        finally:
            MASKERS.pop("upper_test", None)

    def test_plugin_context_exposes_the_registries(self) -> None:
        context = PluginContext()
        assert "csv" in context.readers.formats()
        assert "csv" in context.writers.formats()
        assert "filter" in context.transformations.names()

    def test_loaded_plugin_record(self) -> None:
        assert LoadedPlugin(name="a", source="b").loaded is True
        assert LoadedPlugin(name="a", source="b", error="x").loaded is False
