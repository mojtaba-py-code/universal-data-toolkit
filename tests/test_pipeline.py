"""The pipeline engine and the YAML configuration loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from universal_data.core.dataset import Dataset
from universal_data.core.exceptions import ConfigurationError, PipelineError, SecurityError
from universal_data.observability.metrics import InMemorySink, PrometheusTextSink
from universal_data.pipeline.config import PipelineConfig, load_pipeline
from universal_data.pipeline.pipeline import Pipeline
from universal_data.rules.engine import BusinessRule, RuleEngine
from universal_data.schema.loader import load_schema
from universal_data.transformation.operations import SelectColumns


class TestPipelineBuilder:
    def test_rejects_unknown_error_mode(self) -> None:
        with pytest.raises(ConfigurationError, match="on_error"):
            Pipeline(on_error="explode")

    def test_requires_an_input(self) -> None:
        with pytest.raises(ConfigurationError, match="no input"):
            Pipeline().run()

    def test_runs_from_a_dataset(self, customers_dataset: Dataset) -> None:
        result = Pipeline("x").remove_duplicates().run(customers_dataset)
        assert result.dataset.n_rows == 5
        assert result.succeeded

    def test_runs_from_a_file(self, customers_csv: Path, tmp_path: Path) -> None:
        target = tmp_path / "out.parquet"
        result = (
            Pipeline("customers")
            .read(customers_csv)
            .clean(remove_duplicates=True)
            .write(target)
            .run()
        )
        assert target.exists()
        assert result.context.artifacts["outputs"][0]["rows"] == 5

    def test_from_dataset_and_callable(self, customers_dataset: Dataset) -> None:
        assert Pipeline().from_dataset(customers_dataset).run().dataset.n_rows == 6
        assert Pipeline().from_callable(lambda: customers_dataset).run().dataset.n_rows == 6

    def test_repr_lists_steps(self) -> None:
        pipeline = Pipeline("p").remove_duplicates().filter("age > 1")
        assert "remove_duplicates" in repr(pipeline)


class TestValidationStep:
    def test_report_mode_keeps_every_row(self, customers_dataset: Dataset, schema_file: Path) -> None:
        result = Pipeline("p").validate(load_schema(schema_file)).run(customers_dataset)
        assert result.dataset.n_rows == 6
        assert result.validation is not None
        assert result.validation.invalid_rows > 0

    def test_filter_mode_quarantines_bad_rows(
        self, customers_dataset: Dataset, schema_file: Path
    ) -> None:
        result = Pipeline("p").validate(load_schema(schema_file), mode="filter").run(customers_dataset)
        assert result.dataset.n_rows < 6
        assert result.context.artifacts["invalid_rows"].n_rows > 0

    def test_strict_mode_fails_the_run(self, customers_dataset: Dataset, schema_file: Path) -> None:
        pipeline = Pipeline("p").validate(load_schema(schema_file), mode="strict")
        with pytest.raises(PipelineError, match="validate"):
            pipeline.run(customers_dataset)

    def test_invalid_mode_is_rejected(self, schema_file: Path) -> None:
        with pytest.raises(ConfigurationError, match="validate mode"):
            Pipeline().validate(load_schema(schema_file), mode="nope")


class TestOtherSteps:
    def test_business_rules(self, customers_dataset: Dataset) -> None:
        engine = RuleEngine([BusinessRule(name="adult", condition="age >= 18")])
        result = Pipeline("p").apply_rules(engine, action="drop").run(customers_dataset)
        assert result.dataset.n_rows == 5
        assert result.rule_evaluation is not None

    def test_masking(self, customers_dataset: Dataset) -> None:
        result = Pipeline("p").mask({"email": "redact"}).run(customers_dataset)
        assert result.dataset.frame["email"].dropna().iloc[0] == "[REDACTED]"

    def test_mask_detected_pii(self, customers_dataset: Dataset) -> None:
        result = Pipeline("p").mask_detected_pii("redact").run(customers_dataset)
        assert result.dataset.frame["email"].dropna().iloc[0] == "[REDACTED]"

    def test_enrichment(self, customers_dataset: Dataset) -> None:
        from universal_data.enrichment.enrichers import DerivedColumnEnricher

        result = Pipeline("p").enrich(DerivedColumnEnricher("x", "age + 1")).run(customers_dataset)
        assert "x" in result.dataset.columns

    def test_transform_and_filter(self, customers_dataset: Dataset) -> None:
        result = (
            Pipeline("p")
            .transform(SelectColumns(["customer_id", "age"]))
            .filter("age > 30")
            .run(customers_dataset)
        )
        assert result.dataset.columns == ["customer_id", "age"]
        assert result.dataset.n_rows == 4

    def test_outlier_step(self, customers_dataset: Dataset) -> None:
        result = Pipeline("p").handle_outliers(action="report").run(customers_dataset)
        assert result.succeeded

    def test_quality_gate_passes(self, customers_dataset: Dataset) -> None:
        result = Pipeline("p").quality_check(minimum_score=0.1).run(customers_dataset)
        assert result.quality is not None

    def test_quality_gate_can_fail_the_run(self, customers_dataset: Dataset) -> None:
        pipeline = Pipeline("p").quality_check(minimum_score=0.999)
        with pytest.raises(PipelineError, match="below the configured minimum"):
            pipeline.run(customers_dataset)

    def test_report_step(self, customers_dataset: Dataset, tmp_path: Path) -> None:
        html = tmp_path / "report.html"
        json_report = tmp_path / "report.json"
        Pipeline("p").report(html).run(customers_dataset)
        Pipeline("p").report(json_report, fmt="json").run(customers_dataset)
        assert html.stat().st_size > 0
        assert json_report.stat().st_size > 0


class TestErrorHandling:
    def test_fail_fast_by_default(self, customers_dataset: Dataset) -> None:
        pipeline = Pipeline("p").filter("nonexistent > 1")
        with pytest.raises(PipelineError, match="filter"):
            pipeline.run(customers_dataset)

    def test_skip_mode_continues(self, customers_dataset: Dataset) -> None:
        pipeline = Pipeline("p", on_error="skip").filter("nonexistent > 1").remove_duplicates()
        result = pipeline.run(customers_dataset)
        assert "filter" in result.skipped
        assert result.failures
        assert result.dataset.n_rows == 5
        assert not result.succeeded

    def test_optional_step_does_not_stop_the_run(self, customers_dataset: Dataset) -> None:
        pipeline = Pipeline("p")
        pipeline.filter("nonexistent > 1")
        pipeline.steps[-1].optional = True
        result = pipeline.run(customers_dataset)
        assert "filter" in result.skipped

    def test_conditional_step(self, customers_dataset: Dataset) -> None:
        pipeline = Pipeline("p").remove_duplicates().when(lambda dataset: dataset.n_rows > 100)
        result = pipeline.run(customers_dataset)
        assert result.skipped == ["remove_duplicates"]
        assert result.dataset.n_rows == 6

    def test_when_requires_a_preceding_step(self) -> None:
        with pytest.raises(ConfigurationError, match="must follow a step"):
            Pipeline().when(lambda dataset: True)


class TestMetrics:
    def test_metrics_are_collected(self, customers_dataset: Dataset) -> None:
        result = Pipeline("p").remove_duplicates().run(customers_dataset)
        metrics = result.metrics
        assert metrics.rows_input == 6
        assert metrics.rows_output == 5
        assert metrics.status == "success"
        assert "Processing Summary" in metrics.summary()
        assert metrics.to_dict()["steps"][0]["name"] == "remove_duplicates"

    def test_sink_receives_the_run(self, customers_dataset: Dataset) -> None:
        sink = InMemorySink()
        Pipeline("p", metrics_sink=sink).remove_duplicates().run(customers_dataset)
        assert len(sink.runs) == 1

    def test_prometheus_rendering(self, customers_dataset: Dataset) -> None:
        sink = PrometheusTextSink()
        Pipeline("p", metrics_sink=sink).remove_duplicates().run(customers_dataset)
        rendered = sink.render()
        assert "universal_data_rows_output_total" in rendered
        assert 'pipeline="p"' in rendered

    def test_result_to_dict_and_document(self, customers_dataset: Dataset) -> None:
        result = Pipeline("p").quality_check().run(customers_dataset)
        assert result.to_dict()["rows"] == 6
        assert result.document().to_dict()["metrics"] is not None

    def test_memory_warning_threshold(self, customers_dataset: Dataset) -> None:
        pipeline = Pipeline("p", memory_warning_mb=0.0001).remove_duplicates()
        assert pipeline.run(customers_dataset).succeeded


class TestChunkedExecution:
    def test_streams_a_large_file(self, tmp_path: Path) -> None:
        source = tmp_path / "big.csv"
        pd.DataFrame({"a": range(100), "b": ["x"] * 100}).to_csv(source, index=False)
        target = tmp_path / "out.parquet"
        result = (
            Pipeline("chunked").clean(remove_duplicates=True).run_chunked(
                source, target, chunk_size=30
            )
        )
        assert target.exists()
        assert result.metrics.rows_input == 100
        assert result.metrics.rows_output == 100
        assert len(pd.read_parquet(target)) == 100

    def test_write_steps_are_skipped_in_chunked_mode(self, tmp_path: Path) -> None:
        source = tmp_path / "in.csv"
        pd.DataFrame({"a": range(10)}).to_csv(source, index=False)
        ignored = tmp_path / "ignored.csv"
        target = tmp_path / "out.csv"
        Pipeline("c").write(ignored).run_chunked(source, target, chunk_size=5)
        assert target.exists()
        assert not ignored.exists()

    def test_failure_in_a_chunk_stops_the_run(self, tmp_path: Path) -> None:
        source = tmp_path / "in.csv"
        pd.DataFrame({"a": range(10)}).to_csv(source, index=False)
        pipeline = Pipeline("c").filter("missing_column > 1")
        with pytest.raises(PipelineError, match="chunk 0"):
            pipeline.run_chunked(source, tmp_path / "out.csv", chunk_size=5)


class TestConfiguration:
    def _write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "pipeline.yaml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_full_configuration(
        self, tmp_path: Path, customers_csv: Path, schema_file: Path, rules_file: Path
    ) -> None:
        config = self._write(
            tmp_path,
            f"""
pipeline:
  name: customers
input:
  type: csv
  path: {customers_csv.name}
steps:
  - clean:
      remove_duplicates: true
      trim_whitespace: true
  - validate:
      schema: {schema_file.name}
      mode: report
  - rules:
      file: {rules_file.name}
  - mask:
      email: email
  - quality:
      minimum_score: 0.1
output:
  type: parquet
  path: out/result.parquet
report:
  path: out/report.html
""",
        )
        result = load_pipeline(config, workspace=tmp_path).run()
        assert (tmp_path / "out" / "result.parquet").exists()
        assert (tmp_path / "out" / "report.html").exists()
        assert result.validation is not None
        assert result.rule_evaluation is not None

    def test_transform_and_enrich_steps(self, tmp_path: Path, customers_csv: Path) -> None:
        lookup = tmp_path / "lookup.csv"
        pd.DataFrame({"country": ["IR", "DE", "FR"], "region": ["ME", "EU", "EU"]}).to_csv(
            lookup, index=False
        )
        config = self._write(
            tmp_path,
            f"""
pipeline:
  name: t
input:
  type: csv
  path: {customers_csv.name}
steps:
  - transform:
      - rename:
          age: years
      - filter: "years >= 18"
  - enrich:
      lookup:
        path: {lookup.name}
        on: country
  - enrich:
      derive:
        double_salary: "salary * 2"
output:
  type: csv
  path: out/result.csv
""",
        )
        result = load_pipeline(config, workspace=tmp_path).run()
        assert "region" in result.dataset.columns
        assert "double_salary" in result.dataset.columns
        assert "years" in result.dataset.columns

    def test_inline_rules_and_outliers(self, tmp_path: Path, customers_csv: Path) -> None:
        config = self._write(
            tmp_path,
            f"""
pipeline:
  name: t
input:
  type: csv
  path: {customers_csv.name}
steps:
  - rules:
      action: drop
      rules:
        - name: adult
          condition: "age >= 18"
  - outliers:
      columns: [salary]
      action: cap
  - remove_duplicates:
      subset: [customer_id]
  - filter: "salary >= 0"
output:
  type: csv
  path: out/result.csv
""",
        )
        result = load_pipeline(config, workspace=tmp_path).run()
        assert result.dataset.n_rows >= 1

    def test_unknown_top_level_key(self, tmp_path: Path) -> None:
        config = self._write(tmp_path, "pipeline:\n  name: x\nnonsense: 1\n")
        with pytest.raises(ConfigurationError, match="Unknown top-level key"):
            PipelineConfig.from_file(config)

    def test_unknown_step(self, tmp_path: Path, customers_csv: Path) -> None:
        config = self._write(
            tmp_path,
            f"input:\n  type: csv\n  path: {customers_csv.name}\nsteps:\n  - teleport: {{}}\n",
        )
        with pytest.raises(ConfigurationError, match="Unknown pipeline step"):
            PipelineConfig.from_file(config, workspace=tmp_path).build()

    def test_missing_input_section(self, tmp_path: Path) -> None:
        config = self._write(tmp_path, "pipeline:\n  name: x\n")
        with pytest.raises(ConfigurationError, match="no 'input' section"):
            PipelineConfig.from_file(config).build()

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        config = self._write(tmp_path, "pipeline: [unclosed")
        with pytest.raises(ConfigurationError, match="Could not parse"):
            PipelineConfig.from_file(config)

    def test_non_mapping_configuration(self, tmp_path: Path) -> None:
        config = self._write(tmp_path, "- a\n- b\n")
        with pytest.raises(ConfigurationError, match="must be a mapping"):
            PipelineConfig.from_file(config)

    def test_path_traversal_is_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        secret = tmp_path / "secret.csv"
        secret.write_text("a\n1\n", encoding="utf-8")
        config = workspace / "pipeline.yaml"
        config.write_text(
            "input:\n  type: csv\n  path: ../secret.csv\noutput:\n  type: csv\n  path: out.csv\n",
            encoding="utf-8",
        )
        with pytest.raises(SecurityError, match="escapes the allowed workspace"):
            PipelineConfig.from_file(config, workspace=workspace).build()

    def test_environment_references_are_resolved(
        self, tmp_path: Path, customers_csv: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INPUT_FILE", customers_csv.name)
        config = self._write(
            tmp_path,
            "input:\n  type: csv\n  path: ${INPUT_FILE}\noutput:\n  type: csv\n  path: out.csv\n",
        )
        result = load_pipeline(config, workspace=tmp_path).run()
        assert result.dataset.n_rows == 6

    def test_yaml_boolean_keys_are_normalised(self, tmp_path: Path, customers_csv: Path) -> None:
        lookup = tmp_path / "lookup.csv"
        pd.DataFrame({"country": ["IR", "DE", "FR"], "region": ["ME", "EU", "EU"]}).to_csv(
            lookup, index=False
        )
        # `on:` is parsed as the boolean True by YAML 1.1 unless it is normalised.
        config = self._write(
            tmp_path,
            f"""
input:
  type: csv
  path: {customers_csv.name}
steps:
  - enrich:
      lookup:
        path: {lookup.name}
        on: country
output:
  type: csv
  path: out.csv
""",
        )
        result = load_pipeline(config, workspace=tmp_path).run()
        assert "region" in result.dataset.columns

    def test_database_output_configuration(self, tmp_path: Path, customers_csv: Path) -> None:
        config = self._write(
            tmp_path,
            f"""
input:
  type: csv
  path: {customers_csv.name}
output:
  type: database
  table: customers
  if_exists: replace
  connection:
    driver: sqlite
    database: {(tmp_path / 'out.db').as_posix()}
""",
        )
        result = load_pipeline(config, workspace=tmp_path).run()
        assert result.context.artifacts["outputs"][0]["table"] == "customers"

    def test_database_input_configuration(self, tmp_path: Path, customers_frame: pd.DataFrame) -> None:
        import sqlite3

        database = tmp_path / "in.db"
        with sqlite3.connect(database) as connection:
            customers_frame.to_sql("customers", connection, index=False)
        config = self._write(
            tmp_path,
            f"""
input:
  type: database
  table: customers
  connection:
    driver: sqlite
    database: {database.as_posix()}
output:
  type: csv
  path: out.csv
""",
        )
        result = load_pipeline(config, workspace=tmp_path).run()
        assert result.dataset.n_rows == 6

    def test_database_section_requires_a_connection(self, tmp_path: Path) -> None:
        config = self._write(tmp_path, "input:\n  type: database\n  table: x\n")
        with pytest.raises(ConfigurationError, match="'connection' mapping"):
            PipelineConfig.from_file(config, workspace=tmp_path).build()

    def test_api_input_requires_a_url(self, tmp_path: Path) -> None:
        config = self._write(tmp_path, "input:\n  type: api\n")
        with pytest.raises(ConfigurationError, match="needs a 'url'"):
            PipelineConfig.from_file(config, workspace=tmp_path).build()

    def test_api_input_rejects_unknown_auth(self, tmp_path: Path) -> None:
        config = self._write(
            tmp_path,
            "input:\n  type: api\n  url: https://api.test/x\n  auth:\n    type: kerberos\n",
        )
        with pytest.raises(ConfigurationError, match="Unsupported API auth"):
            PipelineConfig.from_file(config, workspace=tmp_path).build()

    def test_validate_step_requires_a_schema(self, tmp_path: Path, customers_csv: Path) -> None:
        config = self._write(
            tmp_path,
            f"input:\n  type: csv\n  path: {customers_csv.name}\nsteps:\n  - validate: {{}}\n",
        )
        with pytest.raises(ConfigurationError, match="needs a 'schema'"):
            PipelineConfig.from_file(config, workspace=tmp_path).build()

    def test_enrich_step_requires_a_source(self, tmp_path: Path, customers_csv: Path) -> None:
        config = self._write(
            tmp_path,
            f"input:\n  type: csv\n  path: {customers_csv.name}\nsteps:\n  - enrich: {{}}\n",
        )
        with pytest.raises(ConfigurationError, match="lookup"):
            PipelineConfig.from_file(config, workspace=tmp_path).build()

    def test_rules_step_requires_a_source(self, tmp_path: Path, customers_csv: Path) -> None:
        config = self._write(
            tmp_path,
            f"input:\n  type: csv\n  path: {customers_csv.name}\nsteps:\n  - rules: {{}}\n",
        )
        with pytest.raises(ConfigurationError, match="either 'file' or"):
            PipelineConfig.from_file(config, workspace=tmp_path).build()

    def test_step_shorthand_without_options(self, tmp_path: Path, customers_csv: Path) -> None:
        config = self._write(
            tmp_path,
            f"input:\n  type: csv\n  path: {customers_csv.name}\n"
            "steps:\n  - remove_duplicates\noutput:\n  type: csv\n  path: out.csv\n",
        )
        result = load_pipeline(config, workspace=tmp_path).run()
        assert result.dataset.n_rows == 5
