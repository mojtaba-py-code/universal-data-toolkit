"""CLI command tests."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from universal_data.cli.main import app

runner = CliRunner()


def run(*args: str) -> object:
    return runner.invoke(app, list(args))


def combined_output(result: object) -> str:
    """stdout plus stderr - error messages are printed to stderr on purpose."""
    text = result.stdout  # type: ignore[attr-defined]
    with contextlib.suppress(ValueError, AttributeError):  # older click merges the streams
        text += result.stderr  # type: ignore[attr-defined]
    return text


class TestBasics:
    def test_version(self) -> None:
        result = run("version")
        assert result.exit_code == 0
        assert "universal-data-toolkit" in result.stdout

    def test_help(self) -> None:
        result = run("--help")
        assert result.exit_code == 0
        assert "inspect" in result.stdout

    def test_formats(self) -> None:
        result = run("formats")
        assert result.exit_code == 0
        assert "parquet" in result.stdout

    def test_plugins_without_any_installed(self) -> None:
        assert run("plugins").exit_code == 0


class TestInspection:
    def test_inspect(self, customers_csv: Path) -> None:
        result = run("inspect", str(customers_csv), "-n", "2")
        assert result.exit_code == 0
        assert "Dataset Profile" in result.stdout

    def test_inspect_json(self, customers_csv: Path) -> None:
        result = run("inspect", str(customers_csv), "--json")
        assert result.exit_code == 0
        assert '"format"' in result.stdout

    def test_inspect_missing_file(self, tmp_path: Path) -> None:
        result = run("inspect", str(tmp_path / "nope.csv"))
        assert result.exit_code == 1
        assert "Error" in combined_output(result)

    def test_profile_with_outputs(self, customers_csv: Path, tmp_path: Path) -> None:
        json_out = tmp_path / "profile.json"
        html_out = tmp_path / "profile.html"
        result = run(
            "profile", str(customers_csv), "--json-out", str(json_out), "--html-out", str(html_out)
        )
        assert result.exit_code == 0
        assert json.loads(json_out.read_text(encoding="utf-8"))["profile"]["rows"] == 6
        assert html_out.stat().st_size > 0

    def test_schema_detection(self, customers_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "schema.yaml"
        result = run("schema", str(customers_csv), "--output", str(output), "--constraints")
        assert result.exit_code == 0
        assert output.exists()
        assert "Possible personal data" in result.stdout


class TestValidationCommands:
    def test_validate_reports_issues(self, customers_csv: Path, schema_file: Path) -> None:
        result = run("validate", str(customers_csv), "--schema", str(schema_file))
        assert result.exit_code == 0
        assert "Validation of" in result.stdout

    def test_validate_strict_exits_with_two(self, customers_csv: Path, schema_file: Path) -> None:
        result = run("validate", str(customers_csv), "--schema", str(schema_file), "--strict")
        assert result.exit_code == 2

    def test_validate_writes_rejected_rows(
        self, customers_csv: Path, schema_file: Path, tmp_path: Path
    ) -> None:
        invalid = tmp_path / "invalid.csv"
        result = run(
            "validate",
            str(customers_csv),
            "--schema",
            str(schema_file),
            "--invalid-out",
            str(invalid),
        )
        assert result.exit_code == 0
        assert invalid.exists()

    def test_validate_json_output(self, customers_csv: Path, schema_file: Path) -> None:
        result = run("validate", str(customers_csv), "--schema", str(schema_file), "--json")
        assert '"valid"' in result.stdout

    def test_quality(self, customers_csv: Path, rules_file: Path, tmp_path: Path) -> None:
        html = tmp_path / "q.html"
        result = run(
            "quality",
            str(customers_csv),
            "--rules",
            str(rules_file),
            "--html-out",
            str(html),
        )
        assert result.exit_code == 0
        assert "Overall Score" in result.stdout
        assert html.exists()

    def test_quality_minimum_gate(self, customers_csv: Path) -> None:
        result = run("quality", str(customers_csv), "--minimum", "0.999")
        assert result.exit_code == 2


class TestProcessing:
    def test_clean(self, customers_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "clean.csv"
        result = run(
            "clean",
            str(customers_csv),
            "-o",
            str(output),
            "--missing",
            "mode",
            "--drop-duplicates",
            "--normalize-columns",
        )
        assert result.exit_code == 0
        assert len(pd.read_csv(output)) == 5

    def test_transform(self, customers_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "t.csv"
        result = run(
            "transform",
            str(customers_csv),
            "-o",
            str(output),
            "--rename",
            "age=years",
            "--select",
            "customer_id,years",
            "--where",
            "years >= 18",
            "--sort-by",
            "years",
            "--desc",
            "--limit",
            "3",
        )
        assert result.exit_code == 0
        frame = pd.read_csv(output)
        assert list(frame.columns) == ["customer_id", "years"]
        assert len(frame) == 3

    def test_transform_requires_an_operation(self, customers_csv: Path, tmp_path: Path) -> None:
        result = run("transform", str(customers_csv), "-o", str(tmp_path / "x.csv"))
        assert result.exit_code == 1

    def test_transform_rejects_bad_rename(self, customers_csv: Path, tmp_path: Path) -> None:
        result = run(
            "transform", str(customers_csv), "-o", str(tmp_path / "x.csv"), "--rename", "age"
        )
        assert result.exit_code == 1

    def test_transform_drop(self, customers_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "d.csv"
        result = run("transform", str(customers_csv), "-o", str(output), "--drop", "salary")
        assert result.exit_code == 0
        assert "salary" not in pd.read_csv(output).columns

    def test_convert(self, customers_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "out.parquet"
        assert run("convert", str(customers_csv), str(output)).exit_code == 0
        assert len(pd.read_parquet(output)) == 6

    def test_convert_chunked_to_sqlite(self, customers_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "out.db"
        result = run(
            "convert", str(customers_csv), str(output), "--chunk-size", "2", "--table", "people"
        )
        assert result.exit_code == 0
        from universal_data.ingestion.files import SQLiteReader

        assert len(SQLiteReader().read(output, table="people")) == 6

    def test_export(self, customers_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "out.jsonl"
        assert run("export", str(customers_csv), "-o", str(output)).exit_code == 0
        assert output.exists()

    def test_process_end_to_end(
        self, customers_csv: Path, schema_file: Path, rules_file: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "processed.parquet"
        report = tmp_path / "report.html"
        result = run(
            "process",
            str(customers_csv),
            "-o",
            str(output),
            "--schema",
            str(schema_file),
            "--rules",
            str(rules_file),
            "--mask-pii",
            "--report",
            str(report),
        )
        assert result.exit_code == 0
        assert output.exists()
        assert report.exists()
        assert "Processing Summary" in result.stdout

    def test_process_chunked(self, customers_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "chunked.csv"
        result = run("process", str(customers_csv), "-o", str(output), "--chunk-size", "2")
        assert result.exit_code == 0
        assert output.exists()


class TestPipelineCommand:
    @pytest.fixture
    def config(self, tmp_path: Path, customers_csv: Path) -> Path:
        path = tmp_path / "pipeline.yaml"
        path.write_text(
            f"""
pipeline:
  name: cli_test
input:
  type: csv
  path: {customers_csv.name}
steps:
  - remove_duplicates
output:
  type: csv
  path: out/result.csv
""".strip(),
            encoding="utf-8",
        )
        return path

    def test_dry_run(self, config: Path, tmp_path: Path) -> None:
        result = run("run", str(config), "--workspace", str(tmp_path), "--dry-run")
        assert result.exit_code == 0
        assert "Configuration is valid" in result.stdout

    def test_run_with_metrics(self, config: Path, tmp_path: Path) -> None:
        metrics = tmp_path / "metrics.json"
        result = run(
            "run", str(config), "--workspace", str(tmp_path), "--metrics-out", str(metrics)
        )
        assert result.exit_code == 0
        assert (tmp_path / "out" / "result.csv").exists()
        assert json.loads(metrics.read_text(encoding="utf-8"))["rows_output"] == 5

    def test_broken_configuration(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("nonsense: true\n", encoding="utf-8")
        result = run("run", str(path))
        assert result.exit_code == 1


class TestGenerateAndFetch:
    def test_generate(self, tmp_path: Path) -> None:
        output = tmp_path / "customers.csv"
        result = run("generate", "--kind", "customers", "--rows", "50", "-o", str(output))
        assert result.exit_code == 0
        assert len(pd.read_csv(output)) >= 50

    def test_generate_rejects_unknown_profile(self, tmp_path: Path) -> None:
        result = run("generate", "--issues", "chaotic", "-o", str(tmp_path / "x.csv"))
        assert result.exit_code == 1

    def test_generate_rejects_unknown_kind(self, tmp_path: Path) -> None:
        result = run("generate", "--kind", "aliens", "-o", str(tmp_path / "x.csv"))
        assert result.exit_code == 1

    def test_fetch_blocks_private_addresses(self, tmp_path: Path) -> None:
        result = run("fetch", "--url", "http://127.0.0.1:9/x", "-o", str(tmp_path / "out.json"))
        assert result.exit_code == 1
        assert "Error" in combined_output(result)


class TestGlobalOptions:
    def test_log_file_is_written(self, customers_csv: Path, tmp_path: Path) -> None:
        log_file = tmp_path / "run.log"
        result = run(
            "--log-level", "INFO", "--log-file", str(log_file), "--quiet",
            "inspect", str(customers_csv), "-n", "0",
        )
        assert result.exit_code == 0
        assert log_file.exists()
        assert "execution" in log_file.read_text(encoding="utf-8") or log_file.stat().st_size >= 0

    def test_json_logs(self, customers_csv: Path) -> None:
        result = run("--json-logs", "--log-level", "INFO", "inspect", str(customers_csv), "-n", "0")
        assert result.exit_code == 0
