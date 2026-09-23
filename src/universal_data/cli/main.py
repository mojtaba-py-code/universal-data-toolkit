"""The ``data-tool`` command line interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer

from universal_data import __version__
from universal_data.cli.console import (
    console,
    handle_error,
    print_frame,
    print_json,
    print_table,
    success,
    warn,
)
from universal_data.core.dataset import Dataset
from universal_data.core.exceptions import DataToolkitError
from universal_data.observability.logging import configure_logging
from universal_data.security.paths import PathPolicy
from universal_data.security.secrets import load_dotenv

app = typer.Typer(
    name="data-tool",
    help="Universal data processing toolkit: inspect, validate, clean, transform and export data.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def configure(
    log_level: str = typer.Option("WARNING", "--log-level", help="Logging level."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit logs as JSON."),
    log_file: Path | None = typer.Option(None, "--log-file", help="Also write logs to this file."),
    env_file: Path = typer.Option(
        Path(".env"), "--env-file", help="Load environment variables from this file if it exists."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress console logging."),
) -> None:
    """Global options applied before any command runs."""
    load_dotenv(env_file)
    configure_logging(log_level, json_output=json_logs, log_file=log_file, quiet=quiet)


# --------------------------------------------------------------------------
# inspection
# --------------------------------------------------------------------------


@app.command()
def inspect(
    path: Path = typer.Argument(..., help="File to inspect."),
    rows: int = typer.Option(5, "--rows", "-n", help="Preview rows to show."),
    as_json: bool = typer.Option(False, "--json", help="Print machine readable output."),
) -> None:
    """Detect the format of a file and summarise its contents."""
    from universal_data.ingestion.detect import inspect_file

    try:
        signature = inspect_file(path)
        dataset = Dataset.read(path)
        profile = dataset.profile(with_outliers=False)
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)
        return

    if as_json:
        print_json({"file": signature.to_dict(), "profile": profile.to_dict()})
        return

    print_table(
        "File",
        ["property", "value"],
        [
            ["path", signature.path],
            ["format", signature.format],
            ["size", f"{signature.size_bytes:,} bytes"],
            ["encoding", signature.encoding or "-"],
            ["delimiter", repr(signature.delimiter) if signature.delimiter else "-"],
            ["detected from", signature.detected_from],
        ],
    )
    console.print(profile.summary())
    if rows:
        console.print()
        print_frame(dataset.frame, title="Preview", limit=rows)


@app.command()
def profile(
    path: Path = typer.Argument(..., help="File to profile."),
    json_out: Path | None = typer.Option(None, "--json-out", help="Write the profile as JSON."),
    html_out: Path | None = typer.Option(None, "--html-out", help="Write an HTML report."),
) -> None:
    """Produce a full dataset profile with per-column statistics."""
    from universal_data.quality.report import QualityDocument

    try:
        dataset = Dataset.read(path)
        report = dataset.profile()
        console.print(report.summary())
        if json_out or html_out:
            document = QualityDocument(
                dataset=dataset.name,
                source=dataset.source,
                profile=report,
                quality=dataset.quality(),
            )
            if json_out:
                document.to_json(json_out)
                success(f"JSON profile written to {json_out}")
            if html_out:
                document.to_html(html_out)
                success(f"HTML report written to {html_out}")
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)


@app.command()
def schema(
    path: Path = typer.Argument(..., help="File to derive a schema from."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write the schema to YAML."),
    constraints: bool = typer.Option(
        False, "--constraints", help="Infer min/max and enum values from the data."
    ),
) -> None:
    """Detect a schema, including likely PII columns."""
    from universal_data.schema.loader import save_schema

    try:
        dataset = Dataset.read(path)
        detected = dataset.detect_schema(infer_constraints=constraints)
        console.print(detected.describe())
        if detected.pii_columns:
            warn(f"Possible personal data in: {', '.join(detected.pii_columns)}")
        if output:
            save_schema(detected, output)
            success(f"Schema written to {output}")
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)


# --------------------------------------------------------------------------
# validation and quality
# --------------------------------------------------------------------------


@app.command()
def validate(
    path: Path = typer.Argument(..., help="File to validate."),
    schema_path: Path = typer.Option(..., "--schema", "-s", help="Schema file (YAML or JSON)."),
    strict: bool = typer.Option(False, "--strict", help="Exit with an error when validation fails."),
    invalid_out: Path | None = typer.Option(
        None, "--invalid-out", help="Write the rejected rows to this file."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print machine readable output."),
) -> None:
    """Validate a dataset against a schema."""
    from universal_data.schema.loader import load_schema

    try:
        dataset = Dataset.read(path)
        result = dataset.validate(load_schema(schema_path))
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)
        return

    if as_json:
        print_json(result.to_dict())
    else:
        console.print(result.summary())

    if invalid_out and result.invalid_rows:
        invalid = result.invalid_frame(dataset.frame)
        Dataset(invalid, name=f"{dataset.name}_invalid").write(invalid_out)
        success(f"{len(invalid):,} rejected rows written to {invalid_out}")

    if strict and not result.is_valid:
        raise typer.Exit(2)


@app.command()
def quality(
    path: Path = typer.Argument(..., help="File to analyse."),
    schema_path: Path | None = typer.Option(None, "--schema", "-s", help="Optional schema."),
    rules_path: Path | None = typer.Option(None, "--rules", "-r", help="Optional rules YAML."),
    html_out: Path | None = typer.Option(None, "--html-out", help="Write an HTML report."),
    json_out: Path | None = typer.Option(None, "--json-out", help="Write a JSON report."),
    minimum: float | None = typer.Option(
        None, "--minimum", help="Exit with an error when the score is below this value (0-1)."
    ),
) -> None:
    """Score a dataset across the six quality dimensions."""
    try:
        dataset = Dataset.read(path)
        schema = _load_schema(schema_path)
        rules = _load_rules(rules_path)
        report = dataset.quality(schema, rules)
        console.print(report.summary())
        if html_out or json_out:
            document = dataset.report(schema=schema, rules=rules)
            if html_out:
                document.to_html(html_out)
                success(f"HTML report written to {html_out}")
            if json_out:
                document.to_json(json_out)
                success(f"JSON report written to {json_out}")
        if minimum is not None and report.overall_score < minimum:
            console.print(
                f"[red]Quality score {report.overall_score:.3f} is below the minimum {minimum}[/red]"
            )
            raise typer.Exit(2)
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)


# --------------------------------------------------------------------------
# processing
# --------------------------------------------------------------------------


@app.command()
def clean(
    path: Path = typer.Argument(..., help="File to clean."),
    output: Path = typer.Option(..., "--output", "-o", help="Where to write the result."),
    missing: str = typer.Option(
        "none", "--missing", help="Missing-value strategy: none, drop_rows, mean, median, mode, ffill, bfill."
    ),
    duplicates: bool = typer.Option(False, "--drop-duplicates", help="Remove duplicate rows."),
    normalize_columns: bool = typer.Option(
        False, "--normalize-columns", help="Rename columns to snake_case."
    ),
    case: str | None = typer.Option(None, "--case", help="Text case: lower, upper or title."),
) -> None:
    """Clean a dataset and write the result."""
    try:
        dataset = Dataset.read(path)
        cleaned = dataset.clean(
            missing_values=missing,
            remove_duplicates=duplicates,
            normalize_columns=normalize_columns,
            case=case,
        )
        report = cleaned.last_cleaning_report()
        if report:
            console.print(report.summary())
        rows = cleaned.write(output)
        success(f"{rows:,} rows written to {output}")
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)


@app.command()
def transform(
    path: Path = typer.Argument(..., help="Input file."),
    output: Path = typer.Option(..., "--output", "-o", help="Output file."),
    select: str | None = typer.Option(None, "--select", help="Comma separated columns to keep."),
    drop: str | None = typer.Option(None, "--drop", help="Comma separated columns to remove."),
    rename: list[str] = typer.Option(
        [], "--rename", help="Rename a column, given as old=new. Repeatable."
    ),
    where: str | None = typer.Option(None, "--where", help="Row filter expression."),
    sort_by: str | None = typer.Option(None, "--sort-by", help="Comma separated sort columns."),
    descending: bool = typer.Option(False, "--desc", help="Sort in descending order."),
    limit: int | None = typer.Option(None, "--limit", help="Keep only the first N rows."),
) -> None:
    """Apply a chain of transformations to a dataset."""
    from universal_data.transformation.operations import (
        DropColumns,
        FilterRows,
        LimitRows,
        RenameColumns,
        SelectColumns,
        SortRows,
    )

    steps: list[Any] = []
    if rename:
        mapping = {}
        for item in rename:
            if "=" not in item:
                handle_error(ValueError(f"--rename expects old=new, got {item!r}"))
            old, _, new = item.partition("=")
            mapping[old.strip()] = new.strip()
        steps.append(RenameColumns(mapping))
    if select:
        steps.append(SelectColumns([c.strip() for c in select.split(",") if c.strip()]))
    if drop:
        steps.append(DropColumns([c.strip() for c in drop.split(",") if c.strip()]))
    if where:
        steps.append(FilterRows(where))
    if sort_by:
        steps.append(
            SortRows([c.strip() for c in sort_by.split(",") if c.strip()], ascending=not descending)
        )
    if limit:
        steps.append(LimitRows(limit))
    if not steps:
        handle_error(ValueError("No transformation was requested"))

    try:
        dataset = Dataset.read(path)
        result = dataset.transform(*steps)
        rows = result.write(output)
        success(f"{rows:,} rows written to {output}")
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)


@app.command()
def convert(
    path: Path = typer.Argument(..., help="Input file."),
    output: Path = typer.Argument(..., help="Output file; the format follows its extension."),
    chunk_size: int | None = typer.Option(
        None, "--chunk-size", help="Stream the conversion in chunks of this many rows."
    ),
    table: str = typer.Option("data", "--table", help="Table name when writing to SQLite."),
) -> None:
    """Convert a file from one format to another."""
    from universal_data.export.base import writers

    try:
        if chunk_size:
            writer_cls = writers.for_path(output)
            writer = writer_cls()
            options = {"table": table} if output.suffix.lower() in (".db", ".sqlite", ".sqlite3") else {}
            rows = writer.write_chunks(
                (chunk.frame for chunk in Dataset.read_chunks(path, chunk_size)), output, **options
            )
        else:
            dataset = Dataset.read(path)
            options = {"table": table} if output.suffix.lower() in (".db", ".sqlite", ".sqlite3") else {}
            rows = dataset.write(output, **options)
        success(f"{rows:,} rows written to {output}")
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)


@app.command("export")
def export_command(
    path: Path = typer.Argument(..., help="Input file."),
    output: Path = typer.Option(..., "--output", "-o", help="Destination file."),
    fmt: str | None = typer.Option(None, "--format", "-f", help="Force an output format."),
    table: str = typer.Option("data", "--table", help="Table name for SQLite output."),
) -> None:
    """Export a dataset to another format or into a SQLite database."""
    try:
        dataset = Dataset.read(path)
        options = {"table": table} if (fmt == "sqlite" or output.suffix.lower() in (".db", ".sqlite")) else {}
        rows = dataset.write(output, fmt, **options)
        success(f"{rows:,} rows written to {output}")
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)


@app.command()
def process(
    path: Path = typer.Argument(..., help="Input file."),
    output: Path = typer.Option(..., "--output", "-o", help="Output file."),
    schema_path: Path | None = typer.Option(None, "--schema", "-s", help="Schema to validate against."),
    rules_path: Path | None = typer.Option(None, "--rules", "-r", help="Business rules YAML."),
    missing: str = typer.Option("none", "--missing", help="Missing-value strategy."),
    duplicates: bool = typer.Option(True, "--drop-duplicates/--keep-duplicates"),
    mask_pii: bool = typer.Option(False, "--mask-pii", help="Mask detected personal data."),
    chunk_size: int | None = typer.Option(
        None, "--chunk-size", help="Process the file in chunks of this many rows."
    ),
    report_out: Path | None = typer.Option(None, "--report", help="Write an HTML quality report."),
) -> None:
    """Run a standard clean/validate/export pipeline in one command."""
    from universal_data.pipeline.pipeline import Pipeline

    try:
        pipeline = Pipeline(name=path.stem)
        pipeline.clean(missing_values=missing, remove_duplicates=duplicates)
        if schema_path:
            pipeline.validate(_load_schema(schema_path), mode="report")
        if rules_path:
            pipeline.apply_rules(_load_rules(rules_path))
        if mask_pii:
            pipeline.mask_detected_pii()
        pipeline.quality_check()

        if chunk_size:
            if report_out:
                warn("Reports are not generated in chunked mode")
            result = pipeline.run_chunked(path, output, chunk_size=chunk_size)
        else:
            pipeline.read(path)
            pipeline.write(output)
            if report_out:
                pipeline.report(report_out)
            result = pipeline.run()

        console.print(result.metrics.summary())
        if result.quality:
            console.print()
            console.print(result.quality.summary())
        if not result.succeeded:
            raise typer.Exit(2)
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)


@app.command("run")
def run_pipeline(
    config: Path = typer.Argument(..., help="Pipeline configuration (YAML)."),
    workspace: Path | None = typer.Option(
        None, "--workspace", help="Restrict every file access to this directory."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only validate the configuration."),
    metrics_out: Path | None = typer.Option(
        None, "--metrics-out", help="Write run metrics as JSON."
    ),
) -> None:
    """Run a configuration-driven pipeline."""
    import json

    from universal_data.pipeline.config import PipelineConfig

    try:
        pipeline = PipelineConfig.from_file(config, workspace=workspace).build()
        if dry_run:
            success(f"Configuration is valid: {pipeline!r}")
            return
        result = pipeline.run()
        console.print(result.metrics.summary())
        if result.validation:
            console.print()
            console.print(result.validation.summary())
        if metrics_out:
            PathPolicy().resolve_output(metrics_out).write_text(
                json.dumps(result.metrics.to_dict(), indent=2), encoding="utf-8"
            )
            success(f"Metrics written to {metrics_out}")
        if not result.succeeded:
            raise typer.Exit(2)
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)


# --------------------------------------------------------------------------
# sources and utilities
# --------------------------------------------------------------------------


@app.command()
def fetch(
    url: str = typer.Option(..., "--url", help="Endpoint to fetch."),
    output: Path = typer.Option(..., "--output", "-o", help="Where to write the response data."),
    token_env: str | None = typer.Option(
        None, "--token-env", help="Environment variable holding a bearer token."
    ),
    records_path: str | None = typer.Option(
        None, "--records-path", help="Dotted path to the record list inside the response."
    ),
    pages: int = typer.Option(1, "--pages", help="Maximum number of pages to request."),
    page_param: str = typer.Option("page", "--page-param"),
    max_records: int | None = typer.Option(None, "--max-records"),
    allow_http: bool = typer.Option(False, "--allow-http", help="Permit plain HTTP URLs."),
    allow_private: bool = typer.Option(
        False, "--allow-private", help="Permit private/loopback addresses (local testing)."
    ),
) -> None:
    """Fetch JSON records from an HTTP API and save them."""
    from universal_data.ingestion.api import (
        APIClient,
        APIConfig,
        BearerTokenAuth,
        NoAuth,
        PageNumberPaginator,
    )
    from universal_data.security.network import URLPolicy
    from universal_data.security.secrets import SecretResolver

    try:
        auth: Any = NoAuth()
        if token_env:
            token = SecretResolver().get(token_env, required=True)
            assert token is not None
            auth = BearerTokenAuth(token)
        config = APIConfig(url=url, records_path=records_path, max_records=max_records)
        paginator = PageNumberPaginator(page_param=page_param, max_pages=pages) if pages > 1 else None
        policy = URLPolicy(allow_http=allow_http, allow_private_networks=allow_private)
        with APIClient(config, auth=auth, paginator=paginator, url_policy=policy) as client:
            dataset = Dataset.from_api(client)
        rows = dataset.write(output)
        success(f"{rows:,} records written to {output}")
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)


@app.command()
def generate(
    kind: str = typer.Option(
        "customers", "--kind", help="customers, products, orders, transactions or employees."
    ),
    rows: int = typer.Option(1_000, "--rows", "-n"),
    output: Path = typer.Option(..., "--output", "-o"),
    issues: str = typer.Option(
        "realistic", "--issues", help="Defect profile: clean, realistic or severe."
    ),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Generate a synthetic dataset, optionally with realistic defects."""
    from universal_data.generator.synthetic import QualityIssues, SyntheticDataGenerator

    profiles = {
        "clean": QualityIssues.clean(),
        "realistic": QualityIssues.realistic(),
        "severe": QualityIssues.severe(),
    }
    if issues not in profiles:
        handle_error(ValueError(f"--issues must be one of {sorted(profiles)}"))
    try:
        frame = SyntheticDataGenerator(seed).generate(kind, rows, profiles[issues])
        written = Dataset(frame, name=kind).write(output)
        success(f"{written:,} rows written to {output}")
    except (DataToolkitError, OSError, ValueError) as exc:
        handle_error(exc)


@app.command()
def benchmark(
    rows: int = typer.Option(200_000, "--rows", "-n", help="Rows to benchmark with."),
    chunk_size: int = typer.Option(50_000, "--chunk-size"),
    output_dir: Path = typer.Option(
        Path("benchmark_output"), "--output-dir", help="Directory for temporary artefacts."
    ),
) -> None:
    """Compare in-memory, chunked and vectorised processing."""
    from universal_data.cli.benchmarks import run_benchmarks

    try:
        results = run_benchmarks(rows=rows, chunk_size=chunk_size, output_dir=output_dir)
    except (DataToolkitError, OSError) as exc:
        handle_error(exc)
        return
    print_table(
        f"Benchmark ({rows:,} rows)",
        ["scenario", "seconds", "rows/sec", "peak MB"],
        [
            [
                item["scenario"],
                f"{item['seconds']:.3f}",
                f"{item['throughput']:,.0f}",
                f"{item['peak_mb']:.1f}",
            ]
            for item in results
        ],
    )


@app.command()
def plugins() -> None:
    """List installed plugins."""
    from universal_data.plugins.loader import load_plugins

    loaded = load_plugins()
    if not loaded:
        console.print("No plugins are installed.")
        return
    print_table(
        "Plugins",
        ["name", "source", "status"],
        [[item.name, item.source, "loaded" if item.loaded else item.error] for item in loaded],
    )


@app.command()
def formats() -> None:
    """Show the supported input and output formats."""
    from universal_data.export.base import writers
    from universal_data.ingestion.base import readers

    print_table(
        "Supported formats",
        ["direction", "formats"],
        [
            ["read", ", ".join(readers.formats())],
            ["write", ", ".join(writers.formats())],
        ],
    )


@app.command()
def version() -> None:
    """Print the toolkit version."""
    console.print(f"universal-data-toolkit {__version__}")


# --------------------------------------------------------------------------


def _load_schema(path: Path | None) -> Any:
    if path is None:
        return None
    from universal_data.schema.loader import load_schema

    return load_schema(path)


def _load_rules(path: Path | None) -> Any:
    if path is None:
        return None
    import yaml

    from universal_data.rules.engine import RuleEngine

    resolved = PathPolicy().resolve_input(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    definitions = payload.get("rules", payload) if isinstance(payload, dict) else payload
    if not isinstance(definitions, list):
        raise DataToolkitError("The rules file must contain a list of rules", path=str(resolved))
    return RuleEngine.from_config(definitions)


def main() -> None:
    """Console script entry point."""
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        console.print("[yellow]Interrupted[/yellow]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
