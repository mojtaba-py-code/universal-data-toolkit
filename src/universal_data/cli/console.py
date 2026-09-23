"""Console helpers shared by the CLI commands."""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from universal_data.core.exceptions import DataToolkitError

console = Console()
error_console = Console(stderr=True)

EXIT_ERROR = 1
EXIT_INVALID_DATA = 2


def print_json(payload: Any) -> None:
    console.print_json(json.dumps(payload, ensure_ascii=False, default=str))


def print_table(title: str, columns: list[str], rows: list[list[Any]]) -> None:
    table = Table(title=title, header_style="bold", show_lines=False)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    console.print(table)


MAX_PREVIEW_COLUMNS = 8


def print_frame(frame: Any, *, title: str = "", limit: int = 10) -> None:
    """Render the first rows of a DataFrame as a table.

    Wide frames are truncated horizontally; squeezing 40 columns into a terminal
    produces something unreadable rather than something informative.
    """
    columns = list(frame.columns)[:MAX_PREVIEW_COLUMNS]
    preview = frame.loc[:, columns].head(limit)
    table = Table(title=title or None, header_style="bold")
    for column in columns:
        table.add_column(str(column), overflow="ellipsis", max_width=20)
    for _, row in preview.iterrows():
        table.add_row(*["" if value is None else str(value) for value in row.tolist()])
    console.print(table)
    notes = []
    if len(frame) > limit:
        notes.append(f"{len(frame) - limit:,} more rows")
    if len(frame.columns) > len(columns):
        notes.append(f"{len(frame.columns) - len(columns)} more columns")
    if notes:
        console.print(f"[dim]... {', '.join(notes)}[/dim]")


def fail(message: str, code: int = EXIT_ERROR) -> None:
    error_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code)


def handle_error(exc: Exception) -> None:
    """Turn an exception into a clean CLI error message."""
    if isinstance(exc, DataToolkitError):
        fail(str(exc))
    fail(f"{type(exc).__name__}: {exc}")


def success(message: str) -> None:
    console.print(f"[green]OK[/green] {message}")


def warn(message: str) -> None:
    console.print(f"[yellow]![/yellow] {message}")
