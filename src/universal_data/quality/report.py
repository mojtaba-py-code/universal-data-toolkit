"""Report generation (JSON and standalone HTML).

Everything that goes into the HTML is escaped: report input is data, and data
from an untrusted CSV must never become markup.  The page has no external assets
so it can be attached to an email or committed next to the output file.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from universal_data.core.types import PathLike
from universal_data.security.paths import PathPolicy

if TYPE_CHECKING:  # pragma: no cover - avoids a cycle with the profiler
    from universal_data.cleaning.engine import CleaningReport
    from universal_data.inspection.profiler import DatasetProfile
    from universal_data.observability.metrics import RunMetrics
    from universal_data.quality.metrics import QualityReport
    from universal_data.rules.engine import RuleEvaluation
    from universal_data.validation.results import ValidationResult

_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 2rem; background: #f6f7f9; color: #1c1e21; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 .75rem; border-bottom: 1px solid #d8dade; padding-bottom: .3rem; }
.meta { color: #63676c; font-size: .85rem; margin-bottom: 1.5rem; }
.cards { display: flex; flex-wrap: wrap; gap: .75rem; }
.card { background: #fff; border: 1px solid #e2e4e8; border-radius: 8px; padding: .9rem 1.1rem;
        min-width: 150px; flex: 1 1 150px; }
.card .label { font-size: .75rem; text-transform: uppercase; letter-spacing: .04em; color: #6b7075; }
.card .value { font-size: 1.5rem; font-weight: 600; margin-top: .2rem; }
table { border-collapse: collapse; width: 100%; background: #fff; font-size: .875rem;
        border: 1px solid #e2e4e8; border-radius: 8px; overflow: hidden; }
th, td { padding: .5rem .7rem; text-align: left; border-bottom: 1px solid #eceef1; }
th { background: #f0f2f5; font-weight: 600; }
tr:last-child td { border-bottom: none; }
.bar { background: #e6e8eb; border-radius: 4px; height: 8px; width: 120px; display: inline-block;
       vertical-align: middle; overflow: hidden; }
.bar > span { display: block; height: 100%; background: #2f855a; }
.bar.warn > span { background: #d69e2e; }
.bar.bad > span { background: #c53030; }
.grade { font-size: 2.2rem; font-weight: 700; }
ul.reco { padding-left: 1.1rem; }
ul.reco li { margin: .25rem 0; }
.empty { color: #6b7075; font-style: italic; }
@media (prefers-color-scheme: dark) {
  body { background: #16181c; color: #e6e8eb; }
  .card, table { background: #1e2126; border-color: #2c3037; }
  th { background: #23262c; }
  th, td { border-color: #2c3037; }
  h2 { border-color: #2c3037; }
}
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _bar(ratio: float) -> str:
    percent = max(0.0, min(1.0, ratio)) * 100
    css_class = "bar" if percent >= 90 else ("bar warn" if percent >= 75 else "bar bad")
    return f'<span class="{css_class}"><span style="width:{percent:.1f}%"></span></span>'


@dataclass
class QualityDocument:
    """Everything a pipeline learned about a dataset, in one object."""

    dataset: str = "dataset"
    source: str | None = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    profile: DatasetProfile | None = None
    validation: ValidationResult | None = None
    rules: RuleEvaluation | None = None
    quality: QualityReport | None = None
    cleaning: CleaningReport | None = None
    metrics: RunMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source": self.source,
            "generated_at": self.generated_at.isoformat(),
            "profile": self.profile.to_dict() if self.profile else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "business_rules": self.rules.to_dict() if self.rules else None,
            "quality": self.quality.to_dict() if self.quality else None,
            "cleaning": self.cleaning.to_dict() if self.cleaning else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }

    def to_json(self, path: PathLike | None = None, *, policy: PathPolicy | None = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)
        if path is not None:
            resolved = (policy or PathPolicy()).resolve_output(path)
            resolved.write_text(payload, encoding="utf-8")
        return payload

    def to_html(self, path: PathLike | None = None, *, policy: PathPolicy | None = None) -> str:
        markup = self._render_html()
        if path is not None:
            resolved = (policy or PathPolicy()).resolve_output(path)
            resolved.write_text(markup, encoding="utf-8")
        return markup

    # -- rendering ----------------------------------------------------------

    def _render_html(self) -> str:
        sections = [
            self._overview_section(),
            self._quality_section(),
            self._schema_section(),
            self._validation_section(),
            self._rules_section(),
            self._cleaning_section(),
            self._metrics_section(),
        ]
        body = "\n".join(section for section in sections if section)
        return (
            "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Data Quality Report - {_esc(self.dataset)}</title>"
            f"<style>{_CSS}</style></head><body>"
            f"<h1>Data Quality Report</h1>"
            f'<div class="meta">{_esc(self.dataset)}'
            f"{' &middot; ' + _esc(self.source) if self.source else ''}"
            f" &middot; generated {_esc(self.generated_at.strftime('%Y-%m-%d %H:%M UTC'))}</div>"
            f"{body}</body></html>"
        )

    def _overview_section(self) -> str:
        if not self.profile:
            return ""
        profile = self.profile
        cards = [
            ("Rows", f"{profile.rows:,}"),
            ("Columns", str(profile.columns)),
            ("Memory", f"{profile.memory_mb:.2f} MB"),
            ("Missing cells", f"{profile.total_missing:,} ({profile.missing_pct:.1f}%)"),
            ("Duplicate rows", f"{profile.duplicate_rows:,}"),
        ]
        if self.quality:
            cards.append(("Quality score", f"{self.quality.overall_score * 100:.1f}%"))
        rendered = "".join(
            f'<div class="card"><div class="label">{_esc(label)}</div>'
            f'<div class="value">{_esc(value)}</div></div>'
            for label, value in cards
        )
        return f"<h2>Dataset Overview</h2><div class=\"cards\">{rendered}</div>"

    def _quality_section(self) -> str:
        if not self.quality:
            return ""
        rows = "".join(
            f"<tr><td>{_esc(dimension.name.capitalize())}</td>"
            f"<td>{dimension.percentage:.1f}%</td>"
            f"<td>{_bar(dimension.score)}</td>"
            f"<td>{_esc(dimension.weight)}</td></tr>"
            for dimension in self.quality.dimensions
        )
        recommendations = (
            "<ul class=\"reco\">"
            + "".join(f"<li>{_esc(item)}</li>" for item in self.quality.recommendations)
            + "</ul>"
            if self.quality.recommendations
            else '<p class="empty">No recommendations - the dataset looks healthy.</p>'
        )
        return (
            "<h2>Quality Score</h2>"
            f'<p class="grade">{self.quality.overall_score * 100:.1f}% '
            f"<small>(grade {_esc(self.quality.grade)})</small></p>"
            "<table><thead><tr><th>Dimension</th><th>Score</th><th></th><th>Weight</th></tr>"
            f"</thead><tbody>{rows}</tbody></table>"
            f"<h2>Recommendations</h2>{recommendations}"
        )

    def _schema_section(self) -> str:
        if not self.profile:
            return ""
        rows = []
        for column in self.profile.columns_profile:
            extra = ""
            if column.numeric:
                extra = (
                    f"min={_fmt(column.numeric.minimum)}, max={_fmt(column.numeric.maximum)}, "
                    f"mean={_fmt(column.numeric.mean)}"
                )
            elif column.categorical:
                # Escaped once, at the point of insertion below.
                extra = ", ".join(f"{v} ({c})" for v, c in column.categorical.top_values[:3])
            elif column.temporal:
                extra = f"{column.temporal.earliest} .. {column.temporal.latest}"
            rows.append(
                f"<tr><td>{_esc(column.name)}</td><td>{_esc(column.inferred_type)}</td>"
                f"<td>{_esc(column.dtype)}</td><td>{column.missing:,} ({column.missing_pct:.1f}%)</td>"
                f"<td>{column.unique:,}</td><td>{column.outliers or ''}</td>"
                f"<td>{'yes' if column.is_pii else ''}</td><td>{_esc(extra)}</td></tr>"
            )
        return (
            "<h2>Schema and Column Statistics</h2>"
            "<table><thead><tr><th>Column</th><th>Detected type</th><th>dtype</th>"
            "<th>Missing</th><th>Unique</th><th>Outliers</th><th>PII</th><th>Details</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )

    def _validation_section(self) -> str:
        if not self.validation:
            return ""
        if not self.validation.issues:
            return (
                "<h2>Validation</h2>"
                f'<p class="empty">All {self.validation.total_rows:,} rows satisfy '
                f"schema '{_esc(self.validation.schema_name)}'.</p>"
            )
        rows = "".join(
            f"<tr><td>{_esc(issue.severity)}</td><td>{_esc(issue.column or '-')}</td>"
            f"<td>{_esc(issue.check)}</td><td>{_esc(issue.message)}</td>"
            f"<td>{issue.failed_rows:,}</td>"
            f"<td>{_esc(', '.join(str(s) for s in issue.samples[:3]))}</td></tr>"
            for issue in self.validation.issues
        )
        return (
            "<h2>Validation</h2>"
            f"<p>{self.validation.valid_rows:,} valid / {self.validation.invalid_rows:,} invalid rows</p>"
            "<table><thead><tr><th>Severity</th><th>Column</th><th>Check</th><th>Message</th>"
            f"<th>Rows</th><th>Examples</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    def _rules_section(self) -> str:
        if not self.rules:
            return ""
        rows = "".join(
            f"<tr><td>{_esc(outcome.name)}</td><td>{_esc(outcome.condition)}</td>"
            f"<td>{outcome.passed:,}</td><td>{outcome.failed:,}</td>"
            f"<td>{_bar(outcome.pass_rate)} {outcome.pass_rate * 100:.1f}%</td></tr>"
            for outcome in self.rules.outcomes
        )
        return (
            "<h2>Business Rules</h2>"
            f"<p>{self.rules.rows_passed:,} of {self.rules.total_rows:,} rows pass every rule</p>"
            "<table><thead><tr><th>Rule</th><th>Condition</th><th>Passed</th><th>Failed</th>"
            f"<th>Pass rate</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    def _cleaning_section(self) -> str:
        if not self.cleaning or not self.cleaning.steps:
            return ""
        rows = "".join(
            f"<tr><td>{_esc(step.get('operation', 'step'))}</td><td>{_esc(_details(step))}</td></tr>"
            for step in self.cleaning.steps
        )
        return (
            "<h2>Transformation Summary</h2>"
            f"<p>{self.cleaning.rows_before:,} rows in, {self.cleaning.rows_after:,} rows out</p>"
            f"<table><thead><tr><th>Operation</th><th>Details</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    def _metrics_section(self) -> str:
        if not self.metrics:
            return ""
        rows = "".join(
            f"<tr><td>{_esc(step.name)}</td><td>{_esc(step.status)}</td>"
            f"<td>{step.rows_in:,}</td><td>{step.rows_out:,}</td>"
            f"<td>{step.duration_seconds:.3f}s</td><td>{step.throughput:,.0f}</td></tr>"
            for step in self.metrics.steps
        )
        return (
            "<h2>Processing Statistics</h2>"
            f"<p>Execution {_esc(self.metrics.execution_id)} finished in "
            f"{self.metrics.duration_seconds:.2f}s with status {_esc(self.metrics.status)}</p>"
            "<table><thead><tr><th>Step</th><th>Status</th><th>Rows in</th><th>Rows out</th>"
            f"<th>Duration</th><th>Rows/s</th></tr></thead><tbody>{rows}</tbody></table>"
        )


def _details(step: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in step.items() if k != "operation"}, default=str)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)
