"""Business rules, outliers and quality scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from universal_data.core.exceptions import RuleError, TransformationError
from universal_data.core.types import OutlierAction, OutlierMethod, Severity
from universal_data.quality.metrics import QualityAnalyzer, QualityReport, analyze_quality
from universal_data.quality.outliers import compute_bounds, detect_outliers, handle_outliers
from universal_data.quality.report import QualityDocument
from universal_data.rules.engine import FAILURE_COLUMN, BusinessRule, RuleEngine
from universal_data.schema.model import Schema


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [30, 15, 45, 22],
            "email": ["a@b.com", "bad", "c@d.org", None],
            "salary": [1000.0, 2000.0, -50.0, 3000.0],
        }
    )


class TestBusinessRules:
    def test_rule_requires_a_name(self) -> None:
        with pytest.raises(RuleError, match="needs a name"):
            BusinessRule(name="", condition="age > 1")

    def test_from_dict_requires_fields(self) -> None:
        with pytest.raises(RuleError, match="missing 'name'"):
            BusinessRule.from_dict({"condition": "age > 1"})
        with pytest.raises(RuleError, match="missing 'condition'"):
            BusinessRule.from_dict({"name": "x"})

    def test_from_dict_rejects_unknown_severity(self) -> None:
        with pytest.raises(RuleError, match="Unknown rule severity"):
            BusinessRule.from_dict({"name": "x", "condition": "age > 1", "severity": "fatal"})

    def test_evaluation_counts(self, frame: pd.DataFrame) -> None:
        engine = RuleEngine.from_config(
            [
                {"name": "adult", "condition": "age >= 18"},
                {"name": "valid_email", "condition": "is_email(email)"},
                {"name": "positive", "condition": "salary >= 0", "severity": "warning"},
            ]
        )
        evaluation = engine.evaluate(frame)
        outcomes = {outcome.name: outcome for outcome in evaluation.outcomes}
        assert outcomes["adult"].failed == 1
        assert outcomes["valid_email"].failed == 2
        assert outcomes["positive"].failed == 1
        # Warnings do not remove rows from the passing set.
        assert evaluation.rows_failed == 2
        assert evaluation.to_dict()["rows_passed"] == 2

    def test_failure_reasons_are_recorded(self, frame: pd.DataFrame) -> None:
        engine = RuleEngine.from_config(
            [
                {"name": "adult", "condition": "age >= 18"},
                {"name": "valid_email", "condition": "is_email(email)"},
            ]
        )
        evaluation = engine.evaluate(frame)
        failed = evaluation.failed_frame(frame)
        assert FAILURE_COLUMN in failed.columns
        assert "adult" in failed[FAILURE_COLUMN].iloc[0]
        assert "valid_email" in failed[FAILURE_COLUMN].iloc[0]

    def test_split(self, frame: pd.DataFrame) -> None:
        engine = RuleEngine([BusinessRule(name="adult", condition="age >= 18")])
        passed, failed, evaluation = engine.split(frame)
        assert len(passed) == 3
        assert len(failed) == 1
        assert evaluation.is_clean is False

    def test_missing_column_skips_the_rule(self, frame: pd.DataFrame) -> None:
        engine = RuleEngine([BusinessRule(name="x", condition="not_a_column > 1")])
        evaluation = engine.evaluate(frame)
        assert evaluation.outcomes[0].skipped is True
        assert evaluation.rows_failed == 0

    def test_strict_mode_raises_for_missing_column(self, frame: pd.DataFrame) -> None:
        engine = RuleEngine([BusinessRule(name="x", condition="not_a_column > 1")])
        with pytest.raises(RuleError):
            engine.evaluate(frame, strict=True)

    def test_duplicate_rule_names_are_rejected(self) -> None:
        rule = BusinessRule(name="x", condition="age > 1")
        with pytest.raises(RuleError, match="Duplicate rule names"):
            RuleEngine([rule, BusinessRule(name="x", condition="age > 2")])

    def test_add_rejects_duplicates(self) -> None:
        engine = RuleEngine([BusinessRule(name="x", condition="age > 1")])
        with pytest.raises(RuleError, match="already exists"):
            engine.add(BusinessRule(name="x", condition="age > 2"))

    def test_empty_frame(self) -> None:
        engine = RuleEngine([BusinessRule(name="x", condition="age > 1")])
        evaluation = engine.evaluate(pd.DataFrame({"age": []}))
        assert evaluation.total_rows == 0
        assert evaluation.is_clean

    def test_repr_and_len(self) -> None:
        engine = RuleEngine([BusinessRule(name="x", condition="age > 1")])
        assert len(engine) == 1
        assert "x" in repr(engine)

    def test_outcome_serialisation(self, frame: pd.DataFrame) -> None:
        engine = RuleEngine([BusinessRule(name="x", condition="age >= 18", description="d")])
        outcome = engine.evaluate(frame).outcomes[0]
        data = outcome.to_dict()
        assert data["severity"] == str(Severity.ERROR)
        assert data["pass_rate"] == 0.75


class TestOutliers:
    @pytest.fixture
    def series(self) -> pd.Series:
        return pd.Series([10, 11, 12, 13, 12, 11, 10, 500])

    def test_iqr_detection(self, series: pd.Series) -> None:
        assert detect_outliers(series).tolist()[-1] is True

    @pytest.mark.parametrize("method", list(OutlierMethod))
    def test_every_method_runs(self, series: pd.Series, method: OutlierMethod) -> None:
        # A single extreme value inflates the mean and the standard deviation,
        # so the z-score methods need a looser threshold than the IQR default.
        thresholds = {
            OutlierMethod.PERCENTILE: 0.05,
            OutlierMethod.ZSCORE: 1.5,
            OutlierMethod.STDDEV: 1.5,
        }
        assert detect_outliers(series, method, thresholds.get(method)).sum() >= 1

    def test_empty_series(self) -> None:
        bounds = compute_bounds(pd.Series([], dtype=float))
        assert detect_outliers(pd.Series([], dtype=float)).empty
        assert bounds.to_dict()["lower"] is None

    def test_constant_series_has_no_outliers(self) -> None:
        series = pd.Series([5.0] * 10)
        assert detect_outliers(series, OutlierMethod.ZSCORE).sum() == 0

    def test_percentile_threshold_is_validated(self, series: pd.Series) -> None:
        with pytest.raises(TransformationError, match="between 0 and 0.5"):
            compute_bounds(series, OutlierMethod.PERCENTILE, 0.9)

    def test_report_action_changes_nothing(self, series: pd.Series) -> None:
        frame = pd.DataFrame({"v": series})
        result, report = handle_outliers(frame, action=OutlierAction.REPORT)
        assert result.equals(frame)
        assert report["total_outliers"] == 1

    def test_remove_action(self, series: pd.Series) -> None:
        frame = pd.DataFrame({"v": series})
        result, report = handle_outliers(frame, action=OutlierAction.REMOVE)
        assert len(result) == len(frame) - 1
        assert report["rows_removed"] == 1

    def test_cap_action(self, series: pd.Series) -> None:
        frame = pd.DataFrame({"v": series})
        result, _ = handle_outliers(frame, action=OutlierAction.CAP)
        assert result["v"].max() < 500

    def test_replace_action(self, series: pd.Series) -> None:
        frame = pd.DataFrame({"v": series})
        result, _ = handle_outliers(frame, action=OutlierAction.REPLACE, replacement=0)
        assert result["v"].iloc[-1] == 0

    def test_unknown_column(self) -> None:
        with pytest.raises(TransformationError, match="Unknown column"):
            handle_outliers(pd.DataFrame({"a": [1]}), ["b"])


class TestQuality:
    def test_dimensions_are_scored(self, frame: pd.DataFrame) -> None:
        report = analyze_quality(frame, name="test")
        names = {dimension.name for dimension in report.dimensions}
        assert {"completeness", "validity", "uniqueness", "consistency", "accuracy"} <= names
        assert 0 <= report.overall_score <= 1
        assert report.grade in {"A", "B", "C", "D", "F"}

    def test_perfect_dataset_scores_high(self) -> None:
        frame = pd.DataFrame({"id": range(50), "value": [float(i) for i in range(50)]})
        report = analyze_quality(frame)
        assert report.overall_score > 0.95

    def test_duplicates_lower_uniqueness(self) -> None:
        frame = pd.DataFrame({"a": [1, 1, 1, 2]})
        report = analyze_quality(frame)
        assert report.dimension("uniqueness").score < 1.0
        assert any("duplicate" in item for item in report.recommendations)

    def test_missing_values_lower_completeness(self) -> None:
        frame = pd.DataFrame({"a": [1, None, None, 4]})
        report = analyze_quality(frame)
        assert report.dimension("completeness").score == 0.5

    def test_schema_drives_validity(self, frame: pd.DataFrame) -> None:
        schema = Schema.from_dict({"name": "s", "schema": {"age": {"type": "integer", "min": 18}}})
        report = analyze_quality(frame, schema)
        assert report.dimension("validity").details["checked_against"] == "s"
        assert report.dimension("validity").score < 1.0

    def test_rules_affect_consistency(self, frame: pd.DataFrame) -> None:
        rules = RuleEngine([BusinessRule(name="adult", condition="age >= 18")])
        report = analyze_quality(frame, rules=rules)
        assert report.dimension("consistency").details["business_rule_pass_rate"] == 0.75

    def test_timeliness_uses_the_newest_record(self) -> None:
        recent = pd.Timestamp.now().strftime("%Y-%m-%d")
        frame = pd.DataFrame({"created_at": ["2019-01-01", recent], "v": [1, 2]})
        report = analyze_quality(frame)
        assert report.dimension("timeliness").score == pytest.approx(1.0)

    def test_future_dates_are_penalised(self) -> None:
        future = (pd.Timestamp.now() + pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        frame = pd.DataFrame({"created_at": [future, future], "v": [1, 2]})
        report = analyze_quality(frame)
        assert report.dimension("timeliness").score == 0.0
        assert any("future" in item for item in report.recommendations)

    def test_stale_dataset_loses_timeliness(self) -> None:
        frame = pd.DataFrame({"created_at": ["2000-01-01"], "v": [1]})
        report = analyze_quality(frame)
        assert report.dimension("timeliness").score == 0.0

    def test_empty_dataset_is_not_scored(self) -> None:
        report = analyze_quality(pd.DataFrame())
        assert report.scored is False
        assert report.grade == "n/a"          # not "F": absent data, not bad data
        assert report.overall_score == 0.0
        assert report.recommendations
        assert "Not scored" in report.summary()
        assert report.to_dict()["scored"] is False

    def test_custom_weights(self, frame: pd.DataFrame) -> None:
        analyzer = QualityAnalyzer(weights={"completeness": 1.0, "validity": 0.0})
        report = analyzer.analyze(frame)
        assert report.dimension("completeness").weight == 1.0

    def test_column_scores(self, frame: pd.DataFrame) -> None:
        report = analyze_quality(frame)
        assert set(report.column_scores) == set(frame.columns)

    def test_summary_and_dict(self, frame: pd.DataFrame) -> None:
        report = analyze_quality(frame, name="ds")
        assert "Overall Score" in report.summary()
        assert report.to_dict()["dataset"] == "ds"

    @pytest.mark.parametrize(
        ("score", "expected"), [(0.99, "A"), (0.92, "B"), (0.85, "C"), (0.75, "D"), (0.5, "F")]
    )
    def test_grade_boundaries(self, score: float, expected: str) -> None:
        from universal_data.quality.metrics import QualityDimension

        report = QualityReport(dataset="x", rows=1, columns=1)
        report.dimensions.append(QualityDimension(name="completeness", score=score, weight=1.0))
        assert report.grade == expected


class TestQualityDocument:
    def test_json_and_html_output(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        from universal_data.inspection.profiler import profile_dataset

        rules = RuleEngine([BusinessRule(name="adult", condition="age >= 18")])
        document = QualityDocument(
            dataset="customers",
            source="customers.csv",
            profile=profile_dataset(frame, name="customers"),
            quality=analyze_quality(frame),
            rules=rules.evaluate(frame),
        )
        json_path = tmp_path / "report.json"
        html_path = tmp_path / "report.html"
        document.to_json(json_path)
        document.to_html(html_path)
        assert json_path.stat().st_size > 0
        markup = html_path.read_text(encoding="utf-8")
        assert "Data Quality Report" in markup
        assert "Business Rules" in markup

    def test_html_escapes_hostile_content(self, tmp_path: Path) -> None:
        from universal_data.inspection.profiler import profile_dataset

        hostile = pd.DataFrame({"note": ["<script>alert('xss')</script>"] * 3})
        document = QualityDocument(
            dataset="<img src=x onerror=alert(1)>",
            profile=profile_dataset(hostile),
            quality=analyze_quality(hostile),
        )
        markup = document.to_html()
        assert "<script>" not in markup
        assert "&lt;script&gt;" in markup
        # The hostile title survives as inert text, never as a tag.
        assert "<img" not in markup
        assert "&lt;img src=x onerror=alert(1)&gt;" in markup

    def test_empty_document_renders(self) -> None:
        assert "Data Quality Report" in QualityDocument().to_html()
