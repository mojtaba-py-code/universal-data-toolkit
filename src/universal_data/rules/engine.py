"""Business rule engine.

Rules live outside the processing code - normally in YAML - and are evaluated as
vectorised boolean masks.  The engine reports which rows passed, which failed and
why, without deciding what to do about it; that is the caller's choice.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from universal_data.core.exceptions import RuleError
from universal_data.core.types import Severity
from universal_data.observability.logging import get_logger
from universal_data.rules.expression import SafeExpression, compile_expression

logger = get_logger(__name__)

FAILURE_COLUMN = "_rule_failures"


@dataclass(frozen=True)
class BusinessRule:
    """A named condition every row is expected to satisfy."""

    name: str
    condition: str
    description: str = ""
    severity: Severity = Severity.ERROR
    expression: SafeExpression = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise RuleError("Business rule needs a name")
        object.__setattr__(self, "expression", compile_expression(self.condition))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BusinessRule:
        if "name" not in data:
            raise RuleError("Rule definition is missing 'name'", definition=data)
        if "condition" not in data:
            raise RuleError("Rule definition is missing 'condition'", rule=data.get("name"))
        severity = data.get("severity", Severity.ERROR)
        try:
            severity = Severity(str(severity).lower())
        except ValueError as exc:
            raise RuleError(
                "Unknown rule severity", rule=data["name"], severity=data.get("severity")
            ) from exc
        return cls(
            name=str(data["name"]),
            condition=str(data["condition"]),
            description=str(data.get("description", "")),
            severity=severity,
        )


@dataclass
class RuleOutcome:
    """Per-rule statistics for one evaluation."""

    name: str
    severity: Severity
    description: str
    condition: str
    evaluated: int
    passed: int
    failed: int
    skipped: bool = False
    error: str | None = None

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.evaluated) if self.evaluated else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "severity": str(self.severity),
            "description": self.description,
            "condition": self.condition,
            "evaluated": self.evaluated,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "skipped": self.skipped,
            "error": self.error,
        }


@dataclass
class RuleEvaluation:
    """Result of running a rule set over a frame."""

    outcomes: list[RuleOutcome]
    passing_mask: pd.Series
    reasons: pd.Series
    total_rows: int

    @property
    def rows_passed(self) -> int:
        return int(self.passing_mask.sum())

    @property
    def rows_failed(self) -> int:
        return self.total_rows - self.rows_passed

    @property
    def is_clean(self) -> bool:
        return self.rows_failed == 0

    def passed_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[self.passing_mask]

    def failed_frame(self, frame: pd.DataFrame, *, with_reason: bool = True) -> pd.DataFrame:
        failed = frame.loc[~self.passing_mask]
        if with_reason and not failed.empty:
            failed = failed.assign(**{FAILURE_COLUMN: self.reasons.loc[failed.index]})
        return failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "rows_passed": self.rows_passed,
            "rows_failed": self.rows_failed,
            "pass_rate": round(self.rows_passed / self.total_rows, 4) if self.total_rows else 1.0,
            "rules": [outcome.to_dict() for outcome in self.outcomes],
        }


class RuleEngine:
    """Evaluates a set of :class:`BusinessRule` objects against a DataFrame."""

    def __init__(self, rules: Sequence[BusinessRule] | None = None) -> None:
        self.rules: list[BusinessRule] = list(rules or [])
        names = [rule.name for rule in self.rules]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise RuleError("Duplicate rule names", names=sorted(duplicates))

    @classmethod
    def from_config(cls, definitions: Iterable[dict[str, Any]]) -> RuleEngine:
        return cls([BusinessRule.from_dict(item) for item in definitions])

    def add(self, rule: BusinessRule) -> RuleEngine:
        if any(existing.name == rule.name for existing in self.rules):
            raise RuleError("A rule with this name already exists", name=rule.name)
        self.rules.append(rule)
        return self

    def evaluate(self, frame: pd.DataFrame, *, strict: bool = False) -> RuleEvaluation:
        """Run every rule.

        ``strict=True`` turns a rule that cannot be evaluated (missing column,
        bad type) into an error instead of a skipped rule.  The lenient default
        matters for pipelines where an optional column may be absent.
        """
        total = len(frame)
        passing = pd.Series(True, index=frame.index)
        reasons = pd.Series([""] * total, index=frame.index, dtype="object")
        outcomes: list[RuleOutcome] = []

        for rule in self.rules:
            outcome = RuleOutcome(
                name=rule.name,
                severity=rule.severity,
                description=rule.description,
                condition=rule.condition,
                evaluated=total,
                passed=total,
                failed=0,
            )
            try:
                mask = rule.expression.evaluate_mask(frame) if total else pd.Series(
                    [], dtype=bool, index=frame.index
                )
            except RuleError as exc:
                if strict:
                    raise
                outcome.skipped = True
                outcome.evaluated = 0
                outcome.passed = 0
                outcome.error = str(exc)
                logger.warning("Rule '%s' skipped: %s", rule.name, exc)
                outcomes.append(outcome)
                continue

            failed_mask = ~mask
            failed_count = int(failed_mask.sum())
            outcome.passed = total - failed_count
            outcome.failed = failed_count
            outcomes.append(outcome)

            if failed_count and rule.severity is Severity.ERROR:
                passing &= mask
            if failed_count:
                reasons = reasons.where(
                    ~failed_mask,
                    np.where(reasons == "", rule.name, reasons + ";" + rule.name),
                )

        return RuleEvaluation(
            outcomes=outcomes, passing_mask=passing, reasons=reasons, total_rows=total
        )

    def split(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, RuleEvaluation]:
        """Convenience helper returning ``(passed, failed, evaluation)``."""
        evaluation = self.evaluate(frame)
        return evaluation.passed_frame(frame), evaluation.failed_frame(frame), evaluation

    def __len__(self) -> int:
        return len(self.rules)

    def __repr__(self) -> str:
        return f"RuleEngine(rules={[r.name for r in self.rules]})"
