"""Outlier detection and treatment.

Detection never changes the data by itself.  Removing or capping values is a
business decision - a salary of 500,000 is an outlier in most datasets and a
fact in some - so the action has to be requested explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from universal_data.core.exceptions import TransformationError
from universal_data.core.types import OutlierAction, OutlierMethod

DEFAULT_THRESHOLDS: dict[OutlierMethod, float] = {
    OutlierMethod.IQR: 1.5,
    OutlierMethod.ZSCORE: 3.0,
    OutlierMethod.STDDEV: 3.0,
    OutlierMethod.PERCENTILE: 0.01,
}


@dataclass
class OutlierBounds:
    lower: float
    upper: float
    method: OutlierMethod
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": None if np.isnan(self.lower) else round(float(self.lower), 6),
            "upper": None if np.isnan(self.upper) else round(float(self.upper), 6),
            "method": str(self.method),
            "threshold": self.threshold,
        }


def compute_bounds(
    series: pd.Series,
    method: OutlierMethod | str = OutlierMethod.IQR,
    threshold: float | None = None,
) -> OutlierBounds:
    """Return the lower/upper cut-offs for one column."""
    method = OutlierMethod(method)
    threshold = DEFAULT_THRESHOLDS[method] if threshold is None else float(threshold)
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return OutlierBounds(float("nan"), float("nan"), method, threshold)

    if method is OutlierMethod.IQR:
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        spread = q3 - q1
        return OutlierBounds(q1 - threshold * spread, q3 + threshold * spread, method, threshold)

    if method in (OutlierMethod.ZSCORE, OutlierMethod.STDDEV):
        mean = values.mean()
        std = values.std()
        if std == 0 or np.isnan(std):
            return OutlierBounds(float("-inf"), float("inf"), method, threshold)
        return OutlierBounds(mean - threshold * std, mean + threshold * std, method, threshold)

    # Percentile: threshold is the tail probability trimmed from each side.
    if not 0 < threshold < 0.5:
        raise TransformationError(
            "Percentile threshold must be between 0 and 0.5", threshold=threshold
        )
    return OutlierBounds(
        float(values.quantile(threshold)),
        float(values.quantile(1 - threshold)),
        method,
        threshold,
    )


def detect_outliers(
    series: pd.Series,
    method: OutlierMethod | str = OutlierMethod.IQR,
    threshold: float | None = None,
) -> pd.Series:
    """Boolean mask that is ``True`` for outlying values (nulls are ``False``)."""
    bounds = compute_bounds(series, method, threshold)
    if np.isnan(bounds.lower) and np.isnan(bounds.upper):
        return pd.Series(False, index=series.index)
    values = pd.to_numeric(series, errors="coerce")
    return ((values < bounds.lower) | (values > bounds.upper)).fillna(False)


def handle_outliers(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: OutlierMethod | str = OutlierMethod.IQR,
    threshold: float | None = None,
    action: OutlierAction | str = OutlierAction.REPORT,
    replacement: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect outliers and optionally remove, cap or replace them."""
    action = OutlierAction(action)
    targets = (
        [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
        if columns is None
        else list(columns)
    )
    missing = [column for column in targets if column not in frame.columns]
    if missing:
        raise TransformationError("Unknown column(s)", columns=missing)

    result = frame.copy()
    per_column: dict[str, Any] = {}
    drop_mask = pd.Series(False, index=frame.index)

    for column in targets:
        bounds = compute_bounds(frame[column], method, threshold)
        mask = detect_outliers(frame[column], method, threshold)
        count = int(mask.sum())
        per_column[column] = {"outliers": count, **bounds.to_dict()}
        if not count:
            continue
        if action is OutlierAction.REMOVE:
            drop_mask |= mask
        elif action is OutlierAction.CAP:
            result[column] = pd.to_numeric(result[column], errors="coerce").clip(
                lower=bounds.lower, upper=bounds.upper
            )
        elif action is OutlierAction.REPLACE:
            value = replacement if replacement is not None else np.nan
            result[column] = pd.to_numeric(result[column], errors="coerce").mask(mask, value)

    rows_removed = 0
    if action is OutlierAction.REMOVE and drop_mask.any():
        rows_removed = int(drop_mask.sum())
        result = result.loc[~drop_mask]

    return result, {
        "operation": "handle_outliers",
        "method": str(OutlierMethod(method)),
        "action": str(action),
        "columns": per_column,
        "rows_removed": rows_removed,
        "total_outliers": sum(item["outliers"] for item in per_column.values()),
    }
