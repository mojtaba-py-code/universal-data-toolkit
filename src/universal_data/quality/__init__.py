"""Data quality scoring, outlier handling and reporting."""

from universal_data.quality.metrics import (
    QualityAnalyzer,
    QualityDimension,
    QualityReport,
    analyze_quality,
)
from universal_data.quality.outliers import (
    OutlierBounds,
    compute_bounds,
    detect_outliers,
    handle_outliers,
)
from universal_data.quality.report import QualityDocument

__all__ = [
    "OutlierBounds",
    "QualityAnalyzer",
    "QualityDimension",
    "QualityDocument",
    "QualityReport",
    "analyze_quality",
    "compute_bounds",
    "detect_outliers",
    "handle_outliers",
]
