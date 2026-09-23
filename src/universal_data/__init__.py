"""Universal Data Processing Toolkit.

A reusable framework for reading, validating, cleaning, transforming, scoring and
exporting tabular data - available both as a Python library and as the
``data-tool`` command line application.

    from universal_data import Dataset, Pipeline

    dataset = Dataset.from_csv("customers.csv")
    clean = dataset.clean(missing_values="median").remove_duplicates()
    clean.to_parquet("customers.parquet")

Importing this package registers the built-in readers, writers and
transformations, but never configures logging: that is the application's choice.
"""

from __future__ import annotations

__version__ = "1.0.0"

from universal_data.cleaning import CleaningConfig, DataCleaner
from universal_data.core.dataset import Dataset
from universal_data.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    DatabaseError,
    DataProcessingError,
    DataToolkitError,
    DataValidationError,
    ExportError,
    ExtractionError,
    PipelineError,
    PluginError,
    RuleError,
    SchemaError,
    SecurityError,
    TransformationError,
    UnsupportedFormatError,
)
from universal_data.core.types import (
    DuplicateKeep,
    FieldType,
    FileFormat,
    MissingStrategy,
    OutlierAction,
    OutlierMethod,
    Severity,
)
from universal_data.enrichment import Enricher, LookupEnricher
from universal_data.export import writers
from universal_data.ingestion import APIClient, APIConfig, DatabaseClient, DatabaseConfig, readers
from universal_data.inspection import DatasetProfile, profile_dataset
from universal_data.masking import MaskingPolicy
from universal_data.observability import configure_logging
from universal_data.pipeline import Pipeline, PipelineConfig, load_pipeline
from universal_data.quality import QualityDocument, QualityReport, analyze_quality
from universal_data.rules import BusinessRule, RuleEngine, SafeExpression
from universal_data.schema import ColumnSchema, Schema, detect_schema, load_schema
from universal_data.transformation import Transformation, transformations
from universal_data.validation import SchemaValidator, ValidationResult

__all__ = [
    "APIClient",
    "APIConfig",
    "AuthenticationError",
    "BusinessRule",
    "CleaningConfig",
    "ColumnSchema",
    "ConfigurationError",
    "DataCleaner",
    "DataProcessingError",
    "DataToolkitError",
    "DataValidationError",
    "DatabaseClient",
    "DatabaseConfig",
    "DatabaseError",
    "Dataset",
    "DatasetProfile",
    "DuplicateKeep",
    "Enricher",
    "ExportError",
    "ExtractionError",
    "FieldType",
    "FileFormat",
    "LookupEnricher",
    "MaskingPolicy",
    "MissingStrategy",
    "OutlierAction",
    "OutlierMethod",
    "Pipeline",
    "PipelineConfig",
    "PipelineError",
    "PluginError",
    "QualityDocument",
    "QualityReport",
    "RuleEngine",
    "RuleError",
    "SafeExpression",
    "Schema",
    "SchemaError",
    "SchemaValidator",
    "SecurityError",
    "Severity",
    "Transformation",
    "TransformationError",
    "UnsupportedFormatError",
    "ValidationResult",
    "__version__",
    "analyze_quality",
    "configure_logging",
    "detect_schema",
    "load_pipeline",
    "load_schema",
    "profile_dataset",
    "readers",
    "transformations",
    "writers",
]
