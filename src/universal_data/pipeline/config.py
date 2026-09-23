"""Configuration-driven pipelines.

A YAML file describes the input, the steps and the output; this module turns it
into a :class:`~universal_data.pipeline.pipeline.Pipeline`.

Two things are deliberate here:

* every path in the file is resolved against the directory of the configuration
  and confined to a workspace root, so a configuration that ships with a
  repository cannot read ``../../../.ssh/id_rsa``;
* credentials are never written in the file - the database section takes
  ``password_env: PG_PASSWORD`` and the value is read from the environment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from universal_data.core.exceptions import ConfigurationError
from universal_data.core.types import PathLike
from universal_data.observability.logging import get_logger
from universal_data.pipeline.pipeline import Pipeline
from universal_data.rules.engine import RuleEngine
from universal_data.schema.loader import load_schema
from universal_data.security.paths import PathPolicy
from universal_data.security.secrets import SecretResolver, SecretStr

logger = get_logger(__name__)

KNOWN_TOP_LEVEL = {"pipeline", "input", "steps", "output", "report", "logging"}
KNOWN_STEPS = {
    "clean",
    "remove_duplicates",
    "validate",
    "transform",
    "filter",
    "rules",
    "mask",
    "mask_pii",
    "outliers",
    "enrich",
    "quality",
}


@dataclass
class PipelineConfig:
    """A parsed pipeline definition."""

    data: dict[str, Any]
    base_dir: Path
    policy: PathPolicy
    resolver: SecretResolver = field(default_factory=SecretResolver)

    @classmethod
    def from_file(
        cls,
        path: PathLike,
        *,
        workspace: PathLike | None = None,
        resolver: SecretResolver | None = None,
    ) -> PipelineConfig:
        config_path = PathPolicy().resolve_input(path)
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Could not parse the pipeline configuration: {exc}", path=str(config_path)
            ) from exc
        if not isinstance(payload, Mapping):
            raise ConfigurationError("A pipeline configuration must be a mapping")
        return cls.from_dict(
            dict(payload),
            base_dir=config_path.parent,
            workspace=workspace,
            resolver=resolver,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        base_dir: PathLike = ".",
        workspace: PathLike | None = None,
        resolver: SecretResolver | None = None,
    ) -> PipelineConfig:
        unknown = set(data) - KNOWN_TOP_LEVEL
        if unknown:
            raise ConfigurationError(
                "Unknown top-level key(s) in the pipeline configuration",
                keys=sorted(unknown),
                allowed=sorted(KNOWN_TOP_LEVEL),
            )
        data = _normalize_keys(data)
        base = Path(base_dir).resolve()
        declared = (data.get("pipeline") or {}).get("workspace")
        root = Path(workspace) if workspace else (base / declared if declared else Path.cwd())
        policy = PathPolicy.confined_to(root)
        return cls(data=data, base_dir=base, policy=policy, resolver=resolver or SecretResolver())

    # -- helpers ------------------------------------------------------------

    @property
    def name(self) -> str:
        return str((self.data.get("pipeline") or {}).get("name", "pipeline"))

    def _path(self, value: Any, *, output: bool = False) -> Path:
        raw = self.resolver.resolve(value)
        if not isinstance(raw, str) or not raw:
            raise ConfigurationError("Expected a file path", given=repr(value))
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        return (
            self.policy.resolve_output(candidate)
            if output
            else self.policy.resolve_input(candidate)
        )

    def _section(self, key: str) -> dict[str, Any]:
        section = self.data.get(key) or {}
        if not isinstance(section, Mapping):
            raise ConfigurationError(f"'{key}' must be a mapping")
        return dict(section)

    # -- building -----------------------------------------------------------

    def build(self) -> Pipeline:
        """Turn the configuration into an executable pipeline."""
        settings = self._section("pipeline")
        pipeline = Pipeline(
            name=self.name,
            on_error=str(settings.get("on_error", "fail")),
            policy=self.policy,
        )
        self._configure_input(pipeline)
        for entry in self.data.get("steps") or []:
            self._add_step(pipeline, entry)
        self._configure_output(pipeline)
        self._configure_report(pipeline)
        return pipeline

    def _configure_input(self, pipeline: Pipeline) -> None:
        section = self._section("input")
        if not section:
            raise ConfigurationError("The configuration has no 'input' section")
        kind = str(section.get("type", "")).lower()
        options = dict(section.get("options") or {})

        if kind in ("database", "postgresql", "postgres", "mysql", "sqlite_db"):
            self._configure_database_input(pipeline, section, options)
            return
        if kind == "api":
            self._configure_api_input(pipeline, section, options)
            return

        if "path" not in section:
            raise ConfigurationError("A file input needs a 'path'")
        path = self._path(section["path"])
        pipeline.read(path, kind or None, **options)

    def _configure_database_input(
        self, pipeline: Pipeline, section: dict[str, Any], options: dict[str, Any]
    ) -> None:
        from universal_data.core.dataset import Dataset
        from universal_data.ingestion.database import DatabaseClient, DatabaseConfig

        connection = section.get("connection")
        if not isinstance(connection, Mapping):
            raise ConfigurationError("A database input needs a 'connection' mapping")
        config = DatabaseConfig.from_dict(connection, self.resolver)
        table = section.get("table")
        query = section.get("query")

        def factory() -> Dataset:
            with DatabaseClient(config) as client:
                return Dataset.from_database(
                    client, table=table, query=query, params=options.get("params")
                )

        pipeline.from_callable(factory)

    def _configure_api_input(
        self, pipeline: Pipeline, section: dict[str, Any], options: dict[str, Any]
    ) -> None:
        from universal_data.core.dataset import Dataset
        from universal_data.ingestion.api import (
            APIClient,
            APIConfig,
            ApiKeyAuth,
            AuthStrategy,
            BearerTokenAuth,
            NoAuth,
            PageNumberPaginator,
        )
        from universal_data.security.network import URLPolicy

        url = self.resolver.resolve(section.get("url"))
        if not url:
            raise ConfigurationError("An API input needs a 'url'")
        auth_section = dict(section.get("auth") or {})
        auth_type = str(auth_section.pop("type", "none")).lower()
        auth: AuthStrategy
        if auth_type in ("none", ""):
            auth = NoAuth()
        elif auth_type in ("api_key", "apikey"):
            token = self._required_secret(auth_section, "API_KEY")
            auth = ApiKeyAuth(key=token, header=auth_section.get("header", "X-API-Key"))
        elif auth_type in ("bearer", "token"):
            auth = BearerTokenAuth(token=self._required_secret(auth_section, "API_TOKEN"))
        else:
            raise ConfigurationError("Unsupported API auth type", type=auth_type)

        pagination = dict(section.get("pagination") or {})
        paginator = PageNumberPaginator(**pagination) if pagination else None
        api_config = APIConfig(
            url=str(url),
            method=str(section.get("method", "GET")),
            headers=dict(section.get("headers") or {}),
            params=dict(section.get("params") or {}),
            records_path=section.get("records_path"),
            max_records=section.get("max_records"),
        )
        url_policy = URLPolicy(
            allow_http=bool(section.get("allow_http", False)),
            allow_private_networks=bool(section.get("allow_private_networks", False)),
            allowed_hosts=frozenset(section.get("allowed_hosts") or ()),
        )

        def factory() -> Dataset:
            with APIClient(
                api_config, auth=auth, paginator=paginator, url_policy=url_policy,
                rate_limit=section.get("rate_limit"),
            ) as client:
                return Dataset.from_api(client)

        pipeline.from_callable(factory)

    def _required_secret(self, auth_section: dict[str, Any], default_variable: str) -> SecretStr:
        variable = str(auth_section.get("env", default_variable))
        secret = self.resolver.get(variable, required=True)
        assert secret is not None  # required=True guarantees it
        return secret

    def _add_step(self, pipeline: Pipeline, entry: Any) -> None:
        options: Any
        if isinstance(entry, str):
            name, options = entry, {}
        elif isinstance(entry, Mapping) and len(entry) == 1:
            (name, options), = entry.items()
            options = options if options is not None else {}
        else:
            raise ConfigurationError(
                "Each step must be a name or a single-key mapping", step=str(entry)
            )
        if name not in KNOWN_STEPS:
            raise ConfigurationError(
                "Unknown pipeline step", step=name, available=sorted(KNOWN_STEPS)
            )
        getattr(self, f"_step_{name}")(pipeline, options)

    # -- individual steps ---------------------------------------------------

    def _step_clean(self, pipeline: Pipeline, options: Any) -> None:
        pipeline.clean(dict(options or {}))

    def _step_remove_duplicates(self, pipeline: Pipeline, options: Any) -> None:
        options = dict(options or {})
        pipeline.remove_duplicates(options.get("subset"), keep=options.get("keep", "first"))

    def _step_validate(self, pipeline: Pipeline, options: Any) -> None:
        options = dict(options or {})
        schema_path = options.get("schema")
        if not schema_path:
            raise ConfigurationError("validate needs a 'schema' path")
        schema = load_schema(self._path(schema_path), policy=self.policy)
        pipeline.validate(schema, mode=str(options.get("mode", "report")))

    def _step_transform(self, pipeline: Pipeline, options: Any) -> None:
        from universal_data.transformation.base import TransformationChain

        chain = TransformationChain.from_config(options)
        pipeline.transform(*chain.steps)

    def _step_filter(self, pipeline: Pipeline, options: Any) -> None:
        condition = options if isinstance(options, str) else dict(options or {}).get("condition")
        if not condition:
            raise ConfigurationError("filter needs a condition")
        pipeline.filter(str(condition))

    def _step_rules(self, pipeline: Pipeline, options: Any) -> None:
        engine = self._load_rules(options)
        action = "flag"
        if isinstance(options, Mapping):
            action = str(options.get("action", "flag"))
        pipeline.apply_rules(engine, action=action)

    def _load_rules(self, options: Any) -> RuleEngine:
        if isinstance(options, list):
            return RuleEngine.from_config(options)
        options = dict(options or {})
        if "file" in options:
            path = self._path(options["file"])
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            definitions = payload.get("rules", payload) if isinstance(payload, Mapping) else payload
            if not isinstance(definitions, list):
                raise ConfigurationError("The rules file must contain a list of rules", path=str(path))
            return RuleEngine.from_config(definitions)
        if "rules" in options:
            return RuleEngine.from_config(list(options["rules"]))
        raise ConfigurationError("rules needs either 'file' or an inline 'rules' list")

    def _step_mask(self, pipeline: Pipeline, options: Any) -> None:
        from universal_data.masking.maskers import MaskingPolicy

        pipeline.mask(MaskingPolicy.from_dict(dict(options or {})))

    def _step_mask_pii(self, pipeline: Pipeline, options: Any) -> None:
        options = dict(options or {})
        pipeline.mask_detected_pii(str(options.pop("strategy", "hash")), **options)

    def _step_outliers(self, pipeline: Pipeline, options: Any) -> None:
        options = dict(options or {})
        pipeline.handle_outliers(
            columns=options.get("columns"),
            method=options.get("method", "iqr"),
            threshold=options.get("threshold"),
            action=options.get("action", "report"),
        )

    def _step_enrich(self, pipeline: Pipeline, options: Any) -> None:
        from universal_data.core.dataset import Dataset
        from universal_data.enrichment.enrichers import DerivedColumnEnricher, LookupEnricher

        options = dict(options or {})
        if "lookup" in options:
            lookup = dict(options["lookup"])
            path = self._path(lookup.pop("path"))
            frame = Dataset.read(path, policy=self.policy).frame
            pipeline.enrich(LookupEnricher(frame, **lookup))
            return
        if "derive" in options:
            for column, expression in dict(options["derive"]).items():
                pipeline.enrich(DerivedColumnEnricher(column, str(expression)))
            return
        raise ConfigurationError("enrich needs either 'lookup' or 'derive'")

    def _step_quality(self, pipeline: Pipeline, options: Any) -> None:
        options = dict(options or {})
        schema = None
        if "schema" in options:
            schema = load_schema(self._path(options["schema"]), policy=self.policy)
        rules = self._load_rules(options["rules"]) if "rules" in options else None
        pipeline.quality_check(schema, rules, minimum_score=options.get("minimum_score"))

    def _configure_output(self, pipeline: Pipeline) -> None:
        section = self._section("output")
        if not section:
            return
        kind = str(section.get("type", "")).lower()
        options = dict(section.get("options") or {})
        if kind in ("database", "postgresql", "postgres", "mysql"):
            self._configure_database_output(pipeline, section, options)
            return
        if "path" not in section:
            raise ConfigurationError("A file output needs a 'path'")
        target = self._path(section["path"], output=True)
        pipeline.write(target, kind or None, **options)

    def _configure_database_output(
        self, pipeline: Pipeline, section: dict[str, Any], options: dict[str, Any]
    ) -> None:
        from universal_data.core.dataset import Dataset
        from universal_data.ingestion.database import DatabaseClient, DatabaseConfig
        from universal_data.pipeline.pipeline import PipelineContext

        connection = section.get("connection")
        if not isinstance(connection, Mapping):
            raise ConfigurationError("A database output needs a 'connection' mapping")
        config = DatabaseConfig.from_dict(connection, self.resolver)
        table = str(section.get("table") or "")
        if not table:
            raise ConfigurationError("A database output needs a 'table'")
        if_exists = str(section.get("if_exists", "append"))

        def step(dataset: Dataset, context: PipelineContext) -> Dataset:
            with DatabaseClient(config) as client:
                rows = dataset.to_database(client, table, if_exists=if_exists, **options)
            context.artifacts.setdefault("outputs", []).append({"table": table, "rows": rows})
            return dataset

        pipeline.add_step("write_database", step)

    def _configure_report(self, pipeline: Pipeline) -> None:
        section = self._section("report")
        if not section:
            return
        if "path" not in section:
            raise ConfigurationError("A report needs a 'path'")
        target = self._path(section["path"], output=True)
        pipeline.report(target, fmt=str(section.get("format", "html")))


def _normalize_keys(value: Any) -> Any:
    """Undo YAML 1.1 boolean keys.

    PyYAML parses the bare key ``on:`` as ``True`` (and ``off``/``yes``/``no``
    likewise), which silently breaks ``enrich.lookup.on``.  Mapping keys are
    always identifiers here, so converting them back is safe.
    """
    booleans = {True: "on", False: "off"}
    if isinstance(value, Mapping):
        return {booleans.get(key, key) if isinstance(key, bool) else key: _normalize_keys(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_keys(item) for item in value]
    return value


def load_pipeline(path: PathLike, **kwargs: Any) -> Pipeline:
    """Build a pipeline from a YAML file."""
    return PipelineConfig.from_file(path, **kwargs).build()
