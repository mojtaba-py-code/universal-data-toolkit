"""Masking, enrichment and profiling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from universal_data.core.exceptions import ConfigurationError, TransformationError
from universal_data.enrichment.enrichers import (
    CallableEnricher,
    DerivedColumnEnricher,
    EnrichmentChain,
    LookupEnricher,
    MappingEnricher,
)
from universal_data.inspection.profiler import profile_column, profile_dataset
from universal_data.masking.maskers import (
    EmailMasker,
    HashMasker,
    MaskingPolicy,
    NameMasker,
    PartialMasker,
    PhoneMasker,
    RedactMasker,
    TokenMasker,
    create_masker,
)
from universal_data.security.secrets import SecretStr


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "email": ["john.smith@example.com", "a@b.io", None],
            "phone": ["+491701234567", "0912", None],
            "name": ["Ali Rezaei", "Sara", None],
            "country_code": ["IR", "DE", "IR"],
            "amount": [10.0, 20.0, 30.0],
        }
    )


class TestMaskers:
    def test_email_masker(self, frame: pd.DataFrame) -> None:
        masked = EmailMasker().mask(frame["email"])
        assert masked.iloc[0] == "j***@example.com"
        assert pd.isna(masked.iloc[2])

    def test_email_masker_can_hide_the_domain(self, frame: pd.DataFrame) -> None:
        masked = EmailMasker(mask_domain=True).mask(frame["email"])
        assert masked.iloc[0] == "j***@***.com"

    def test_email_masker_handles_garbage(self) -> None:
        masked = EmailMasker().mask(pd.Series(["not-an-email"]))
        assert masked.iloc[0] == "***"

    def test_phone_masker_keeps_the_tail(self, frame: pd.DataFrame) -> None:
        masked = PhoneMasker(keep_last=4).mask(frame["phone"])
        assert masked.iloc[0].endswith("4567")
        assert "170" not in masked.iloc[0]

    def test_phone_masker_with_short_number(self, frame: pd.DataFrame) -> None:
        assert PhoneMasker(keep_last=8).mask(frame["phone"]).iloc[1] == "****"

    def test_name_masker(self, frame: pd.DataFrame) -> None:
        masked = NameMasker().mask(frame["name"])
        assert masked.iloc[0] == "A. R."
        assert masked.iloc[1] == "S."

    def test_partial_masker(self) -> None:
        masked = PartialMasker(keep_first=2, keep_last=2).mask(pd.Series(["1234567890", "abc"]))
        assert masked.iloc[0] == "12******90"
        assert masked.iloc[1] == "***"

    def test_hash_masker_is_deterministic(self, frame: pd.DataFrame) -> None:
        masker = HashMasker(salt=SecretStr("pepper"))
        first = masker.mask(frame["email"])
        second = HashMasker(salt=SecretStr("pepper")).mask(frame["email"])
        assert first.iloc[0] == second.iloc[0]
        assert len(first.iloc[0]) == 16

    def test_hash_masker_salt_changes_the_output(self, frame: pd.DataFrame) -> None:
        one = HashMasker(salt=SecretStr("a")).mask(frame["email"]).iloc[0]
        two = HashMasker(salt=SecretStr("b")).mask(frame["email"]).iloc[0]
        assert one != two

    def test_hash_masker_never_exposes_the_salt(self) -> None:
        masker = HashMasker(salt=SecretStr("pepper"))
        assert "pepper" not in repr(masker)

    def test_hash_masker_reads_salt_from_environment(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UD_MASKING_SALT", "env-salt")
        assert HashMasker().mask(frame["email"]).iloc[0] == (
            HashMasker(salt=SecretStr("env-salt")).mask(frame["email"]).iloc[0]
        )

    @pytest.mark.parametrize("algorithm", ["md5", "sha1"])
    def test_weak_hash_algorithms_are_rejected(self, algorithm: str) -> None:
        with pytest.raises(ConfigurationError, match="not acceptable"):
            HashMasker(algorithm=algorithm)

    def test_unknown_hash_algorithm(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown hash algorithm"):
            HashMasker(algorithm="rot13")

    def test_token_masker_is_consistent_within_a_run(self) -> None:
        masker = TokenMasker()
        series = pd.Series(["a", "b", "a"])
        masked = masker.mask(series)
        assert masked.iloc[0] == masked.iloc[2]
        assert masked.iloc[0] != masked.iloc[1]

    def test_redact_masker(self, frame: pd.DataFrame) -> None:
        masked = RedactMasker().mask(frame["email"])
        assert masked.iloc[0] == "[REDACTED]"
        assert pd.isna(masked.iloc[2])

    def test_create_masker_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown masking strategy"):
            create_masker("nope")


class TestMaskingPolicy:
    def test_from_dict_with_both_forms(self, frame: pd.DataFrame) -> None:
        policy = MaskingPolicy.from_dict(
            {"email": "email", "phone": {"strategy": "phone", "keep_last": 2}}
        )
        masked, report = policy.apply(frame)
        assert masked["email"].iloc[0].startswith("j***@")
        assert masked["phone"].iloc[0].endswith("67")
        assert report["applied"] == {"email": "email", "phone": "phone"}

    def test_missing_columns_are_reported_not_fatal(self, frame: pd.DataFrame) -> None:
        policy = MaskingPolicy.from_dict({"absent": "hash"})
        _, report = policy.apply(frame)
        assert report["skipped"] == ["absent"]

    def test_rule_needs_a_strategy(self) -> None:
        with pytest.raises(ConfigurationError, match="needs a strategy"):
            MaskingPolicy.from_dict({"email": {"keep": 1}})

    def test_rule_must_be_a_string_or_mapping(self) -> None:
        with pytest.raises(ConfigurationError, match="Invalid masking rule"):
            MaskingPolicy.from_dict({"email": 12})

    def test_for_columns_helper(self, frame: pd.DataFrame) -> None:
        policy = MaskingPolicy.for_columns(["email", "name"], "redact")
        assert len(policy) == 2
        masked, _ = policy.apply(frame)
        assert masked["name"].iloc[0] == "[REDACTED]"

    def test_original_frame_is_untouched(self, frame: pd.DataFrame) -> None:
        before = frame["email"].tolist()
        MaskingPolicy.from_dict({"email": "hash"}).apply(frame)
        assert frame["email"].tolist() == before


class TestEnrichment:
    @pytest.fixture
    def lookup(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "country_code": ["IR", "DE"],
                "country_name": ["Iran", "Germany"],
                "region": ["Middle East", "Europe"],
            }
        )

    def test_lookup_enricher(self, frame: pd.DataFrame, lookup: pd.DataFrame) -> None:
        result = LookupEnricher(lookup, on="country_code").enrich(frame)
        assert result["region"].tolist() == ["Middle East", "Europe", "Middle East"]
        assert len(result) == len(frame)

    def test_lookup_enricher_selects_columns(
        self, frame: pd.DataFrame, lookup: pd.DataFrame
    ) -> None:
        result = LookupEnricher(lookup, on="country_code", columns=["region"]).enrich(frame)
        assert "country_name" not in result.columns

    def test_lookup_enricher_rejects_duplicate_keys(self, lookup: pd.DataFrame) -> None:
        duplicated = pd.concat([lookup, lookup], ignore_index=True)
        with pytest.raises(ConfigurationError, match="duplicate keys"):
            LookupEnricher(duplicated, on="country_code")

    def test_lookup_enricher_needs_a_key(self, lookup: pd.DataFrame) -> None:
        with pytest.raises(ConfigurationError, match="needs 'on'"):
            LookupEnricher(lookup)

    def test_lookup_enricher_validates_columns(self, lookup: pd.DataFrame) -> None:
        with pytest.raises(ConfigurationError, match="missing column"):
            LookupEnricher(lookup, on="country_code", columns=["nope"])

    def test_lookup_enricher_requires_the_key_in_the_dataset(self, lookup: pd.DataFrame) -> None:
        with pytest.raises(TransformationError, match="lookup key"):
            LookupEnricher(lookup, on="country_code").enrich(pd.DataFrame({"a": [1]}))

    def test_lookup_from_file(self, tmp_path: Path, frame: pd.DataFrame, lookup: pd.DataFrame) -> None:
        path = tmp_path / "lookup.csv"
        lookup.to_csv(path, index=False)
        result = LookupEnricher.from_file(path, on="country_code").enrich(frame)
        assert "region" in result.columns

    def test_mapping_enricher(self, frame: pd.DataFrame) -> None:
        result = MappingEnricher("country_code", "country", {"IR": "Iran"}, default="?").enrich(frame)
        assert result["country"].tolist() == ["Iran", "?", "Iran"]

    def test_mapping_enricher_requires_the_source(self, frame: pd.DataFrame) -> None:
        with pytest.raises(TransformationError):
            MappingEnricher("nope", "x", {}).enrich(frame)

    def test_derived_column_enricher(self, frame: pd.DataFrame) -> None:
        result = DerivedColumnEnricher("double", "amount * 2").enrich(frame)
        assert result["double"].tolist() == [20.0, 40.0, 60.0]

    def test_callable_enricher_caches_per_key(self, frame: pd.DataFrame) -> None:
        calls: list[str] = []

        def lookup(key: str) -> dict[str, str]:
            calls.append(key)
            return {"region": f"region-{key}"}

        enricher = CallableEnricher("country_code", lookup)
        result = enricher.enrich(frame)
        assert result["region"].tolist() == ["region-IR", "region-DE", "region-IR"]
        assert calls == ["IR", "DE"]
        assert enricher.cache_size == 2

    def test_callable_enricher_survives_failures(self, frame: pd.DataFrame) -> None:
        def failing(key: str) -> dict[str, str]:
            raise RuntimeError("service down")

        result = CallableEnricher("country_code", failing, columns=["region"]).enrich(frame)
        assert result["region"].isna().all()

    def test_callable_enricher_can_raise(self, frame: pd.DataFrame) -> None:
        def failing(key: str) -> dict[str, str]:
            raise RuntimeError("service down")

        enricher = CallableEnricher("country_code", failing, on_error="raise")
        with pytest.raises(TransformationError, match="lookup failed"):
            enricher.enrich(frame)

    def test_callable_enricher_validates_on_error(self) -> None:
        with pytest.raises(ConfigurationError, match="on_error"):
            CallableEnricher("a", lambda key: {}, on_error="ignore")

    def test_chain(self, frame: pd.DataFrame, lookup: pd.DataFrame) -> None:
        chain = EnrichmentChain(
            [LookupEnricher(lookup, on="country_code"), DerivedColumnEnricher("x", "amount + 1")]
        )
        result = chain.enrich(frame)
        assert {"region", "x"} <= set(result.columns)
        assert len(chain) == 2


class TestProfiler:
    def test_dataset_profile(self, frame: pd.DataFrame) -> None:
        profile = profile_dataset(frame, name="people", source="memory")
        assert profile.rows == 3
        assert profile.columns == 5
        assert profile.column("amount").numeric is not None
        assert profile.column("email").is_pii
        assert "Dataset Profile" in profile.summary()
        assert profile.to_dict()["rows"] == 3

    def test_numeric_statistics(self) -> None:
        column = profile_column(pd.Series([1.0, 2.0, 3.0, 100.0]), "v")
        assert column.numeric is not None
        assert column.numeric.minimum == 1.0
        assert column.numeric.maximum == 100.0
        assert column.numeric.zeros == 0

    def test_numeric_statistics_on_empty_column(self) -> None:
        column = profile_column(pd.Series([None, None], dtype="float64"), "v")
        assert column.numeric is not None
        assert column.numeric.to_dict()["mean"] is None

    def test_categorical_statistics(self) -> None:
        column = profile_column(pd.Series(["a", "a", "b", ""]), "c")
        assert column.categorical is not None
        assert column.categorical.top_values[0] == ("a", 2)
        assert column.categorical.empty_strings == 1

    def test_temporal_statistics(self) -> None:
        values = [f"2020-01-{day:02d}" for day in range(1, 20)] + ["2021-01-01", "bad"]
        column = profile_column(pd.Series(values), "d")
        assert column.temporal is not None
        assert column.temporal.unparseable == 1
        assert column.temporal.range_days == 366

    def test_duplicate_and_missing_percentages(self) -> None:
        frame = pd.DataFrame({"a": [1, 1, None]})
        profile = profile_dataset(frame)
        assert profile.duplicate_rows == 1
        assert profile.missing_pct == pytest.approx(33.33, abs=0.1)

    def test_empty_dataset(self) -> None:
        profile = profile_dataset(pd.DataFrame())
        assert profile.rows == 0
        assert profile.summary().startswith("Dataset Profile")
