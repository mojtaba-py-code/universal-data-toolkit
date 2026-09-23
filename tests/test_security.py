"""Security tests: path traversal, secrets, URL validation and redaction."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from universal_data.core.exceptions import ConfigurationError, SecurityError
from universal_data.security.network import URLPolicy, validate_url
from universal_data.security.paths import PathPolicy, sanitize_filename
from universal_data.security.redaction import (
    RedactingFilter,
    redact_mapping,
    redact_text,
)
from universal_data.security.secrets import (
    SecretResolver,
    SecretStr,
    load_dotenv,
    parse_dotenv,
)


class TestPathPolicy:
    def test_resolves_relative_path_inside_workspace(self, tmp_path: Path) -> None:
        target = tmp_path / "data.csv"
        target.write_text("a,b\n1,2\n", encoding="utf-8")
        policy = PathPolicy.confined_to(tmp_path)
        assert policy.resolve_input("data.csv") == target.resolve()

    @pytest.mark.parametrize(
        "attack",
        [
            "../../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "subdir/../../outside.csv",
        ],
    )
    def test_rejects_traversal(self, tmp_path: Path, attack: str) -> None:
        policy = PathPolicy.confined_to(tmp_path / "workspace")
        with pytest.raises(SecurityError, match="escapes the allowed workspace"):
            policy.resolve_input(attack, must_exist=False)

    def test_rejects_absolute_path_outside_workspace(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.csv"
        policy = PathPolicy.confined_to(tmp_path)
        with pytest.raises(SecurityError):
            policy.resolve_input(outside, must_exist=False)

    def test_rejects_nul_byte(self) -> None:
        with pytest.raises(SecurityError, match="NUL"):
            PathPolicy().resolve_input("data\x00.csv", must_exist=False)

    def test_rejects_empty_path(self) -> None:
        with pytest.raises(SecurityError, match="Empty path"):
            PathPolicy().resolve_input("   ", must_exist=False)

    def test_rejects_unexpected_extension(self, tmp_path: Path) -> None:
        policy = PathPolicy(allowed_suffixes=frozenset({".csv", ".parquet"}))
        with pytest.raises(SecurityError, match="extension is not permitted"):
            policy.resolve_output(tmp_path / "payload.exe")

    def test_rejects_missing_input(self, tmp_path: Path) -> None:
        with pytest.raises(SecurityError, match="does not exist"):
            PathPolicy().resolve_input(tmp_path / "nope.csv")

    def test_rejects_directory_as_input(self, tmp_path: Path) -> None:
        with pytest.raises(SecurityError, match="not a regular file"):
            PathPolicy().resolve_input(tmp_path)

    def test_enforces_size_limit(self, tmp_path: Path) -> None:
        target = tmp_path / "big.csv"
        target.write_text("x" * 1024, encoding="utf-8")
        policy = PathPolicy(max_bytes=10)
        with pytest.raises(SecurityError, match="size limit"):
            policy.resolve_input(target)

    def test_output_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "out.csv"
        resolved = PathPolicy().resolve_output(target)
        assert resolved.parent.is_dir()

    def test_output_rejects_directory(self, tmp_path: Path) -> None:
        with pytest.raises(SecurityError, match="is a directory"):
            PathPolicy().resolve_output(tmp_path)

    @pytest.mark.parametrize("name", ["NUL", "nul", "con.csv", "COM1.parquet", "aux"])
    def test_rejects_windows_device_names(self, tmp_path: Path, name: str) -> None:
        """Writing to <workspace>/NUL silently discards the data on Windows."""
        policy = PathPolicy.confined_to(tmp_path)
        with pytest.raises(SecurityError, match="reserved device name"):
            policy.resolve_output(name)

    def test_ordinary_names_are_not_mistaken_for_devices(self, tmp_path: Path) -> None:
        policy = PathPolicy.confined_to(tmp_path)
        assert policy.resolve_output("console.csv").name == "console.csv"
        assert policy.resolve_output("connections.parquet").name == "connections.parquet"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("../../evil.csv", "evil.csv"),
            ("a/b/c.csv", "c.csv"),
            ("..\\..\\windows\\file.csv", "file.csv"),
            ("CON.txt", "CON_.txt"),
            ("", "file"),
            ("  spaced  .csv", "spaced_.csv"),
        ],
    )
    def test_sanitize_filename(self, raw: str, expected: str) -> None:
        assert sanitize_filename(raw) == expected


class TestURLPolicy:
    def test_rejects_non_http_scheme(self) -> None:
        with pytest.raises(SecurityError, match="http and https"):
            validate_url("file:///etc/passwd")

    def test_rejects_plain_http_by_default(self) -> None:
        with pytest.raises(SecurityError, match="Plain HTTP"):
            validate_url("http://example.com/data")

    def test_allows_http_when_requested(self) -> None:
        policy = URLPolicy(allow_http=True, allow_private_networks=True)
        assert policy.validate("http://localhost:8000/data").startswith("http://")

    @pytest.mark.parametrize(
        "url",
        [
            "https://127.0.0.1/admin",
            "https://10.0.0.5/internal",
            "https://192.168.1.1/router",
            "https://[::1]/local",
        ],
    )
    def test_blocks_private_addresses(self, url: str) -> None:
        with pytest.raises(SecurityError, match="non-public address"):
            validate_url(url)

    def test_blocks_cloud_metadata_hostname(self) -> None:
        with pytest.raises(SecurityError, match="metadata"):
            validate_url("https://metadata.google.internal/computeMetadata/v1/")

    def test_enforces_host_allowlist(self) -> None:
        policy = URLPolicy(allowed_hosts=frozenset({"api.example.com"}), allow_private_networks=True)
        assert policy.validate("https://api.example.com/v1")
        assert policy.validate("https://eu.api.example.com/v1")
        with pytest.raises(SecurityError, match="allow-list"):
            policy.validate("https://evil.test/v1")

    def test_rejects_url_without_host(self) -> None:
        with pytest.raises(SecurityError):
            validate_url("https:///no-host")

    def test_rejects_empty_url(self) -> None:
        with pytest.raises(SecurityError, match="Empty URL"):
            validate_url("")


class TestSecrets:
    def test_secret_never_prints_its_value(self) -> None:
        secret = SecretStr("super-secret-token")
        assert "super-secret" not in repr(secret)
        assert "super-secret" not in str(secret)
        assert f"{secret}" == "***"
        assert secret.get_secret_value() == "super-secret-token"

    def test_secret_equality_and_hash(self) -> None:
        assert SecretStr("a") == SecretStr("a")
        assert SecretStr("a") != "a"
        assert len({SecretStr("a"), SecretStr("a")}) == 1

    def test_resolver_reads_from_environment(self) -> None:
        resolver = SecretResolver({"DB_PASSWORD": "hunter2"})
        assert resolver.get("DB_PASSWORD").get_secret_value() == "hunter2"
        assert resolver.get("MISSING") is None

    def test_resolver_requires_missing_secret(self) -> None:
        resolver = SecretResolver({})
        with pytest.raises(ConfigurationError, match="Required secret"):
            resolver.get("MISSING", required=True)

    @pytest.mark.parametrize("reference", ["env:TOKEN", "${TOKEN}"])
    def test_resolve_reference_forms(self, reference: str) -> None:
        resolver = SecretResolver({"TOKEN": "abc123"})
        assert resolver.resolve(reference) == "abc123"

    def test_resolve_inline_with_default(self) -> None:
        resolver = SecretResolver({})
        assert resolver.resolve("${HOST:-localhost}") == "localhost"

    def test_resolve_unknown_variable_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="undefined environment variable"):
            SecretResolver({}).resolve("${NOT_SET}")

    def test_resolve_mapping_is_recursive(self) -> None:
        resolver = SecretResolver({"USER": "app", "PASS": "s3cr3t"})
        resolved = resolver.resolve_mapping(
            {"db": {"user": "${USER}", "password": "env:PASS"}, "hosts": ["${USER}"]}
        )
        assert resolved == {"db": {"user": "app", "password": "s3cr3t"}, "hosts": ["app"]}

    def test_parse_dotenv(self) -> None:
        parsed = parse_dotenv(
            "# comment\nexport A=1\nB='two'\nC=\"three\"\nnot a pair\n=nokey\n"
        )
        assert parsed == {"A": "1", "B": "two", "C": "three"}

    def test_load_dotenv_does_not_override_existing(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("SAMPLE_KEY=from_file\n", encoding="utf-8")
        environ: dict[str, str] = {}
        import os

        os.environ.pop("SAMPLE_KEY", None)
        loaded = load_dotenv(env_file)
        assert loaded == {"SAMPLE_KEY": "from_file"}
        assert os.environ["SAMPLE_KEY"] == "from_file"
        os.environ["SAMPLE_KEY"] = "kept"
        load_dotenv(env_file)
        assert os.environ["SAMPLE_KEY"] == "kept"
        os.environ.pop("SAMPLE_KEY", None)
        assert environ == {}

    def test_load_dotenv_ignores_missing_file(self, tmp_path: Path) -> None:
        assert load_dotenv(tmp_path / "absent.env") == {}


class TestRedaction:
    @pytest.mark.parametrize(
        ("raw", "must_not_contain"),
        [
            ("postgresql://user:s3cret@db:5432/app", "s3cret"),
            ("Authorization: Bearer abcdef1234567890", "abcdef1234567890"),
            ("password=hunter2 and more", "hunter2"),
            ("https://api.test/v1?api_key=KEY12345&page=2", "KEY12345"),
        ],
    )
    def test_redact_text(self, raw: str, must_not_contain: str) -> None:
        assert must_not_contain not in redact_text(raw)

    def test_redact_text_keeps_harmless_content(self) -> None:
        assert redact_text("rows=100 status=ok") == "rows=100 status=ok"

    def test_redact_mapping(self) -> None:
        payload = {
            "user": "app",
            "password": "hunter2",
            "nested": {"api_key": "abc", "size": 3},
            "items": [{"token": "xyz"}, "plain"],
        }
        redacted = redact_mapping(payload)
        assert redacted["password"] == "***"
        assert redacted["nested"]["api_key"] == "***"
        assert redacted["nested"]["size"] == 3
        assert redacted["items"][0]["token"] == "***"

    def test_logging_filter_scrubs_records(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = logging.getLogger("universal_data.test_redaction")
        logger.addFilter(RedactingFilter())
        with caplog.at_level(logging.INFO, logger=logger.name):
            logger.info("connecting to postgresql://app:topsecret@db/app")
        assert "topsecret" not in caplog.text
        logger.filters.clear()
