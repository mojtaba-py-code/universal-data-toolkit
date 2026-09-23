"""Redaction of secrets before anything reaches a log file or a report."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

MASK = "***"

SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "client_secret",
        "cookie",
        "credentials",
        "dsn",
        "password",
        "passwd",
        "private_key",
        "pwd",
        "refresh_token",
        "secret",
        "session_key",
        "token",
        "x-api-key",
    }
)

_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-z0-9+.\-]+://)(?P<user>[^:/?#@\s]+):(?P<pw>[^@\s]+)@")
_BEARER = re.compile(r"(?i)\b(bearer|token)\s+[A-Za-z0-9._\-~+/=]{8,}")
_KEY_VALUE = re.compile(
    r"(?i)\b(" + "|".join(sorted(SENSITIVE_KEYS)) + r")\b\s*[=:]\s*(\"[^\"]*\"|'[^']*'|[^\s,;&]+)"
)
_QUERY_PARAM = re.compile(
    r"(?i)([?&](?:" + "|".join(sorted(SENSITIVE_KEYS)) + r"))=([^&\s]+)"
)


def redact_text(text: str) -> str:
    """Mask credentials embedded in a free-form string."""
    if not text:
        return text
    out = _URL_CREDENTIALS.sub(lambda m: f"{m.group('scheme')}{m.group('user')}:{MASK}@", text)
    out = _QUERY_PARAM.sub(lambda m: f"{m.group(1)}={MASK}", out)
    # Bearer first: otherwise "Authorization: Bearer <token>" matches the generic
    # key/value rule, masks the word "Bearer" and leaves the token in the log.
    out = _BEARER.sub(lambda m: f"{m.group(1)} {MASK}", out)
    return _KEY_VALUE.sub(lambda m: f"{m.group(1)}={MASK}", out)


def redact_mapping(data: Mapping[str, Any], *, depth: int = 0) -> dict[str, Any]:
    """Return a copy of *data* with sensitive keys and values masked."""
    if depth > 8:
        return {"...": "truncated"}
    result: dict[str, Any] = {}
    for key, value in data.items():
        if str(key).lower().replace("-", "_") in SENSITIVE_KEYS:
            result[key] = MASK
        elif isinstance(value, Mapping):
            result[key] = redact_mapping(value, depth=depth + 1)
        elif isinstance(value, (list, tuple)):
            result[key] = [
                redact_mapping(v, depth=depth + 1) if isinstance(v, Mapping) else redact_value(v)
                for v in value
            ]
        else:
            result[key] = redact_value(value)
    return result


def redact_value(value: Any) -> Any:
    return redact_text(value) if isinstance(value, str) else value


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs secrets from messages, args and extras."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_mapping(record.args)
            else:
                record.args = tuple(redact_value(arg) for arg in record.args)
        for key, value in list(record.__dict__.items()):
            if key.lower() in SENSITIVE_KEYS:
                record.__dict__[key] = MASK
            elif isinstance(value, Mapping) and key not in ("args",):
                record.__dict__[key] = redact_mapping(value)
        return True
