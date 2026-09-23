"""REST API ingestion.

The client covers the parts that are always rewritten from scratch in one-off
scripts: authentication, retries with exponential backoff, rate limiting,
pagination and response validation.

Every URL - including redirect targets - passes through
:class:`~universal_data.security.network.URLPolicy` before a connection is made,
and credentials only ever exist as :class:`SecretStr` values.
"""

from __future__ import annotations

import base64
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

from universal_data.core.exceptions import AuthenticationError, ExtractionError
from universal_data.observability.logging import get_logger
from universal_data.security.network import URLPolicy
from universal_data.security.secrets import SecretStr

logger = get_logger(__name__)

DEFAULT_TIMEOUT = (10.0, 60.0)  # (connect, read) seconds
MAX_RESPONSE_BYTES = 256 * 1024 * 1024
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class AuthStrategy(ABC):
    """Applies credentials to an outgoing request."""

    @abstractmethod
    def apply(self, headers: dict[str, str], params: dict[str, Any]) -> None:
        """Mutate *headers*/*params* in place."""

    def refresh(self) -> None:
        """Hook for token based strategies; no-op by default."""


class NoAuth(AuthStrategy):
    def apply(self, headers: dict[str, str], params: dict[str, Any]) -> None:
        return


@dataclass
class ApiKeyAuth(AuthStrategy):
    key: SecretStr
    header: str | None = "X-API-Key"
    query_param: str | None = None

    def apply(self, headers: dict[str, str], params: dict[str, Any]) -> None:
        if self.query_param:
            params[self.query_param] = self.key.get_secret_value()
        else:
            headers[self.header or "X-API-Key"] = self.key.get_secret_value()


@dataclass
class BearerTokenAuth(AuthStrategy):
    token: SecretStr

    def apply(self, headers: dict[str, str], params: dict[str, Any]) -> None:
        headers["Authorization"] = f"Bearer {self.token.get_secret_value()}"


@dataclass
class BasicAuth(AuthStrategy):
    username: str
    password: SecretStr

    def apply(self, headers: dict[str, str], params: dict[str, Any]) -> None:
        raw = f"{self.username}:{self.password.get_secret_value()}".encode()
        headers["Authorization"] = f"Basic {base64.b64encode(raw).decode('ascii')}"


@dataclass
class OAuth2ClientCredentials(AuthStrategy):
    """Client-credentials grant with in-memory token caching."""

    token_url: str
    client_id: str
    client_secret: SecretStr
    scope: str | None = None
    url_policy: URLPolicy = field(default_factory=URLPolicy)
    _token: SecretStr | None = field(default=None, init=False, repr=False)
    _expires_at: float = field(default=0.0, init=False, repr=False)

    def apply(self, headers: dict[str, str], params: dict[str, Any]) -> None:
        if self._token is None or time.monotonic() >= self._expires_at:
            self.refresh()
        assert self._token is not None
        headers["Authorization"] = f"Bearer {self._token.get_secret_value()}"

    def refresh(self) -> None:
        url = self.url_policy.validate(self.token_url)
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret.get_secret_value(),
        }
        if self.scope:
            payload["scope"] = self.scope
        try:
            response = requests.post(url, data=payload, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException as exc:
            raise AuthenticationError(f"Token request failed: {exc}", token_url=self.token_url) from exc
        if response.status_code >= 400:
            raise AuthenticationError(
                "Token endpoint rejected the credentials", status=response.status_code
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise AuthenticationError("Token endpoint returned a non-JSON body") from exc
        token = body.get("access_token")
        if not token:
            raise AuthenticationError("Token endpoint response has no access_token")
        self._token = SecretStr(str(token))
        self._expires_at = time.monotonic() + float(body.get("expires_in", 3600)) - 30


@dataclass
class RetryPolicy:
    """Exponential backoff with full jitter."""

    max_attempts: int = 3
    backoff_factor: float = 0.5
    max_backoff: float = 30.0
    statuses: frozenset[int] = RETRYABLE_STATUSES
    respect_retry_after: bool = True

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None and self.respect_retry_after:
            return min(retry_after, self.max_backoff)
        base = min(self.backoff_factor * (2 ** (attempt - 1)), self.max_backoff)
        # Retry jitter, not a secret.
        return random.uniform(0, base)  # noqa: S311  # nosec B311


class RateLimiter:
    """Simple client-side throttle: at most *rate* requests per second."""

    def __init__(self, requests_per_second: float | None = None) -> None:
        self.min_interval = 1.0 / requests_per_second if requests_per_second else 0.0
        self._last_call = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


class Paginator(ABC):
    """Decides how to walk from one page to the next."""

    @abstractmethod
    def next_params(
        self, payload: Any, response: requests.Response, params: dict[str, Any], page: int
    ) -> dict[str, Any] | None:
        """Return params for the next page, or ``None`` when finished."""

    def next_url(self, payload: Any, response: requests.Response, page: int) -> str | None:
        return None

    def initial_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return dict(params)


class NoPagination(Paginator):
    def next_params(
        self, payload: Any, response: requests.Response, params: dict[str, Any], page: int
    ) -> dict[str, Any] | None:
        return None


@dataclass
class PageNumberPaginator(Paginator):
    page_param: str = "page"
    size_param: str | None = "per_page"
    page_size: int = 100
    start_page: int = 1
    max_pages: int = 100

    def initial_params(self, params: dict[str, Any]) -> dict[str, Any]:
        merged = dict(params)
        merged.setdefault(self.page_param, self.start_page)
        if self.size_param:
            merged.setdefault(self.size_param, self.page_size)
        return merged

    def next_params(
        self, payload: Any, response: requests.Response, params: dict[str, Any], page: int
    ) -> dict[str, Any] | None:
        if page >= self.max_pages or not _payload_has_rows(payload):
            return None
        nxt = dict(params)
        nxt[self.page_param] = int(params.get(self.page_param, self.start_page)) + 1
        return nxt


@dataclass
class OffsetPaginator(Paginator):
    offset_param: str = "offset"
    limit_param: str = "limit"
    page_size: int = 100
    max_pages: int = 100

    def initial_params(self, params: dict[str, Any]) -> dict[str, Any]:
        merged = dict(params)
        merged.setdefault(self.offset_param, 0)
        merged.setdefault(self.limit_param, self.page_size)
        return merged

    def next_params(
        self, payload: Any, response: requests.Response, params: dict[str, Any], page: int
    ) -> dict[str, Any] | None:
        rows = _extract_rows(payload)
        if page >= self.max_pages or len(rows) < int(params.get(self.limit_param, self.page_size)):
            return None
        nxt = dict(params)
        nxt[self.offset_param] = int(params.get(self.offset_param, 0)) + len(rows)
        return nxt


@dataclass
class CursorPaginator(Paginator):
    """Follows a cursor value found at *cursor_path* inside the response body."""

    cursor_param: str = "cursor"
    cursor_path: str = "next_cursor"
    max_pages: int = 100

    def next_params(
        self, payload: Any, response: requests.Response, params: dict[str, Any], page: int
    ) -> dict[str, Any] | None:
        if page >= self.max_pages:
            return None
        cursor: Any = payload
        for key in self.cursor_path.split("."):
            if not isinstance(cursor, Mapping) or key not in cursor:
                return None
            cursor = cursor[key]
        if not cursor:
            return None
        nxt = dict(params)
        nxt[self.cursor_param] = cursor
        return nxt


@dataclass
class LinkHeaderPaginator(Paginator):
    """Uses the RFC 5988 ``Link: <...>; rel="next"`` header (GitHub style)."""

    max_pages: int = 100

    def next_params(
        self, payload: Any, response: requests.Response, params: dict[str, Any], page: int
    ) -> dict[str, Any] | None:
        return None

    def next_url(self, payload: Any, response: requests.Response, page: int) -> str | None:
        if page >= self.max_pages:
            return None
        link = response.links.get("next")
        return str(link["url"]) if link and "url" in link else None


def _extract_rows(payload: Any, records_path: str | None = None) -> list[Any]:
    """Find the list of records inside a JSON payload."""
    if records_path:
        current = payload
        for key in records_path.split("."):
            if not isinstance(current, Mapping) or key not in current:
                return []
            current = current[key]
        return list(current) if isinstance(current, Sequence) and not isinstance(current, str) else []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("data", "results", "items", "records", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        lists = [value for value in payload.values() if isinstance(value, list)]
        if len(lists) == 1:
            return lists[0]
        return [payload]
    return []


def _payload_has_rows(payload: Any) -> bool:
    return bool(_extract_rows(payload))


@dataclass
class APIConfig:
    """Everything needed to fetch a dataset from an HTTP endpoint."""

    url: str
    method: str = "GET"
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    json_body: dict[str, Any] | None = None
    timeout: tuple[float, float] = DEFAULT_TIMEOUT
    verify_tls: bool = True
    records_path: str | None = None
    max_records: int | None = None
    max_response_bytes: int = MAX_RESPONSE_BYTES
    user_agent: str = "universal-data-toolkit/1.0"


class APIClient:
    """Fetches JSON records from a REST endpoint."""

    def __init__(
        self,
        config: APIConfig,
        *,
        auth: AuthStrategy | None = None,
        retry: RetryPolicy | None = None,
        paginator: Paginator | None = None,
        rate_limit: float | None = None,
        url_policy: URLPolicy | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.auth = auth or NoAuth()
        self.retry = retry or RetryPolicy()
        self.paginator = paginator or NoPagination()
        self.limiter = RateLimiter(rate_limit)
        self.url_policy = url_policy or URLPolicy()
        self._session = session or requests.Session()
        self._owns_session = session is None

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> APIClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _request(self, url: str, params: dict[str, Any]) -> tuple[Any, requests.Response]:
        headers = {"Accept": "application/json", "User-Agent": self.config.user_agent}
        headers.update(self.config.headers)
        request_params = dict(params)
        self.auth.apply(headers, request_params)

        target = self.url_policy.validate(url)
        last_error: Exception | None = None

        for attempt in range(1, self.retry.max_attempts + 1):
            self.limiter.wait()
            try:
                response = self._session.request(
                    self.config.method.upper(),
                    target,
                    headers=headers,
                    params=request_params,
                    json=self.config.json_body,
                    timeout=self.config.timeout,
                    verify=self.config.verify_tls,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("Request attempt %s failed: %s", attempt, exc)
                if attempt == self.retry.max_attempts:
                    break
                time.sleep(self.retry.delay_for(attempt))
                continue

            with response:
                if response.status_code in REDIRECT_STATUSES:
                    location = response.headers.get("Location", "")
                    if not location:
                        raise ExtractionError("Redirect without a Location header", url=target)
                    target = self.url_policy.validate(requests.compat.urljoin(target, location))
                    logger.debug("Following validated redirect")
                    continue

                if response.status_code in self.retry.statuses and attempt < self.retry.max_attempts:
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    delay = self.retry.delay_for(attempt, retry_after)
                    logger.warning(
                        "HTTP %s from the API, retrying in %.1fs", response.status_code, delay
                    )
                    time.sleep(delay)
                    continue

                if response.status_code in (401, 403):
                    raise AuthenticationError(
                        "The API rejected the credentials", status=response.status_code
                    )
                if response.status_code >= 400:
                    raise ExtractionError(
                        "API returned an error status",
                        status=response.status_code,
                        reason=response.reason,
                    )
                return self._decode(response), response

        raise ExtractionError(
            f"API request failed after {self.retry.max_attempts} attempts: {last_error}",
            url=self.config.url,
        )

    def _decode(self, response: requests.Response) -> Any:
        content_type = response.headers.get("Content-Type", "")
        body = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            body.extend(chunk)
            if len(body) > self.config.max_response_bytes:
                raise ExtractionError(
                    "Response exceeds the configured size limit",
                    limit=self.config.max_response_bytes,
                )
        if not body:
            return []
        text = body.decode(response.encoding or "utf-8", errors="replace")
        if "json" not in content_type.lower() and text.lstrip()[:1] not in ("{", "["):
            raise ExtractionError(
                "Expected a JSON response", content_type=content_type or "<missing>"
            )
        import json

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"Response is not valid JSON: {exc.msg}") from exc

    def iter_pages(self) -> Iterator[list[Any]]:
        """Yield the records of each page in turn."""
        params = self.paginator.initial_params(self.config.params)
        url = self.config.url
        page = 0
        collected = 0

        while True:
            page += 1
            payload, response = self._request(url, params)
            rows = _extract_rows(payload, self.config.records_path)
            if self.config.max_records is not None:
                remaining = self.config.max_records - collected
                if remaining <= 0:
                    return
                rows = rows[:remaining]
            collected += len(rows)
            if rows:
                yield rows
            if self.config.max_records is not None and collected >= self.config.max_records:
                return

            next_url = self.paginator.next_url(payload, response, page)
            if next_url:
                url = next_url
                continue
            next_params = self.paginator.next_params(payload, response, params, page)
            if next_params is None:
                return
            params = next_params

    def fetch_records(self) -> list[Any]:
        records: list[Any] = []
        for page in self.iter_pages():
            records.extend(page)
        return records

    def fetch(self) -> pd.DataFrame:
        """Fetch every page and return one DataFrame."""
        records = self.fetch_records()
        if not records:
            return pd.DataFrame()
        return pd.json_normalize(records)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
