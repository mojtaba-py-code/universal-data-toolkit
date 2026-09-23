"""API ingestion: authentication, retries, pagination and response validation."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from conftest import FakeSession, make_response
from universal_data.core.dataset import Dataset
from universal_data.core.exceptions import AuthenticationError, ExtractionError, SecurityError
from universal_data.ingestion.api import (
    APIClient,
    APIConfig,
    ApiKeyAuth,
    BasicAuth,
    BearerTokenAuth,
    CursorPaginator,
    LinkHeaderPaginator,
    NoAuth,
    OffsetPaginator,
    PageNumberPaginator,
    RateLimiter,
    RetryPolicy,
    _extract_rows,
)
from universal_data.security.network import URLPolicy
from universal_data.security.secrets import SecretStr

LOCAL_POLICY = URLPolicy(allow_private_networks=True)
URL = "https://api.example.test/items"


def build_client(responses: list[requests.Response], **kwargs: Any) -> APIClient:
    session = FakeSession(responses)
    config = kwargs.pop("config", APIConfig(url=URL))
    return APIClient(config, url_policy=LOCAL_POLICY, session=session, **kwargs)


class TestAuthStrategies:
    def test_no_auth_adds_nothing(self) -> None:
        headers: dict[str, str] = {}
        NoAuth().apply(headers, {})
        assert headers == {}

    def test_api_key_header(self) -> None:
        headers: dict[str, str] = {}
        ApiKeyAuth(SecretStr("k")).apply(headers, {})
        assert headers["X-API-Key"] == "k"

    def test_api_key_query_parameter(self) -> None:
        params: dict[str, Any] = {}
        ApiKeyAuth(SecretStr("k"), query_param="api_key").apply({}, params)
        assert params["api_key"] == "k"

    def test_bearer_token(self) -> None:
        headers: dict[str, str] = {}
        BearerTokenAuth(SecretStr("tok")).apply(headers, {})
        assert headers["Authorization"] == "Bearer tok"

    def test_basic_auth(self) -> None:
        headers: dict[str, str] = {}
        BasicAuth("user", SecretStr("pass")).apply(headers, {})
        assert headers["Authorization"].startswith("Basic ")
        assert "pass" not in headers["Authorization"]


class TestRetryPolicy:
    def test_backoff_grows_but_is_capped(self) -> None:
        policy = RetryPolicy(backoff_factor=1.0, max_backoff=4.0)
        assert 0 <= policy.delay_for(1) <= 1.0
        assert 0 <= policy.delay_for(5) <= 4.0

    def test_retry_after_header_wins(self) -> None:
        policy = RetryPolicy(max_backoff=10)
        assert policy.delay_for(1, retry_after=7.0) == 7.0

    def test_rate_limiter_without_limit_is_free(self) -> None:
        limiter = RateLimiter(None)
        limiter.wait()
        assert limiter.min_interval == 0.0


class TestFetching:
    def test_fetches_a_list_payload(self) -> None:
        client = build_client([make_response([{"id": 1}, {"id": 2}])])
        frame = client.fetch()
        assert len(frame) == 2

    def test_fetches_wrapped_payload(self) -> None:
        client = build_client([make_response({"data": [{"id": 1}], "total": 1})])
        assert len(client.fetch()) == 1

    def test_records_path(self) -> None:
        client = build_client(
            [make_response({"result": {"rows": [{"id": 1}, {"id": 2}]}})],
            config=APIConfig(url=URL, records_path="result.rows"),
        )
        assert len(client.fetch()) == 2

    def test_empty_response_gives_empty_frame(self) -> None:
        client = build_client([make_response([])])
        assert client.fetch().empty

    def test_max_records_truncates(self) -> None:
        client = build_client(
            [make_response([{"id": i} for i in range(10)])],
            config=APIConfig(url=URL, max_records=3),
        )
        assert len(client.fetch()) == 3

    def test_auth_headers_are_sent(self) -> None:
        session = FakeSession([make_response([{"id": 1}])])
        client = APIClient(
            APIConfig(url=URL),
            auth=BearerTokenAuth(SecretStr("tok")),
            url_policy=LOCAL_POLICY,
            session=session,
        )
        client.fetch()
        assert session.calls[0]["headers"]["Authorization"] == "Bearer tok"

    def test_dataset_from_api(self) -> None:
        client = build_client([make_response([{"id": 1}])])
        dataset = Dataset.from_api(client, name="items")
        assert dataset.n_rows == 1
        assert dataset.source == URL


class TestErrorHandling:
    def test_unauthorized_raises_authentication_error(self) -> None:
        client = build_client([make_response({"error": "nope"}, status=401)])
        with pytest.raises(AuthenticationError):
            client.fetch()

    def test_server_error_raises_extraction_error(self) -> None:
        client = build_client(
            [make_response({}, status=500)], retry=RetryPolicy(max_attempts=1)
        )
        with pytest.raises(ExtractionError, match="error status"):
            client.fetch()

    def test_retries_then_succeeds(self) -> None:
        responses = [
            make_response({}, status=503),
            make_response([{"id": 1}]),
        ]
        client = build_client(
            responses, retry=RetryPolicy(max_attempts=3, backoff_factor=0.0, max_backoff=0.0)
        )
        assert len(client.fetch()) == 1

    def test_connection_errors_are_retried_then_reported(self) -> None:
        class FailingSession:
            def request(self, *args: Any, **kwargs: Any) -> Any:
                raise requests.ConnectionError("boom")

            def close(self) -> None:
                return None

        client = APIClient(
            APIConfig(url=URL),
            url_policy=LOCAL_POLICY,
            session=FailingSession(),  # type: ignore[arg-type]
            retry=RetryPolicy(max_attempts=2, backoff_factor=0.0),
        )
        with pytest.raises(ExtractionError, match="failed after 2 attempts"):
            client.fetch()

    def test_non_json_response_is_rejected(self) -> None:
        client = build_client(
            [make_response("<html>hello</html>", headers={"Content-Type": "text/html"})]
        )
        with pytest.raises(ExtractionError, match="Expected a JSON response"):
            client.fetch()

    def test_malformed_json_is_reported(self) -> None:
        client = build_client([make_response("{not json")])
        with pytest.raises(ExtractionError, match="not valid JSON"):
            client.fetch()

    def test_response_size_limit(self) -> None:
        payload = [{"id": i, "padding": "x" * 100} for i in range(100)]
        client = build_client(
            [make_response(payload)], config=APIConfig(url=URL, max_response_bytes=128)
        )
        with pytest.raises(ExtractionError, match="size limit"):
            client.fetch()

    def test_redirect_target_is_validated(self) -> None:
        redirect = make_response(
            {}, status=302, headers={"Location": "http://169.254.169.254/latest/meta-data"}
        )
        session = FakeSession([redirect])
        client = APIClient(APIConfig(url=URL), url_policy=URLPolicy(), session=session)
        with pytest.raises(SecurityError):
            client.fetch()

    def test_redirect_without_location_is_rejected(self) -> None:
        response = make_response({}, status=302)
        response.headers.pop("Location", None)
        client = build_client([response])
        with pytest.raises(ExtractionError, match="Redirect without"):
            client.fetch()

    def test_url_is_validated_before_the_request(self) -> None:
        client = APIClient(
            APIConfig(url="https://127.0.0.1/secret"),
            session=FakeSession([]),
        )
        with pytest.raises(SecurityError):
            client.fetch()


class TestPagination:
    def test_page_number_paginator(self) -> None:
        responses = [
            make_response([{"id": 1}]),
            make_response([{"id": 2}]),
            make_response([]),
        ]
        session = FakeSession(responses)
        client = APIClient(
            APIConfig(url=URL),
            paginator=PageNumberPaginator(page_size=1, max_pages=5),
            url_policy=LOCAL_POLICY,
            session=session,
        )
        assert len(client.fetch()) == 2
        assert session.calls[1]["params"]["page"] == 2

    def test_page_number_respects_max_pages(self) -> None:
        responses = [make_response([{"id": i}]) for i in range(5)]
        client = build_client(
            responses, paginator=PageNumberPaginator(page_size=1, max_pages=2)
        )
        assert len(client.fetch()) == 2

    def test_offset_paginator_stops_on_short_page(self) -> None:
        responses = [
            make_response([{"id": 1}, {"id": 2}]),
            make_response([{"id": 3}]),
        ]
        client = build_client(responses, paginator=OffsetPaginator(page_size=2, max_pages=5))
        assert len(client.fetch()) == 3

    def test_cursor_paginator(self) -> None:
        responses = [
            make_response({"data": [{"id": 1}], "next_cursor": "abc"}),
            make_response({"data": [{"id": 2}], "next_cursor": None}),
        ]
        session = FakeSession(responses)
        client = APIClient(
            APIConfig(url=URL),
            paginator=CursorPaginator(),
            url_policy=LOCAL_POLICY,
            session=session,
        )
        assert len(client.fetch()) == 2
        assert session.calls[1]["params"]["cursor"] == "abc"

    def test_link_header_paginator(self) -> None:
        first = make_response(
            [{"id": 1}],
            headers={"Link": '<https://api.example.test/items?page=2>; rel="next"'},
        )
        second = make_response([{"id": 2}])
        client = build_client([first, second], paginator=LinkHeaderPaginator(max_pages=5))
        assert len(client.fetch()) == 2


class TestRecordExtraction:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ([{"a": 1}], 1),
            ({"data": [{"a": 1}, {"a": 2}]}, 2),
            ({"results": []}, 0),
            ({"a": 1}, 1),
            ("text", 0),
        ],
    )
    def test_extract_rows(self, payload: Any, expected: int) -> None:
        assert len(_extract_rows(payload)) == expected

    def test_extract_rows_with_missing_path(self) -> None:
        assert _extract_rows({"a": 1}, "b.c") == []


def test_client_closes_owned_session() -> None:
    with APIClient(APIConfig(url=URL), url_policy=LOCAL_POLICY) as client:
        assert client._owns_session is True
