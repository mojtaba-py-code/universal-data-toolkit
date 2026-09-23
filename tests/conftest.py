"""Shared fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from universal_data.core.dataset import Dataset
from universal_data.generator.synthetic import QualityIssues, SyntheticDataGenerator
from universal_data.security.paths import PathPolicy


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An isolated directory used as the pipeline workspace."""
    return tmp_path


@pytest.fixture
def policy(tmp_path: Path) -> PathPolicy:
    return PathPolicy.confined_to(tmp_path)


@pytest.fixture
def customers_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 5],
            "name": ["  Ali ", "Sara", "REZA", "maryam", None, None],
            "email": [
                "ali@example.com",
                "sara@example.com",
                "not-an-email",
                "maryam@example.com",
                None,
                None,
            ],
            "age": [34, 17, 45, 29, 51, 51],
            "country": ["IR", "DE", "IR", "FR", "DE", "DE"],
            "salary": [1000.0, 2500.0, -50.0, 3200.0, 99999.0, 99999.0],
            "signup_date": [
                "2024-01-15",
                "2024-02-20",
                "not a date",
                "2024-03-10",
                "2024-04-05",
                "2024-04-05",
            ],
        }
    )


@pytest.fixture
def customers_dataset(customers_frame: pd.DataFrame) -> Dataset:
    return Dataset(customers_frame, name="customers")


@pytest.fixture
def customers_csv(tmp_path: Path, customers_frame: pd.DataFrame) -> Path:
    path = tmp_path / "customers.csv"
    customers_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def dirty_dataset() -> Dataset:
    generator = SyntheticDataGenerator(seed=99)
    return Dataset(generator.customers(200, issues=QualityIssues.realistic()), name="dirty")


@pytest.fixture
def schema_file(tmp_path: Path) -> Path:
    path = tmp_path / "customer_schema.yaml"
    path.write_text(
        """
name: customer
primary_key: [customer_id]
schema:
  customer_id:
    type: integer
    nullable: false
    unique: true
  name:
    type: string
    required: false
    pii: true
  email:
    type: email
    required: true
    pii: true
  age:
    type: integer
    min: 18
    max: 100
  country:
    type: categorical
    allowed: [IR, DE, FR]
  salary:
    type: float
    min: 0
  signup_date:
    type: datetime
""".strip(),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def rules_file(tmp_path: Path) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """
rules:
  - name: adult
    condition: "age >= 18"
  - name: valid_email
    condition: "is_email(email)"
  - name: positive_salary
    condition: "salary >= 0"
    severity: warning
""".strip(),
        encoding="utf-8",
    )
    return path


def make_response(
    payload: Any,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    url: str = "https://api.example.test/items",
) -> requests.Response:
    """Build a ready-to-read ``requests.Response`` without touching the network."""
    response = requests.Response()
    response.status_code = status
    response.url = url
    response.headers.update({"Content-Type": "application/json", **(headers or {})})
    body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
    response._content = body.encode("utf-8") if isinstance(body, str) else body
    response._content_consumed = True
    response.encoding = "utf-8"
    return response


class FakeSession:
    """Returns queued responses and records the requests it was given."""

    def __init__(self, responses: list[requests.Response]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError("FakeSession ran out of queued responses")
        return self.responses.pop(0)

    def close(self) -> None:
        return None


@pytest.fixture
def fake_session_factory() -> Any:
    return FakeSession
