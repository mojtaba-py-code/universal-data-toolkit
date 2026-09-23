"""Synthetic dataset generator.

Used by the examples, the benchmarks and the test suite.  The point is not
realistic-looking names but *realistically broken* data: the generator can inject
the specific defects the toolkit is built to find, at a controlled rate, with a
fixed seed so results are reproducible.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

FIRST_NAMES = [
    "Ali", "Sara", "Reza", "Maryam", "Hassan", "Leila", "Omar", "Nadia", "John", "Emma",
    "Lucas", "Sofia", "Mateo", "Chen", "Yuki", "Anna", "David", "Fatima", "Ivan", "Elena",
]
LAST_NAMES = [
    "Karimi", "Ahmadi", "Hosseini", "Rezaei", "Smith", "Johnson", "Muller", "Rossi",
    "Garcia", "Silva", "Novak", "Kowalski", "Tanaka", "Wang", "Ibrahim", "Petrov",
]
COUNTRIES = [
    ("IR", "Iran", "Middle East"),
    ("DE", "Germany", "Europe"),
    ("FR", "France", "Europe"),
    ("US", "United States", "North America"),
    ("BR", "Brazil", "South America"),
    ("JP", "Japan", "Asia"),
    ("AE", "United Arab Emirates", "Middle East"),
    ("TR", "Turkey", "Europe"),
]
DOMAINS = ["example.com", "mail.com", "test.org", "company.net"]
CATEGORIES = ["electronics", "books", "clothing", "home", "sports", "toys"]
DEPARTMENTS = ["engineering", "sales", "marketing", "support", "finance"]
STATUSES = ["new", "paid", "shipped", "delivered", "cancelled"]
CHANNELS = ["web", "mobile", "store", "partner"]


@dataclass
class QualityIssues:
    """Defect rates to inject, each expressed as a share of rows."""

    missing: float = 0.0
    duplicates: float = 0.0
    invalid_emails: float = 0.0
    invalid_dates: float = 0.0
    negative_values: float = 0.0
    outliers: float = 0.0
    inconsistent_formatting: float = 0.0
    mixed_types: float = 0.0

    @classmethod
    def clean(cls) -> QualityIssues:
        return cls()

    @classmethod
    def realistic(cls) -> QualityIssues:
        """The defect mix a typical export from a business system carries."""
        return cls(
            missing=0.06,
            duplicates=0.03,
            invalid_emails=0.04,
            invalid_dates=0.03,
            negative_values=0.02,
            outliers=0.01,
            inconsistent_formatting=0.10,
            mixed_types=0.02,
        )

    @classmethod
    def severe(cls) -> QualityIssues:
        return cls(
            missing=0.20,
            duplicates=0.10,
            invalid_emails=0.15,
            invalid_dates=0.10,
            negative_values=0.08,
            outliers=0.05,
            inconsistent_formatting=0.30,
            mixed_types=0.08,
        )


class SyntheticDataGenerator:
    """Generates related customer, product, order, transaction and employee data."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        # Seeded, reproducible test data - deliberately not a CSPRNG.
        self.random = random.Random(seed)  # noqa: S311  # nosec B311
        self.numpy = np.random.default_rng(seed)

    # -- building blocks ----------------------------------------------------

    def _name(self) -> tuple[str, str]:
        return self.random.choice(FIRST_NAMES), self.random.choice(LAST_NAMES)

    def _email(self, first: str, last: str, index: int) -> str:
        domain = self.random.choice(DOMAINS)
        return f"{first.lower()}.{last.lower()}{index}@{domain}"

    def _date(self, start_days_ago: int = 900, end_days_ago: int = 0) -> datetime:
        offset = self.random.randint(end_days_ago, start_days_ago)
        return datetime.now() - timedelta(days=offset, minutes=self.random.randint(0, 1440))

    def _phone(self) -> str:
        return f"+{self.random.randint(1, 99)}{self.random.randint(1000000000, 9999999999)}"

    # -- datasets -----------------------------------------------------------

    def customers(self, rows: int = 1_000, issues: QualityIssues | None = None) -> pd.DataFrame:
        records = []
        for index in range(1, rows + 1):
            first, last = self._name()
            code, country, _ = self.random.choice(COUNTRIES)
            records.append(
                {
                    "customer_id": index,
                    "first_name": first,
                    "last_name": last,
                    "email": self._email(first, last, index),
                    "phone": self._phone(),
                    "age": self.random.randint(18, 78),
                    "country_code": code,
                    "country": country,
                    "city": f"City{self.random.randint(1, 60)}",
                    "signup_date": self._date().strftime("%Y-%m-%d"),
                    "lifetime_value": round(self.random.lognormvariate(6, 0.9), 2),
                    "is_active": self.random.random() > 0.25,
                    "segment": self.random.choice(["bronze", "silver", "gold", "platinum"]),
                }
            )
        frame = pd.DataFrame(records)
        return self._apply_issues(frame, issues, text_columns=["first_name", "last_name", "city"])

    def products(self, rows: int = 200, issues: QualityIssues | None = None) -> pd.DataFrame:
        records = []
        for index in range(1, rows + 1):
            category = self.random.choice(CATEGORIES)
            cost = round(self.random.uniform(3, 400), 2)
            records.append(
                {
                    "product_id": index,
                    "sku": f"{category[:3].upper()}-{index:05d}",
                    "name": f"{category.title()} item {index}",
                    "category": category,
                    "unit_cost": cost,
                    "unit_price": round(cost * self.random.uniform(1.15, 2.4), 2),
                    "stock": self.random.randint(0, 900),
                    "created_at": self._date(1400).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        frame = pd.DataFrame(records)
        return self._apply_issues(frame, issues, text_columns=["name", "category"])

    def orders(
        self,
        rows: int = 5_000,
        *,
        customer_ids: list[int] | None = None,
        product_ids: list[int] | None = None,
        issues: QualityIssues | None = None,
    ) -> pd.DataFrame:
        customer_ids = customer_ids or list(range(1, 1_001))
        product_ids = product_ids or list(range(1, 201))
        records = []
        for index in range(1, rows + 1):
            quantity = self.random.randint(1, 8)
            price = round(self.random.uniform(5, 500), 2)
            records.append(
                {
                    "order_id": index,
                    "customer_id": self.random.choice(customer_ids),
                    "product_id": self.random.choice(product_ids),
                    "quantity": quantity,
                    "unit_price": price,
                    "total_amount": round(quantity * price, 2),
                    "discount_pct": self.random.choice([0, 0, 0, 5, 10, 15, 25]),
                    "status": self.random.choice(STATUSES),
                    "channel": self.random.choice(CHANNELS),
                    "order_date": self._date(365).strftime("%Y-%m-%d"),
                }
            )
        frame = pd.DataFrame(records)
        return self._apply_issues(frame, issues, text_columns=["status", "channel"])

    def transactions(self, rows: int = 5_000, issues: QualityIssues | None = None) -> pd.DataFrame:
        records = []
        for _ in range(1, rows + 1):
            amount = round(self.random.lognormvariate(4, 1.1), 2)
            records.append(
                {
                    "transaction_id": "".join(
                        self.random.choices(string.hexdigits.lower(), k=16)
                    ),
                    "order_id": self.random.randint(1, max(rows // 2, 2)),
                    "amount": amount,
                    "currency": self.random.choice(["USD", "EUR", "IRR", "AED"]),
                    "method": self.random.choice(["card", "wire", "wallet", "cash"]),
                    "succeeded": self.random.random() > 0.08,
                    "created_at": self._date(200).isoformat(sep=" ", timespec="seconds"),
                }
            )
        frame = pd.DataFrame(records)
        return self._apply_issues(frame, issues, text_columns=["currency", "method"])

    def employees(self, rows: int = 300, issues: QualityIssues | None = None) -> pd.DataFrame:
        records = []
        for index in range(1, rows + 1):
            first, last = self._name()
            records.append(
                {
                    "employee_id": index,
                    "full_name": f"{first} {last}",
                    "email": self._email(first, last, index),
                    "department": self.random.choice(DEPARTMENTS),
                    "salary": round(self.random.uniform(1_200, 9_500), 2),
                    "hire_date": self._date(3_000).strftime("%Y-%m-%d"),
                    "manager_id": self.random.choice([None, *range(1, max(index, 2))]),
                    "is_remote": self.random.random() > 0.6,
                }
            )
        frame = pd.DataFrame(records)
        return self._apply_issues(frame, issues, text_columns=["full_name", "department"])

    def country_lookup(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"country_code": code, "country_name": name, "region": region}
             for code, name, region in COUNTRIES]
        )

    # -- defect injection ---------------------------------------------------

    def _sample_index(self, frame: pd.DataFrame, rate: float) -> np.ndarray:
        count = int(len(frame) * rate)
        if count <= 0:
            return np.array([], dtype=int)
        return self.numpy.choice(len(frame), size=min(count, len(frame)), replace=False)

    def _apply_issues(
        self,
        frame: pd.DataFrame,
        issues: QualityIssues | None,
        *,
        text_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if issues is None:
            return frame
        result = frame.copy()
        text_columns = [column for column in (text_columns or []) if column in result.columns]

        if issues.missing > 0:
            candidates = [c for c in result.columns if not c.endswith("_id")]
            for column in candidates:
                positions = self._sample_index(result, issues.missing / max(len(candidates) / 3, 1))
                if len(positions):
                    # bool and int columns cannot hold NaN, so widen them first.
                    if not pd.api.types.is_float_dtype(result[column]):
                        result[column] = result[column].astype(object)
                    result.loc[result.index[positions], column] = None

        if issues.invalid_emails > 0 and "email" in result.columns:
            positions = self._sample_index(result, issues.invalid_emails)
            broken = ["not-an-email", "missing@", "@nodomain.com", "double@@mail.com", " "]
            for offset, position in enumerate(positions):
                result.iloc[position, result.columns.get_loc("email")] = broken[
                    offset % len(broken)
                ]

        if issues.invalid_dates > 0:
            date_columns = [c for c in result.columns if "date" in c or c.endswith("_at")]
            for column in date_columns:
                positions = self._sample_index(result, issues.invalid_dates)
                broken = ["31/02/2023", "not a date", "0000-00-00", "2023-13-45"]
                for offset, position in enumerate(positions):
                    result.iloc[position, result.columns.get_loc(column)] = broken[
                        offset % len(broken)
                    ]

        if issues.negative_values > 0:
            numeric = [
                c
                for c in result.columns
                if pd.api.types.is_numeric_dtype(result[c]) and not c.endswith("_id")
            ]
            for column in numeric:
                positions = self._sample_index(result, issues.negative_values)
                if len(positions):
                    index = result.index[positions]
                    result.loc[index, column] = -result.loc[index, column].abs()

        if issues.outliers > 0:
            numeric = [
                c
                for c in result.columns
                if pd.api.types.is_numeric_dtype(result[c]) and not c.endswith("_id")
            ]
            for column in numeric:
                positions = self._sample_index(result, issues.outliers)
                if len(positions):
                    index = result.index[positions]
                    result[column] = result[column].astype(float)
                    result.loc[index, column] = result[column].abs().max() * self.numpy.uniform(
                        20, 100, size=len(index)
                    )

        if issues.inconsistent_formatting > 0 and text_columns:
            for column in text_columns:
                positions = self._sample_index(result, issues.inconsistent_formatting)
                styles = [str.upper, str.lower, lambda v: f"  {v} ", lambda v: v.replace(" ", "  ")]
                for offset, position in enumerate(positions):
                    value = result.iloc[position, result.columns.get_loc(column)]
                    if isinstance(value, str):
                        result.iloc[position, result.columns.get_loc(column)] = styles[
                            offset % len(styles)
                        ](value)

        if issues.mixed_types > 0:
            numeric = [
                c
                for c in result.columns
                if pd.api.types.is_numeric_dtype(result[c]) and not c.endswith("_id")
            ]
            for column in numeric[:1]:
                positions = self._sample_index(result, issues.mixed_types)
                result[column] = result[column].astype(object)
                for offset, position in enumerate(positions):
                    result.iloc[position, result.columns.get_loc(column)] = (
                        "N/A" if offset % 2 else "unknown"
                    )

        if issues.duplicates > 0:
            positions = self._sample_index(result, issues.duplicates)
            if len(positions):
                copies = result.iloc[positions].copy()
                result = pd.concat([result, copies], ignore_index=True)
                result = result.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        return result

    def generate(
        self, kind: str, rows: int = 1_000, issues: QualityIssues | None = None, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate one of the built-in datasets by name."""
        builders = {
            "customers": self.customers,
            "products": self.products,
            "orders": self.orders,
            "transactions": self.transactions,
            "employees": self.employees,
        }
        if kind not in builders:
            raise ValueError(f"Unknown dataset '{kind}'; available: {sorted(builders)}")
        return builders[kind](rows, issues=issues, **kwargs)  # type: ignore[operator]


def generate_dataset(
    kind: str = "customers",
    rows: int = 1_000,
    *,
    seed: int = 42,
    issues: QualityIssues | None = None,
) -> pd.DataFrame:
    return SyntheticDataGenerator(seed).generate(kind, rows, issues)
