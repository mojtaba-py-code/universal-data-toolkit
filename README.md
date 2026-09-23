# Universal Data Processing Toolkit

[![CI](https://github.com/mojtaba-py-code/universal-data-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/mojtaba-py-code/universal-data-toolkit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](pyproject.toml)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)](#testing)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A reusable toolkit for the part of data work that gets rewritten in every
project: reading a file whose format nobody documented, finding out what is
wrong with it, cleaning it, proving it satisfies a contract, and loading it
somewhere else - with logs, metrics and a report to show for it.

It is both a Python library and a command line application.

```bash
data-tool inspect customers.csv
data-tool process customers.csv -o customers.parquet --schema schemas/customer.yaml --mask-pii
data-tool run configs/customer_pipeline.yaml
```

```python
from universal_data import Dataset

dataset = Dataset.from_csv("customers.csv")
clean = (
    dataset
    .clean(missing_values="median", remove_duplicates=True)
    .filter("age >= 18")
    .mask_detected_pii()
)
clean.to_parquet("customers.parquet")
```

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CLI usage](#cli-usage)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Supported formats](#supported-formats)
- [Pipeline examples](#pipeline-examples)
- [Data quality](#data-quality)
- [Security](#security)
- [Performance](#performance)
- [Testing](#testing)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## Features

**Ingestion**
- 11 input formats with detection from the file signature, not the extension
- Encoding and delimiter sniffing (UTF-8/BOM/CP1252, `,` `;` `\t` `|`)
- SQLite, PostgreSQL and MySQL through SQLAlchemy with pooling and transactions
- REST ingestion with retries, exponential backoff with jitter, rate limiting,
  four pagination styles and four authentication strategies
- Chunked reading for files larger than RAM

**Understanding the data**
- Dataset and column profiling: shape, memory, nulls, cardinality, duplicates,
  numeric/categorical/temporal statistics, outliers
- Schema inference including logical types (`email`, `uuid`, `phone`) and likely
  PII columns
- Schema validation with required/nullable/unique, ranges, regex, enums, length
  and composite primary keys - returning a row mask so bad rows can be
  quarantined instead of losing the batch

**Changing the data**
- Cleaning: missing values (8 strategies), duplicates, text normalisation, type
  conversion, date parsing, numeric coercion
- 14 composable transformations, chainable and configurable from YAML
- Business rules in a safe expression language, evaluated vectorised
- Outlier detection (IQR, z-score, standard deviation, percentile) with
  report/remove/cap/replace
- Enrichment from lookup tables, mappings, derived columns or external services
  with per-key caching
- PII masking: email, phone, name, partial, salted hash, tokenisation, redaction

**Operating it**
- Data quality scoring across six dimensions with recommendations
- Self-contained HTML and JSON reports
- Structured JSON logging with rotation, execution ids and secret redaction
- Per-step metrics with a Prometheus text exporter
- Pipelines in Python or YAML, with conditional steps and fail/skip modes
- Plugin system through entry points

---

## Architecture

```mermaid
flowchart TD
    subgraph Sources
        F[Files] --- D[Databases] --- A[HTTP APIs]
    end
    Sources --> DET[Format detection]
    DET --> ING[Ingestion]
    ING --> SCH[Schema detection]
    SCH --> VAL[Validation]
    VAL --> CLEAN[Cleaning]
    CLEAN --> TRANS[Transformation]
    TRANS --> RULES[Business rules]
    RULES --> ENR[Enrichment]
    ENR --> MASK[Masking]
    MASK --> QUAL[Quality analysis]
    QUAL --> EXP[Export]
    EXP --> OUT[Files / Databases]
    QUAL --> REP[HTML and JSON reports]
    EXP --> MET[Metrics and logs]
```

Three decisions shape the codebase:

1. **`Dataset` is immutable.** Every operation returns a new one carrying its
   history, so a failed step never leaves a half-modified frame behind and the
   report can state truthfully what produced the output.
2. **Extension happens through registries.** Readers, writers and
   transformations are looked up by name, so a new format is a registration
   rather than an edit to a dispatch function - the same mechanism plugins use.
3. **Expressions are parsed, never executed.** Rules and filters come from YAML,
   so `eval` and `DataFrame.query` are both out of the question. See
   [Security](#security).

Details, including the design patterns used and why, are in
[docs/architecture.md](docs/architecture.md).

---

## Installation

Requires Python 3.12 or newer.

```bash
git clone https://github.com/mojtaba-py-code/universal-data-toolkit.git
cd universal-data-toolkit

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .                   # library plus the data-tool command
pip install -e ".[dev]"            # plus pytest, ruff, mypy, bandit
pip install -e ".[postgres,mysql]" # optional database drivers
```

Copy the environment template and fill in what you need:

```bash
cp .env.example .env
```

Nothing in `.env` is required to use the file-based features.

---

## Quick start

```bash
# 1. Generate a deliberately messy dataset
data-tool generate --kind customers --rows 5000 -o data/customers.csv --issues realistic

# 2. See what is in it
data-tool inspect data/customers.csv

# 3. Derive a schema from it, then edit the file by hand
data-tool schema data/customers.csv -o schemas/customers.yaml --constraints

# 4. Check the data against the schema
data-tool validate data/customers.csv --schema schemas/customers.yaml

# 5. Clean, validate, mask and export in one command
data-tool process data/customers.csv -o output/customers.parquet \
    --schema schemas/customers.yaml --mask-pii --report output/report.html
```

The repository ships with sample data, so this also works straight away:

```bash
python examples/08_end_to_end_demo.py
```

---

## CLI usage

```text
data-tool inspect      Detect the format and summarise a file
data-tool profile      Full profile with per-column statistics
data-tool schema       Infer a schema and flag likely PII
data-tool validate     Check a dataset against a schema
data-tool quality      Score the six quality dimensions
data-tool clean        Clean a dataset and write the result
data-tool transform    Rename, select, filter, sort, limit
data-tool convert      Convert between formats, optionally streamed
data-tool export       Export to another format or into SQLite
data-tool process      One-command clean/validate/mask/export pipeline
data-tool run          Run a YAML-defined pipeline
data-tool fetch        Pull JSON records from an HTTP API
data-tool generate     Generate synthetic data with controlled defects
data-tool benchmark    Compare row-wise, vectorised, in-memory and chunked
data-tool formats      List supported input and output formats
data-tool plugins      List installed plugins
data-tool version      Print the version
```

```bash
$ data-tool inspect sample_data/customers_raw.csv
       File
 property        value
 format          csv
 size            89,612 bytes
 encoding        utf-8
 delimiter       ','

Dataset Profile: customers_raw

Shape:          824 rows x 13 columns
Memory:         0.16 MB
Missing cells:  165 (1.5%)
Duplicate rows: 24 (2.9%)
Possible PII:   first_name, last_name, email, phone

Columns:
  customer_id     integer     missing=  0.0%  unique=800
  email           email       missing=  2.2%  unique=761
  age             integer     missing=  1.5%  unique=61
  ...
```

Exit codes: `0` success, `1` error, `2` the data failed a gate. Full reference in
[docs/cli.md](docs/cli.md).

---

## Python API

### Reading

```python
from universal_data import Dataset

Dataset.read("data.csv")                        # format detected from the signature
Dataset.from_excel("book.xlsx", sheet_name="Q1")
Dataset.from_parquet("data.parquet")
Dataset.from_sqlite("app.db", table="customers")
Dataset.from_records([{"id": 1}, {"id": 2}])

for chunk in Dataset.read_chunks("huge.csv", chunk_size=100_000):
    ...
```

```python
from universal_data import DatabaseClient, DatabaseConfig

config = DatabaseConfig.from_env("PG")          # PG_DRIVER, PG_HOST, PG_PASSWORD ...
with DatabaseClient(config) as client:
    dataset = Dataset.from_database(
        client,
        query="SELECT * FROM orders WHERE created_at > :since",
        params={"since": "2024-01-01"},
    )
```

```python
from universal_data.ingestion.api import APIClient, APIConfig, BearerTokenAuth, PageNumberPaginator
from universal_data.security.secrets import SecretResolver

token = SecretResolver().get("API_TOKEN", required=True)
with APIClient(
    APIConfig(url="https://api.example.com/users", records_path="data.items"),
    auth=BearerTokenAuth(token),
    paginator=PageNumberPaginator(page_size=100, max_pages=50),
    rate_limit=5,
) as client:
    users = Dataset.from_api(client)
```

### Inspecting and validating

```python
profile = dataset.profile()
print(profile.summary())

schema = dataset.detect_schema(infer_constraints=True)
print(schema.pii_columns)                       # ['email', 'phone', 'first_name']

from universal_data.schema.loader import load_schema
result = dataset.validate(load_schema("schemas/customer.yaml"))
print(result.summary())

valid, rejected = dataset.keep_valid(schema)    # quarantine instead of failing
```

### Cleaning and transforming

```python
cleaned = dataset.clean(
    normalize_columns=True,
    trim_whitespace=True,
    case="lower",
    convert_types={"age": "int", "amount": "float"},
    date_columns=["signup_date"],
    missing_values="median",
    remove_duplicates=True,
)
print(cleaned.last_cleaning_report().summary())

from universal_data.transformation import Aggregate, CreateColumn, FilterRows

report = cleaned.transform(
    FilterRows("status != 'cancelled' AND quantity > 0"),
    CreateColumn("net_amount", "quantity * unit_price * (1 - discount_pct / 100)"),
    Aggregate(group_by=["channel"], aggregations={"net_amount": ["sum", "mean"]}),
)
```

### Rules, masking, enrichment

```python
from universal_data.rules import BusinessRule, RuleEngine

rules = RuleEngine([
    BusinessRule(name="adult", condition="age >= 18"),
    BusinessRule(name="valid_email", condition="is_email(email)"),
    BusinessRule(name="has_contact", condition="email IS NOT NULL OR phone IS NOT NULL"),
])
flagged, evaluation = cleaned.apply_rules(rules, action="flag")
print(evaluation.rows_failed)

from universal_data.masking import MaskingPolicy

masked = flagged.mask(MaskingPolicy.from_dict({
    "email": "email",                                   # j***@example.com
    "phone": {"strategy": "phone", "keep_last": 4},     # *******4567
    "national_id": {"strategy": "hash"},                # salted SHA-256
}))

from universal_data.enrichment import LookupEnricher

enriched = masked.enrich(
    LookupEnricher.from_file("lookups/countries.csv", on="country_code", columns=["region"])
)
```

### Quality and export

```python
quality = cleaned.quality(schema, rules)
print(quality.overall_score, quality.grade)     # 0.972 A

document = cleaned.report(schema=schema, rules=rules)
document.to_html("report.html")
document.to_json("report.json")

cleaned.to_parquet("out.parquet")
cleaned.to_csv("out.csv")
cleaned.to_sqlite("out.db", table="customers")
with DatabaseClient(config) as client:
    cleaned.to_database(client, "customers", if_exists="replace")
```

### Pipelines

```python
from universal_data import Pipeline

result = (
    Pipeline("customers", on_error="fail")
    .read("customers.csv")
    .clean(missing_values="median", remove_duplicates=True)
    .validate(schema, mode="filter")
    .apply_rules(rules, action="flag")
    .mask_detected_pii()
    .quality_check(minimum_score=0.9)
    .write("customers.parquet")
    .report("report.html")
    .run()
)

print(result.metrics.summary())
print(result.quality.summary())
```

Files larger than memory:

```python
Pipeline("orders").clean(missing_values="median").run_chunked(
    "orders_20gb.csv", "orders.parquet", chunk_size=250_000
)
```

---

## Configuration

```yaml
pipeline:
  name: customer_processing
  on_error: fail

input:
  type: csv
  path: ../sample_data/customers_raw.csv

steps:
  - clean:
      normalize_columns: true
      convert_types: {age: int, lifetime_value: float}
      date_columns: [signup_date]
      missing_values: median
      remove_duplicates: true

  - validate:
      schema: ../schemas/customer.yaml
      mode: report

  - rules:
      file: customer_rules.yaml
      action: flag

  - enrich:
      lookup:
        path: ../sample_data/country_lookup.csv
        on: country_code
        columns: [region]

  - outliers:
      columns: [lifetime_value]
      method: iqr
      action: cap

  - mask:
      email: email
      phone: {strategy: phone, keep_last: 4}

  - quality:
      minimum_score: 0.5

output:
  type: parquet
  path: ../output/customers_clean.parquet

report:
  path: ../output/customers_report.html
```

```bash
data-tool run configs/customer_pipeline.yaml
```

The full reference - inputs, every step, schema files, rule files and the
expression language - is in [docs/pipelines.md](docs/pipelines.md).

---

## Supported formats

| Format | Read | Write | Streaming | Notes |
| --- | :---: | :---: | :---: | --- |
| CSV | yes | yes | yes | Encoding and delimiter detected |
| TSV | yes | yes | yes | |
| Excel (`.xlsx`, `.xlsm`, `.xls`) | yes | yes | - | Multi-sheet read; 1,048,575-row write limit enforced |
| JSON | yes | yes | - | Nested payloads flattened; `record_path` supported |
| JSON Lines | yes | yes | yes | |
| XML | yes | - | - | stdlib parser, no external entity resolution |
| YAML | yes | - | - | `safe_load` only |
| Parquet | yes | yes | yes | pyarrow; row-group streaming |
| Feather / Arrow IPC | yes | yes | - | |
| Pickle | opt-in | - | - | Refused unless explicitly allowed |
| SQLite | yes | yes | yes | Table names validated and quoted |
| PostgreSQL | yes | yes | yes | via SQLAlchemy, `[postgres]` extra |
| MySQL / MariaDB | yes | yes | yes | via SQLAlchemy, `[mysql]` extra |
| REST / JSON APIs | yes | - | paged | Auth, retries, rate limiting, SSRF guard |

---

## Pipeline examples

Every example is runnable from a clean checkout:

| Example | What it shows |
| --- | --- |
| [01_csv_clean_csv.py](examples/01_csv_clean_csv.py) | CSV in, cleaned CSV out |
| [02_csv_transform_parquet.py](examples/02_csv_transform_parquet.py) | Derived columns, aggregation, Parquet |
| [03_api_validate_database.py](examples/03_api_validate_database.py) | API (or JSON fallback), validation, database load |
| [04_excel_clean_database.py](examples/04_excel_clean_database.py) | Excel, masking, SQL query on the result |
| [05_large_csv_chunked.py](examples/05_large_csv_chunked.py) | A file larger than the chunk size, streamed |
| [06_join_and_analyse.py](examples/06_join_and_analyse.py) | Three datasets joined, enriched, aggregated |
| [07_config_driven_pipeline.py](examples/07_config_driven_pipeline.py) | Running a YAML pipeline from Python |
| [08_end_to_end_demo.py](examples/08_end_to_end_demo.py) | The full twelve-step demonstration |

```bash
python examples/08_end_to_end_demo.py
```

---

## Data quality

Six dimensions, each a ratio in `[0, 1]` that keeps the counts it was computed
from. Dimensions that cannot be measured (timeliness without a date column) are
omitted and the remaining weights are renormalised rather than scored as zero.

```mermaid
graph LR
    Q[Quality score] --> C[Completeness 25%]
    Q --> V[Validity 25%]
    Q --> U[Uniqueness 15%]
    Q --> K[Consistency 15%]
    Q --> A[Accuracy 10%]
    Q --> T[Timeliness 10%]
```

| Dimension | Measures |
| --- | --- |
| Completeness | Non-null cells over total cells |
| Validity | Rows satisfying the schema, or values agreeing with their inferred type |
| Uniqueness | Absence of duplicate rows and duplicate primary keys |
| Consistency | Share of values matching the dominant format per text column, plus the business-rule pass rate |
| Accuracy | Proxy: numeric values inside plausible bounds (IQR) |
| Timeliness | Recency of the newest record, penalised for future-dated rows |

```text
Dataset Quality Report: customers

Completeness  100.0%
Validity       93.0%
Uniqueness    100.0%
Consistency    93.0%
Accuracy      100.0%
Timeliness    100.0%

Overall Score: 97.2% (grade A)

Recommendations:
  - Review type conversions; some values do not match their column type
```

Reports are produced as self-contained HTML (no external assets, every value
escaped) and as JSON for machine consumption. They contain the dataset
overview, the schema, per-column statistics, validation issues with examples,
business-rule results, the transformation summary, processing statistics and the
recommendations.

---

## Security

Security is handled at the boundaries where untrusted input enters, not
sprinkled through the business logic.

| Risk | Control |
| --- | --- |
| Code execution through an expression | `ast`-based allow-list parser; no `eval`, no `DataFrame.query` |
| Code execution through a data file | Pickle refused unless explicitly allowed |
| Path traversal | `PathPolicy`: workspace confinement, symlink and extension checks |
| SQL injection | Bound parameters everywhere; identifiers validated and quoted |
| Destructive SQL from a config file | `read_query` requires one statement, a `SELECT`/`WITH` start **and** no writing keyword - a `WITH ... DELETE` is valid SQL and is blocked |
| SSRF | `URLPolicy` blocks non-HTTPS, private/loopback/link-local addresses and metadata hosts; every redirect hop re-validated |
| Credential leakage | `SecretStr`, `RedactingFilter` on every handler, masked connection strings |
| Secrets in version control | `password_env:` in config, `.env` git-ignored, `.env.example` committed |
| XSS in reports | Every value passed through `html.escape` |
| XXE | stdlib XML parser |
| Weak pseudonymisation | Salted hashing; `md5`/`sha1` refused |

```python
>>> SafeExpression("__import__('os').system('rm -rf /')")
RuleError: Expression uses a construct that is not allowed: Call

>>> PathPolicy.confined_to("/srv/data").resolve_input("../../etc/passwd")
SecurityError: Path escapes the allowed workspace

>>> validate_url("https://169.254.169.254/latest/meta-data/")
SecurityError: Access to cloud metadata endpoints is blocked

>>> print(DatabaseConfig(..., password=SecretStr("hunter2")).safe_url())
postgresql+psycopg://app:***@db.internal:5432/app
```

The threat model and every control is documented in
[docs/security.md](docs/security.md). Security behaviour is covered by tests -
path traversal, SQL injection, sandbox escapes, SSRF, secret redaction and
report escaping all have cases in the suite.

---

## Performance

100,000 rows, Python 3.12, 2-core i5-4210U:

| Scenario | Seconds | Rows/sec | Extra memory |
| --- | ---: | ---: | ---: |
| Row-wise `DataFrame.apply` | 1.871 | 53,980 | 6.7 MB |
| Vectorised expression engine | 0.008 | 11,992,543 | 2.3 MB |
| In-memory pipeline | 1.539 | 65,620 | 24.0 MB |
| Chunked pipeline (25k chunks) | 1.449 | 69,722 | ~0 MB |

The same calculation is roughly 200 times faster through the expression engine
than through `apply`, which is why derived columns and business rules are
expressions rather than callbacks. Chunking costs nothing in wall-clock time and
keeps memory flat.

```bash
data-tool benchmark --rows 200000
python benchmarks/run_benchmarks.py --rows 200000
```

`docs/performance.md` covers chunk-size selection, what chunked mode cannot do,
memory accounting, and why multiprocessing and async were deliberately left out.

---

## Testing

```bash
pytest -q                                        # 690 tests
pytest -q --cov=universal_data --cov-report=term-missing
pytest -m "not slow"                             # skip the benchmark comparison
ruff check src tests scripts examples benchmarks
mypy
bandit -r src -c pyproject.toml
```

The PostgreSQL tests are skipped unless a server is reachable, which covers a
second SQL dialect for real rather than only through SQLite:

```bash
docker compose up -d postgres
UD_TEST_POSTGRES=1 PG_PASSWORD=... pytest tests/test_postgres_integration.py
```

```text
690 passed in 19.44s
TOTAL   5794 statements   1518 branches   92% coverage
```

The suite covers unit behaviour for every module, integration through SQLite and
a stubbed HTTP session, all major CLI commands, security properties, and the
benchmark harness. Core modules sit above 90%: `Dataset` 97%, pipeline engine
98%, validation 97%, cleaning 94-97%, rules 97%, quality 94%.

---

## Docker

```bash
cp .env.example .env            # set PG_PASSWORD
docker compose up -d postgres
docker compose run --rm data-tool inspect /data/sample_data/customers_raw.csv
docker compose run --rm data-tool run /data/configs/postgres_pipeline.yaml
```

The image is a multi-stage build: a wheel is built in the first stage and
installed into a slim runtime carrying no build tools. It runs as an
unprivileged user and works on `/data`, which is where the project directory is
mounted - nothing is written into the image.

```bash
docker build -t universal-data-toolkit .
docker run --rm -v "$PWD:/data" universal-data-toolkit inspect /data/sample_data/customers_raw.csv
```

---

## CI/CD

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and
pull request:

| Job | What it does |
| --- | --- |
| `lint` | Ruff and mypy |
| `test` | pytest on Python 3.12 and 3.13, coverage floor at 90% |
| `security` | Bandit, dependency audit, and a check that no `.env` file is tracked |
| `demo` | Regenerates the sample data, runs both YAML pipelines and the end-to-end demo, uploads the HTML reports |
| `build` | Builds the wheel and validates it with `twine` |
| `docker` | Builds the image and smoke-tests the CLI inside it |

---

## Project structure

```text
universal-data-toolkit/
├── src/universal_data/
│   ├── core/            Dataset, exceptions, shared enums
│   ├── security/        path policy, secrets, URL policy, log redaction
│   ├── ingestion/       detection, file readers, database client, API client
│   ├── schema/          schema model, inference, YAML loading
│   ├── validation/      column checks and the schema validator
│   ├── cleaning/        cleaning operations and engine
│   ├── transformation/  transformation interface, built-ins, registry
│   ├── rules/           safe expression engine, business rule engine
│   ├── quality/         scoring, outliers, HTML/JSON reports
│   ├── enrichment/      lookups, mappings, derived columns, services
│   ├── masking/         PII masking strategies and policies
│   ├── export/          writer interface and built-in writers
│   ├── inspection/      dataset and column profiling
│   ├── pipeline/        pipeline engine and YAML loader
│   ├── observability/   structured logging, metrics, memory
│   ├── plugins/         entry-point discovery
│   ├── generator/       synthetic data with configurable defects
│   └── cli/             Typer application and benchmarks
├── tests/               690 tests
├── examples/            8 runnable examples
├── configs/             pipeline and rule definitions
├── schemas/             schema contracts
├── sample_data/         generated sample datasets
├── docs/                architecture, security, CLI, pipelines, performance, plugins
├── benchmarks/          benchmark runner and recorded results
├── scripts/             sample data generator
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Troubleshooting

**`UnsupportedFormatError: Could not determine the file format`**
The extension is unknown and the content does not match a known signature. Pass
the format explicitly: `Dataset.read(path, "csv")` or `--format csv`.

**`SecurityError: Path escapes the allowed workspace`**
A config-driven pipeline may only touch files inside its workspace. Move the
file in, or run with `--workspace` pointing at a directory that contains both.

**`SecurityError: URL resolves to a non-public address`**
The SSRF guard blocks private and loopback addresses. For a local test server,
add `--allow-private` (CLI) or `allow_private_networks: true` (config). Do not
use it against production endpoints.

**`RuleError: Expression uses a construct that is not allowed`**
The expression language is a deliberate subset. Attribute access, indexing,
comprehensions and lambdas are rejected. See the function list in
[docs/pipelines.md](docs/pipelines.md).

**`ExportError: A chunk has a different schema than the first chunk`**
Chunked CSV reading inferred different dtypes for different chunks. Pin them:
`clean(convert_types={"amount": "float"})` or pass `dtype=...` as a read option.

**`ConfigurationError: Configuration references an undefined environment variable`**
The config uses `${NAME}` or `password_env: NAME` and the variable is not set.
Add it to `.env` or export it.

**`SecurityError: Reading pickle files executes arbitrary code`**
Intentional. Pass `allow_pickle=True` only for files you produced yourself.

**Memory warnings on a large file**
Switch to chunked mode: `--chunk-size 100000` or `Pipeline.run_chunked(...)`.

**Hashed identifiers change between runs**
`UD_MASKING_SALT` is not set, so a random salt is generated per run. Set it to a
stable secret to keep hashes joinable.

---

## Roadmap

- Avro and ORC readers/writers
- Incremental/CDC ingestion with watermark tracking
- A FastAPI service exposing profiling and validation over HTTP
- Airflow and Prefect operators wrapping `Pipeline`
- Cloud object storage adapters (S3, GCS, Azure Blob)
- Great Expectations-compatible schema export
- Streaming ingestion from Kafka

These stay behind the existing interfaces: nothing in the core depends on an
external service.

---

## License

MIT - see [LICENSE](LICENSE).
