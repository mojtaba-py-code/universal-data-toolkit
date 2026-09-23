# Security

The toolkit executes instructions that come from configuration files and reads
data from files, databases and the network. That makes four things untrusted:
the configuration, the file paths, the data itself and the endpoints. Each is
handled at the boundary where it enters the system.

## Threat model

| Threat | Where it enters | Control |
| --- | --- | --- |
| Arbitrary code execution through an expression | `filter`, `create_column`, business rules | `SafeExpression` parses with `ast` and allows an explicit node list; no `eval`, no `DataFrame.query` |
| Arbitrary code execution through a data file | `.pkl` input | Refused unless `allow_pickle=True` or `UD_ALLOW_PICKLE=1` |
| Path traversal | YAML paths, CLI arguments | `PathPolicy` resolves, confines to a workspace and rejects symlinks |
| SQL injection | Table/column names, query text | Identifiers validated against `^[A-Za-z_][A-Za-z0-9_$]*$` and quoted by the dialect preparer; values always bound |
| Destructive SQL from a config file | `input.query` | `read_query` requires a single statement starting with `SELECT`/`WITH` **and** containing no writing keyword; string literals and comments are stripped before the check |
| Server-side request forgery | API URLs and redirects | `URLPolicy` blocks non-HTTP schemes, plain HTTP, private/loopback/link-local addresses and cloud metadata hosts; every redirect hop is re-validated |
| Credential leakage | Logs, reports, tracebacks, reprs | `SecretStr` never prints its value; `RedactingFilter` scrubs records; `DatabaseConfig.safe_url()` masks the password |
| Credentials in version control | Config files | Configuration carries `password_env: NAME`, never a password; `.env` is git-ignored and `.env.example` is committed instead |
| XSS in a generated report | Values from an untrusted CSV | Everything in the HTML report is passed through `html.escape` |
| XXE / entity expansion | XML input | `pandas.read_xml(parser="etree")`, the stdlib parser, which does not resolve external entities |
| Memory exhaustion | Oversized input | `PathPolicy(max_bytes=...)`, `APIConfig.max_response_bytes`, chunked processing and memory warnings |
| Reversible pseudonymisation | Hash masking | Salted hashing, SHA-256 or better; `md5`/`sha1` are refused |

## Secrets

Nothing is hard-coded. Configuration references the environment:

```yaml
connection:
  driver: postgresql
  host: ${PGHOST:-localhost}
  database: ${PGDATABASE:-datatoolkit}
  username: ${PGUSER:-datatoolkit}
  password_env: PG_PASSWORD
```

* `env:NAME` and `${NAME}` resolve to the value of an environment variable.
* `${NAME:-fallback}` supplies a default for non-sensitive settings only.
* `password_env` is the only accepted way to give a database password.
* A referenced variable that is not set raises `ConfigurationError` at build
  time, before a connection is attempted.

`SecretStr` wraps every secret so that printing, logging or serialising it
yields `***`. The raw value is only available through `get_secret_value()`.

```python
>>> secret = SecretStr("hunter2")
>>> print(secret), repr(secret)
*** SecretStr('***')
```

## Filesystem

```python
policy = PathPolicy.confined_to("/srv/pipelines/customer")
policy.resolve_input("../../etc/passwd")   # SecurityError
```

`PathPolicy` rejects NUL bytes, empty paths, symlinks, paths that resolve
outside the workspace, unexpected extensions and oversized files. Config-driven
pipelines always build a confined policy: relative paths are resolved against
the directory of the YAML file, and the workspace defaults to the current
working directory (`--workspace` overrides it).

Writers go through `atomic_path`: data is written to a temporary file next to
the destination and renamed on success, so a crash never replaces a good dataset
with a truncated one.

## Databases

```python
client.read_query("SELECT * FROM customers WHERE email = :email", params={"email": value})
```

* Values are always bound parameters, never interpolated.
* Identifiers cannot be bound in SQL, so they are validated against a strict
  pattern and quoted by SQLAlchemy's dialect preparer.
* Writes go through `execute` or `write_frame`, which run inside a transaction.
* Connection pooling uses `pool_pre_ping` so a recycled connection does not fail
  a whole batch.

### What `read_query` actually guarantees

Checking that a statement *starts with* `SELECT` or `WITH` is not enough:

```sql
WITH x AS (SELECT 1) DELETE FROM customers   -- valid SQL, and it deletes
```

CTE-prefixed DML is standard in SQLite, PostgreSQL and MySQL. `read_query`
therefore strips comments and string literals, then requires all three of:

1. the statement starts with `SELECT` or `WITH`,
2. there is exactly one statement,
3. no writing keyword appears anywhere (`insert`, `update`, `delete`, `drop`,
   `alter`, `create`, `truncate`, `into`, `attach`, `pragma`, `copy`, …).

Stripping the literals first means `WHERE name = 'a;b'` and
`WHERE action != 'delete'` are still accepted - the guard reads syntax, not
data.

This is a guard against a careless or hostile configuration file. It is **not a
sandbox**: it is a keyword filter, and a sufficiently exotic dialect feature
could still slip past it. The authoritative control is the database role the
toolkit connects with, which should have `SELECT`-only grants on the tables it
reads.

## HTTP

Every URL is validated before a socket is opened, including each redirect hop:

```python
validate_url("https://169.254.169.254/latest/meta-data/")   # SecurityError
validate_url("http://example.com/data")                     # SecurityError - https required
```

Redirects are followed manually (`allow_redirects=False`) so the policy applies
to the target, not only to the first URL. Responses are size-limited and must be
JSON. Authentication is a strategy object; tokens are `SecretStr` and are
attached to headers at request time, never stored in the config object.

Set `allow_private_networks=True` (CLI: `--allow-private`) only for a local test
server you control.

## Logging

`RedactingFilter` is attached to every handler the toolkit configures. It masks

* credentials embedded in URLs (`postgresql://user:***@host/db`),
* `Authorization: Bearer ***`,
* `password=`, `api_key=`, `token=` … in messages and query strings,
* any mapping key that looks sensitive, recursively.

Log files are JSON, one object per line, with rotation (10 MB, five backups by
default). Data values are never logged wholesale; validation samples are
truncated to 64 characters.

## Known residual risks

These are limitations of the approach, documented rather than hidden.

**DNS rebinding.** `URLPolicy` resolves the hostname and rejects private
addresses, then `requests` resolves it again when it opens the socket. A
hostile DNS server can answer with a public address for the check and a private
one for the connection. Mitigations, in order of strength: use `allowed_hosts`
to pin the endpoints a pipeline may reach, route outbound traffic through an
egress proxy, or run the toolkit in a network namespace with no route to the
internal network.

**Regular expressions in rules.** `matches()` compiles a pattern from the rule
file. The pattern length is capped, but Python's `re` has no execution timeout,
so a crafted pattern can in principle burn CPU. CPython 3.11+ handles the
classic `(a+)+$` case in linear time, and rule files are operator-supplied
rather than user-supplied, so this is accepted rather than mitigated. Treat a
rule file with the same care as code review.

**Time-of-check to time-of-use on paths.** `PathPolicy` validates a path and
the caller then opens it. A local attacker who can create a symlink in the
workspace between those two moments could redirect the write. This matters only
on a shared machine with a writable workspace; the fix there is filesystem
permissions, not application code.

**`method="multi"` bulk inserts.** Rows per statement are capped from the
driver's bind-parameter limit (SQLite reports its own; 65,535 is assumed for
PostgreSQL and MySQL). A driver with a lower limit would need an entry in
`PARAMETER_LIMITS`.

## Reporting an issue

Open a private security advisory on the repository rather than a public issue.
