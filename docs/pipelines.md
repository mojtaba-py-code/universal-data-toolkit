# Pipeline configuration

A pipeline is a YAML file with four sections: `pipeline`, `input`, `steps` and
`output` (plus an optional `report`). Run it with:

```bash
data-tool run configs/customer_pipeline.yaml
```

Unknown top-level keys and unknown step names are errors, so a typo fails at
build time rather than silently doing nothing.

## Skeleton

```yaml
pipeline:
  name: customer_processing
  on_error: fail          # fail (default) | skip
  workspace: .            # optional; every path is confined to it

input:
  type: csv
  path: ../sample_data/customers_raw.csv
  options: {}             # passed to the reader

steps:
  - clean: {...}
  - validate: {...}
  - rules: {...}
  - transform: [...]
  - enrich: {...}
  - outliers: {...}
  - mask: {...}
  - quality: {...}

output:
  type: parquet
  path: ../output/customers_clean.parquet

report:
  path: ../output/customers_report.html
  format: html            # html | json
```

Paths are resolved against the directory of the YAML file and confined to the
workspace (the current directory by default, `--workspace` to override), so a
configuration cannot read outside the project it ships with.

## Inputs

### Files

```yaml
input:
  type: csv               # csv, tsv, json, jsonl, xml, yaml, excel, parquet, feather, sqlite
  path: data/customers.csv
  options:
    encoding: utf-8       # detected automatically when omitted
    sep: ";"
```

Omit `type` to detect the format from the file signature.

### Databases

```yaml
input:
  type: database
  table: customers        # or: query: "SELECT ... WHERE created_at > :since"
  options:
    params:
      since: "2024-01-01"
  connection:
    driver: postgresql    # postgresql | mysql | sqlite
    host: ${PGHOST:-localhost}
    port: ${PGPORT:-5432}
    database: ${PGDATABASE}
    username: ${PGUSER}
    password_env: PG_PASSWORD
```

`query` accepts a single `SELECT`/`WITH` statement and bound parameters only.

### APIs

```yaml
input:
  type: api
  url: https://api.example.com/customers
  method: GET
  headers:
    Accept: application/json
  params:
    updated_since: "2024-01-01"
  records_path: data.items
  max_records: 50000
  rate_limit: 5           # requests per second
  auth:
    type: bearer          # none | api_key | bearer
    env: API_TOKEN
  pagination:
    page_param: page
    page_size: 100
    max_pages: 50
  allowed_hosts: [api.example.com]
```

## Steps

### clean

Every key of `CleaningConfig` is accepted:

```yaml
- clean:
    normalize_columns: true
    trim_whitespace: true
    normalize_spaces: true
    case: lower                 # lower | upper | title
    remove_special_characters: false
    string_columns: [name, city]
    convert_types:
      age: int
      lifetime_value: float
    date_columns: [signup_date]
    date_format: "%Y-%m-%d"     # omitted means mixed-format parsing
    timezone: UTC
    numeric_columns: [salary]
    numeric_min: 0
    missing_values: median      # none | drop_rows | drop_columns | constant | mean | median | mode | ffill | bfill
    missing_value: 0            # for the constant strategy
    missing_threshold: 0.5      # for drop_columns
    remove_duplicates: true
    duplicate_subset: [customer_id]
    duplicate_keep: first       # first | last | none
```

### validate

```yaml
- validate:
    schema: ../schemas/customer.yaml
    mode: report                # report | filter | strict
```

`report` records the result and continues, `filter` keeps only the valid rows
(the rejected ones are available as `result.context.artifacts["invalid_rows"]`),
`strict` fails the run.

### rules

```yaml
- rules:
    file: customer_rules.yaml   # or an inline list under `rules:`
    action: flag                # flag | drop
```

`flag` adds a `_rule_failures` column naming the rules a row violated; `drop`
removes rows that fail an error-level rule. Warning-level rules never remove
rows.

### transform

A list of single-key mappings, applied in order:

```yaml
- transform:
    - rename:
        customer_name: name
    - drop_columns: [internal_notes]
    - select: [id, name, amount]
    - filter: "amount > 0 AND status != 'cancelled'"
    - create_column:
        column: net_amount
        expression: "quantity * unit_price * (1 - discount_pct / 100)"
    - map_values:
        column: country_code
        mapping: {IR: Iran, DE: Germany}
        default: Unknown
    - convert_types:
        amount: float
    - replace:
        column: phone
        pattern: "[^0-9+]"
        replacement: ""
    - aggregate:
        group_by: [region]
        aggregations:
          net_amount: [sum, mean]
          order_id: count
    - pivot:
        index: [region]
        columns: month
        values: revenue
        aggfunc: sum
    - melt:
        id_vars: [region]
    - sort:
        by: [revenue]
        ascending: false
    - limit: 1000
```

Aggregation functions are limited to
`sum, mean, median, min, max, count, size, nunique, std, var, first, last, prod`.

### enrich

```yaml
- enrich:
    lookup:
      path: ../sample_data/country_lookup.csv
      on: country_code
      columns: [region]
- enrich:
    derive:
      full_name: "first_name + ' ' + last_name"
      is_adult: "age >= 18"
```

A lookup table with duplicate keys is rejected, because the join would multiply
rows silently.

### outliers

```yaml
- outliers:
    columns: [lifetime_value]
    method: iqr           # iqr | zscore | stddev | percentile
    threshold: 1.5
    action: cap           # report | remove | cap | replace
```

`report` is the default: outliers are never removed unless asked for.

### mask / mask_pii

```yaml
- mask:
    email: email
    phone:
      strategy: phone
      keep_last: 4
    national_id:
      strategy: hash
      length: 16

- mask_pii:
    strategy: hash        # masks every column schema detection flags as personal
```

Strategies: `email`, `phone`, `name`, `partial`, `hash`, `token`, `redact`.
Set `UD_MASKING_SALT` so hashes stay stable between runs.

### quality

```yaml
- quality:
    schema: ../schemas/customer.yaml
    rules: {file: customer_rules.yaml}
    minimum_score: 0.9    # fails the run below this
```

## Outputs

```yaml
output:
  type: parquet
  path: ../output/customers.parquet
  options:
    compression: zstd
```

```yaml
output:
  type: postgresql
  table: customers
  if_exists: replace      # fail | replace | append
  connection:
    driver: postgresql
    host: ${PGHOST}
    database: ${PGDATABASE}
    username: ${PGUSER}
    password_env: PG_PASSWORD
```

## Schema files

```yaml
name: customer
strict: false             # true rejects columns that are not declared
primary_key: [customer_id]

schema:
  customer_id:
    type: integer
    nullable: false
    unique: true
  email:
    type: email
    required: true
    pii: true
  age:
    type: integer
    min: 18
    max: 100
  segment:
    type: categorical
    allowed: [bronze, silver, gold, platinum]
  code:
    type: string
    pattern: "[A-Z]{3}-[0-9]{4}"
    min_length: 8
    max_length: 8
```

Types: `integer`, `float`, `boolean`, `string`, `categorical`, `datetime`,
`date`, `email`, `url`, `phone`, `uuid`, `ip`.

## Rule files

```yaml
rules:
  - name: adult_customer
    condition: "age >= 18"
    description: The service is not offered to minors.
  - name: valid_email
    condition: "is_email(email)"
  - name: known_segment
    condition: "segment in ['bronze', 'silver', 'gold', 'platinum']"
    severity: warning         # error (default) removes rows under action: drop
```

### Expression language

A restricted subset of Python, evaluated vectorised over the frame:

* comparisons `== != < <= > >=`, `in`, `not in`, chained comparisons
* boolean `and`/`or`/`not`, also accepted as `AND`/`OR`/`NOT`
* SQL-style `IS NULL` / `IS NOT NULL`, and `<>` for `!=`
* arithmetic `+ - * / // % **`
* conditional expressions: `'senior' if age >= 30 else 'junior'`
* backtick-quoted column names: `` `first name` == 'Ali' ``

Functions: `is_null`, `not_null`, `len`, `lower`, `upper`, `strip`, `abs`,
`round`, `matches`, `contains`, `startswith`, `endswith`, `between`, `coalesce`,
`is_email`, `is_url`, `to_number`, `to_datetime`, `year`, `month`, `day`,
`days_since`, `least`, `greatest`.

Anything else - attribute access, subscripting, comprehensions, lambdas,
imports, `eval` - is rejected when the expression is parsed.
