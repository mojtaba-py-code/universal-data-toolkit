# CLI reference

```bash
data-tool --help
```

Global options come before the command:

| Option | Default | Meaning |
| --- | --- | --- |
| `--log-level` | `WARNING` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--json-logs` | off | One JSON object per log line |
| `--log-file PATH` | none | Also write rotating JSON logs to a file |
| `--env-file PATH` | `.env` | Load environment variables if the file exists |
| `--quiet`, `-q` | off | No console logging |

Exit codes: `0` success, `1` an error occurred, `2` the data failed a gate
(validation with `--strict`, a quality minimum, a failed pipeline step).

## Inspection

```bash
data-tool inspect sample_data/customers_raw.csv          # format, encoding, profile, preview
data-tool inspect data.csv --json                        # machine readable
data-tool profile data.csv --html-out report.html        # full profile plus an HTML report
data-tool schema data.csv -o schemas/detected.yaml --constraints
data-tool formats                                        # supported input/output formats
```

`inspect` reports the detected format, encoding, delimiter, shape, memory,
missing values, duplicates, per-column types and the columns that look like
personal data.

## Validation and quality

```bash
data-tool validate data.csv --schema schemas/customer.yaml
data-tool validate data.csv --schema schemas/customer.yaml --strict          # exit 2 on failure
data-tool validate data.csv --schema schemas/customer.yaml \
    --invalid-out output/rejected.csv                                        # quarantine bad rows

data-tool quality data.csv --rules configs/customer_rules.yaml \
    --html-out output/quality.html --minimum 0.9
```

## Processing

```bash
data-tool clean data.csv -o clean.csv --missing median --drop-duplicates --normalize-columns

data-tool transform data.csv -o out.csv \
    --rename age=years --select customer_id,years \
    --where "years >= 18" --sort-by years --desc --limit 100

data-tool convert data.csv out.parquet                   # format conversion
data-tool convert huge.csv out.db --chunk-size 50000 --table orders

data-tool export data.csv -o out.jsonl

data-tool process raw.csv -o clean.parquet \
    --schema schemas/customer.yaml \
    --rules configs/customer_rules.yaml \
    --mask-pii --report output/report.html

data-tool process huge.csv -o clean.parquet --chunk-size 100000
```

`process` is the one-command pipeline: clean, optionally validate, apply rules,
mask, score and export. `--chunk-size` switches it to streaming mode, where
memory stays proportional to the chunk rather than to the file.

## Configuration-driven pipelines

```bash
data-tool run configs/customer_pipeline.yaml
data-tool run configs/customer_pipeline.yaml --dry-run          # validate the config only
data-tool run configs/customer_pipeline.yaml --workspace /srv/data --metrics-out metrics.json
```

See [pipelines.md](pipelines.md) for the file format.

## Sources and utilities

```bash
data-tool fetch --url https://api.example.com/users -o users.json \
    --token-env API_TOKEN --records-path data.items --pages 10

data-tool generate --kind customers --rows 100000 -o big.csv --issues realistic
data-tool benchmark --rows 200000
data-tool plugins
data-tool version
```

`fetch` refuses plain HTTP and private addresses by default; `--allow-http` and
`--allow-private` exist for local testing and should not be used against
production endpoints.

## Docker

```bash
docker compose run --rm data-tool inspect /data/sample_data/customers_raw.csv
docker compose run --rm data-tool run /data/configs/customer_pipeline.yaml
```

The project directory is mounted at `/data` and the container runs as an
unprivileged user.
