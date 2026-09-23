# Performance and memory

## Measured results

Reproduce with:

```bash
python benchmarks/run_benchmarks.py --rows 100000 --chunk-size 25000
# or
data-tool benchmark --rows 200000
```

100,000 order rows, Python 3.12.5, Windows 10, Intel i5-4210U (2 cores), SSD:

| Scenario | Seconds | Rows/sec | Extra memory (MB) |
| --- | ---: | ---: | ---: |
| Row-wise `DataFrame.apply` | 1.871 | 53,980 | 6.7 |
| Vectorised expression engine | 0.008 | 11,992,543 | 2.3 |
| In-memory pipeline (read, clean, write) | 1.539 | 65,620 | 24.0 |
| Chunked pipeline (25,000-row chunks) | 1.449 | 69,722 | 0.0 |

Two things come out of this:

* **Vectorisation is worth roughly two orders of magnitude.** The same
  calculation - `quantity * unit_price * (1 - discount_pct / 100)` - takes 1.87 s
  through `apply` and 8 ms through `SafeExpression`, because the expression
  engine compiles down to pandas/NumPy operations instead of calling a Python
  function per row. That is why derived columns and business rules are
  expressions rather than callbacks.
* **Chunking costs nothing in time and everything in memory headroom.** On this
  dataset the chunked run was marginally faster and its resident memory did not
  grow measurably, while the in-memory run needed 24 MB for a 5 MB file. The
  ratio grows with the file: the whole-file path needs roughly
  `file size x 2-3` to read, transform and write, chunking needs
  `chunk size x 2-3`.

Absolute numbers depend on the machine; the ratios are what transfer.

## Where the time goes

| Operation | Cost | Notes |
| --- | --- | --- |
| Format detection | O(1) | 32 magic bytes plus a 64 KB sample for encoding and delimiter |
| CSV read | I/O bound | pandas C parser; encoding and separator detected once |
| Schema inference | O(columns x 5,000) | Detection samples 5,000 non-null values per column |
| Type checks | Vectorised | One pass per column per check |
| Business rules | Vectorised | One pass per operator; nulls fail closed |
| Deduplication | O(n log n) | Hash-based; `subset` reduces the key width |
| Profiling | O(n) per column | `deep=True` memory accounting walks Python strings, which dominates for text-heavy data |
| Parquet write | CPU bound | Snappy by default; `zstd` is smaller and slower |

## Choosing a chunk size

```python
from universal_data import Dataset
from universal_data.observability.memory import suggest_chunk_size

sample = Dataset.read("huge.csv", nrows=5_000)
chunk = suggest_chunk_size(sample.frame, target_mb=128)
```

`suggest_chunk_size` measures the average row width on a sample and returns the
row count that keeps one chunk near the target. Row *width* is far more stable
than row count, so a 5,000-row sample is enough.

Rules of thumb:

* 128 MB per chunk is a good default on a machine with several GB free.
* Smaller chunks mean more per-chunk overhead (schema inference, writer flushes).
* Larger chunks mean fewer pandas copies but a higher peak.

## What chunked mode cannot do

Streaming sees one chunk at a time, so anything that needs the whole dataset
does not work:

* deduplication across chunk boundaries (`remove_duplicates` only sees its own
  chunk);
* global sorting and global aggregation;
* validation counts, which are reported per chunk.

For those, either process in memory or run a two-pass job: stream a reduced
dataset out, then aggregate it whole.

Chunked CSV reading also infers dtypes per chunk, so a column that is numeric in
the first million rows can arrive as text later. The Parquet writer casts each
chunk to the schema of the first one and raises a clear `ExportError` naming the
drifting columns when the cast is impossible. Pin the types to avoid it:

```python
Pipeline("orders").clean(convert_types={"amount": "float"}).run_chunked(src, dst)
```

## Memory accounting

```python
from universal_data.observability.memory import MemoryTracker, dataframe_memory_mb

dataframe_memory_mb(frame)                 # what pandas holds, strings included
MemoryTracker().snapshot()                 # process RSS, available memory, percent used
MemoryTracker().check_headroom(size_mb)    # warning text when a transform would not fit
```

`Pipeline` warns when the loaded dataset exceeds `memory_warning_mb` (512 MB by
default) and adds the headroom estimate. The factor of two in the estimate is
not arbitrary: most pandas operations copy, so a 1 GB frame needs about 2 GB to
be transformed.

## Deliberate omissions

* **No multiprocessing.** Pandas operations release the GIL in the C layer and
  are already vectorised; a process pool would add pickling of DataFrames across
  process boundaries, which usually costs more than it saves. Where parallelism
  helps - many independent files - the natural unit is one process per file,
  which the CLI already supports through the shell.
* **No async.** Ingestion is I/O bound only in the API client, and the
  bottleneck there is the remote rate limit, not local concurrency. Adding an
  async stack would colour the whole codebase for a gain that a rate limiter
  cancels out.

Both would be added behind the existing interfaces if a real workload justified
them: `Reader`/`Writer` and `AuthStrategy` do not assume synchronous execution.
