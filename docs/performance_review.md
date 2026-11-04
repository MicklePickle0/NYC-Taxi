# Performance and cache notes

The notebook is structured for a CSV with more than 50 million trips. The raw data is unavailable in this checkout, so there is **no measured full-file runtime, peak RSS, disk budget or production cleaning count** to report. Synthetic execution verifies the workflow on a small fixture, not its speed at full scale.

## Where work and memory go

| Stage | Strategy |
|---|---|
| Parse | Read the trip CSV once with explicit types into a file-backed DuckDB table. Avoid repeated raw CSV scans. |
| Validate and clean | Assign the first failed rule and count sequential removals in SQL over the stage. Exact duplicate removal compares all 20 original columns. |
| Aggregate | Count surviving trips by zone and clock hour in SQL, then join the complete grid. Do not create a full cleaned trip Parquet or full trip dataframe. |
| Prepare samples | Cache hourly aggregates, city totals and the training-selected top-zone sample as Parquet. |
| Fit forecasts | Use SciPy sparse least squares. Do not form a dense roughly 438,000-row categorical design matrix. |
| Infer weather associations | Use the roughly 8,758-row city-hour table rather than fitting a large formula model over the zone panel. |

The full grid is 263 × 8,759 = 2,303,617 rows in 2025. The weather-complete top-50 sample has about 438,000 rows before lag availability restrictions; the city series has 8,758 hours. These sizes are calendar/model expectations, not a benchmark.

DuckDB defaults to **2 GB working memory and four threads**, with temporary files on disk. The limit does not cap pandas, SciPy, plotting or the Python process as a whole. First-run execution needs room for the parsed raw stage, DuckDB spill and final outputs concurrently. Exact deduplication is potentially expensive but preserves row identity rather than assuming hashes cannot collide. Consult [DuckDB's workload tuning guide](https://duckdb.org/docs/stable/guides/performance/how_to_tune_workloads) and [SciPy's LSQR documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html) when tuning for another machine.

## Repeated runs and disk reclamation

The cache manifest records the source paths, sizes and nanosecond modification times, relevant configuration and a cache version. A matching manifest permits reuse of the processed hourly/modeling Parquet files. It is a **metadata fingerprint**, not a cryptographic verification of the source bytes. Force a rebuild after a replacement that preserves metadata or when validating a new implementation.

The raw trip staging table is dropped after successful trip aggregation. DuckDB may retain its allocated disk space inside `data/processed/taxi_work.duckdb`; dropping the table does not promise filesystem shrinkage. Close the notebook's connection before deleting that working database to reclaim disk. Retain the processed Parquet files and manifest if you want a later run to use the cache.

When changing a cleaning rule, weather alignment, sample definition or output schema, update the cache version or force a rebuild. Never use cache presence alone as evidence that outputs correspond to the current configuration.

## What to measure on a real run

The notebook writes `outputs/pipeline_benchmark.json` with pipeline wall time, cache-hit status, library versions, settings and cleaning counts. Peak process RSS is included where the platform's `resource` module supports it; it includes earlier work in the same kernel and is unavailable from this mechanism on Windows. It is not a per-cell allocation measurement or a whole-notebook runtime.

For a full-data benchmark, also record input file bytes, machine RAM and disk type, total notebook runtime and maximum staging/spill disk use. Publish those measurements with their configuration before making a runtime or memory guarantee for the full dataset.

## Verification performed for this change

The full test suite passed 17 checks with Python 3.14 and the pinned dependencies. A fresh Jupyter kernel executed every notebook cell on 118,258 synthetic trips spanning 2025, building the complete 2,303,617-row zone-hour grid and a three-zone forecast sample. The run produced the expected analysis exports and checked FWL against the full regression. Cache reuse preserved checkpoints; appending an exact duplicate invalidated the cache, increased the raw count and left retained pickups unchanged.

Separate fixtures cover exact original-row equality, null-valued cleaning exceptions, nonfinite dropoff times, unexpected source columns, fixed-EST weather alignment, both DST transitions, train-only zone ranking/scaling, sparse versus dense least squares, exposure-aware retransformation and zero-demand metrics. Notebook schema, code syntax, stripped cell outputs and local Markdown links were also checked. These are correctness checks, not measurements at 50M+ scale.
