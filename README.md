# NYC taxi pickups: one notebook

Run [notebooks/nyc_taxi_analysis.ipynb](notebooks/nyc_taxi_analysis.ipynb) from a fresh kernel. It contains the complete workflow: raw CSV validation, cleaning, hourly aggregation, exploration, model fitting, diagnostics, forecast evaluation and weather analysis. There are no prerequisite pipeline scripts or other analysis notebooks.

The pipeline is designed for a trip CSV with more than 50 million rows. DuckDB processes trips on disk; pandas receives only hourly aggregates and the top-50-zone modeling sample. A full-data runtime or peak-memory result has **not** been measured in this checkout because the raw files are unavailable.

## Run it

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook notebooks/nyc_taxi_analysis.ipynb
```

On Windows, create the environment with `py -m venv .venv` and activate with `.venv\Scripts\activate`. Use Python 3.11 or newer. Put the raw inputs at these default paths, or change the notebook's configuration cell:

- `data/taxi.csv`: the original 20-column 2025 yellow-taxi trip CSV.
- `data/weather.csv`: hourly weather with three metadata lines before its header.

Restart the kernel and run all cells. The raw files are never overwritten. Processed Parquet files, the disk-backed DuckDB working database and its spill directory live under `data/processed/`; report tables and figures go under `outputs/`. Generated data and outputs stay out of Git.

The default DuckDB limits are **2 GB memory and four threads**. These constrain DuckDB's working memory, not the entire Python process. Keep enough free disk space for the parsed trip staging table, spill files and aggregates; see [performance notes](docs/performance_review.md).

## What the analysis answers

- **Forecasting:** predict completed pickups for the training-selected top 50 zones, with January–August for training, September–October for validation and November–December for the final test. Compare calendar regression models and a seven-day lag model with a same-zone, same-clock-hour-last-week baseline. Evaluation is rolling one hour ahead, using observations already available at that time.
- **Weather explanation:** estimate city-level associations with seven weather measures after calendar adjustment. The notebook includes clustered inference, HAC sensitivity, residual diagnostics, robust F-tests, auxiliary regressions and a joint Frisch–Waugh–Lovell check.

The repeated autumn clock hour contains two elapsed hours. Models work with pickups per exposure hour and convert predictions back to counts for scoring. Actual weather and same-hour trip summaries are excluded from forecast predictors. Weather coefficients describe associations with completed pickups; they do not establish a causal effect or measure people who could not find a taxi.

## Reuse and checks

On later runs, the notebook reuses processed files when source paths, file sizes, modification timestamps and relevant configuration match its cache manifest. This is a metadata fingerprint, not a content checksum. Set `rebuild=True` in the configuration cell after replacing data while preserving those attributes or when you want a clean rebuild.

Useful exports include `outputs/analysis_summary.json`, `outputs/pipeline_benchmark.json`, and `outputs/tables/validation_scores.csv` and `final_test_scores.csv`. The benchmark distinguishes a rebuild from a cache hit.

For development checks:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests exercise a synthetic fixture, including fresh-kernel notebook execution. Passing them verifies behavior on that fixture; it does not benchmark the full trip file.

Read [START_HERE.md](START_HERE.md) for the run checklist, [cleaning rules](docs/corrections.md) for data decisions, and [modeling notes](docs/modeling/README.md) for interpretation. [The report plan](docs/modeling/report_plan.md) explains how to write up the generated results.
