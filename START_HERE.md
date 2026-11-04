# Start here

1. Install the dependencies as shown in [README.md](README.md).
2. Obtain the two raw files from the team. They are local inputs, not tracked in Git:
   - `data/taxi.csv`
   - `data/weather.csv`
3. Open **[notebooks/nyc_taxi_analysis.ipynb](notebooks/nyc_taxi_analysis.ipynb)**. Check the configuration cell: project root, input path properties, DuckDB memory/thread limits and `rebuild` option. This workflow expects 2025 inputs. Set `NYC_TAXI_ROOT` if launching outside the repository or `notebooks/` directory.
4. Restart the kernel and run all cells. The notebook builds or reuses its hourly cache, then runs the entire analysis.
5. Review the cleaning report and sample checks before using the exported figures and tables in `outputs/`. Record the selected forecast model and its held-out test metrics from this run.

The notebook is the only executable analysis entry point. It does not need a cleaned trip Parquet file, a separate preparation notebook or external pipeline scripts.

Expect an on-disk staging database during the first run. The design avoids loading the full trip CSV into pandas, but still needs substantial free disk space. [Performance notes](docs/performance_review.md) explain the limits and cache behavior. The raw data is unavailable in this checkout, so no full-file runtime or cleaning count is asserted here.

Use [docs/modeling/README.md](docs/modeling/README.md) to understand the forecast split and inference, [docs/corrections.md](docs/corrections.md) for cleaning rules, and [docs/modeling/report_plan.md](docs/modeling/report_plan.md) for the write-up.
