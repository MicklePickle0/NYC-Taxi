# CLAUDE.md: NYC Taxi Demand

## Data files

| File | Details |
|---|---|
| `data/taxi.csv` | ~5.3 GB merged CSV of 2025 yellow taxi trips (Jan 1 to Dec 31). Columns: `VendorID, tpep_pickup_datetime, tpep_dropoff_datetime, passenger_count, trip_distance, RatecodeID, store_and_fwd_flag, PULocationID, DOLocationID, payment_type, fare_amount, extra, mta_tax, tip_amount, tolls_amount, improvement_surcharge, total_amount, congestion_surcharge, Airport_fee, cbd_congestion_fee`. 48,722,602 rows. |
| `data/weather.csv` | Open-Meteo hourly weather, downloaded with `timezone=America/New_York`. 3 metadata lines before the header (`skiprows=3`). Columns: `temperature_2m, precipitation, rain, snowfall, snow_depth, is_day, apparent_temperature, sunshine_duration, cloud_cover, shortwave_radiation` (units in the header names). `cloud_cover` and `shortwave_radiation` were added in a re-download so we can measure darkness and brightness. The file has no sun-elevation column; calculate it from lat/lon/time if needed (e.g. `pvlib`). 8,760 rows, no missing hours, no nulls. |

## Data rules

- **Never modify the raw data files.** Load them into dataframes (or read with DuckDB) and work on the copy in memory.
- Save cleaned outputs to new files in `data/processed/`, never over the originals.

## Data quality findings (quick check)

From the first 2M trip rows (early January) and the last rows of the file:

- About 41.5k rows (2%) have a negative `fare_amount`.
- About 6.4k rows are in unknown zones 264 and 265.
- 21 rows have a pickup outside 2025.
- 3 rows have a dropoff before the pickup.
- Pickup times load as proper `TIMESTAMP`s in DuckDB.
- The last rows (Dec 31) have blank `passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge` and `Airport_fee`. Don't impute these. Pickup demand only needs pickup time and zone.

## Daylight saving: trips vs. weather

**Trips follow real New York clock time.** On 2025-03-09, 02:00 has 0 pickups (the hour doesn't exist). On 2025-11-02, 01:00 has about 1.6× the normal pickups, because two real hours share that label.

**Weather is fixed UTC−5 (EST) all year,** despite being downloaded with `timezone=America/New_York`:
- The header shows a single fixed offset (`utc_offset_seconds = -18000`, `GMT-5`).
- Both DST days have 24 rows, and 2025-03-09 02:00 is present.
- The first daylight hour (`is_day`) doesn't jump across either DST change.

So from Mar 9 to Nov 2, weather labels are one hour behind the trip hours unless converted.

**Fix** (in `scripts/cleaning.ipynb`):
- Localize the weather times to `Etc/GMT+5`, convert to `America/New_York`, then drop the timezone.
- Mar 9 02:00 drops out on its own.
- The duplicated Nov 2 01:00 is merged to match the trips: sum `precipitation`, `rain`, `snowfall` and `sunshine_duration`; average `temperature_2m`, `apparent_temperature`, `snow_depth`, `cloud_cover` and `shortwave_radiation`; take the max of `is_day`.
- The result has 8,759 unique hours.

Done in `scripts/01_clean_weather.py`: Open-Meteo labels `precipitation`, `rain`, `snowfall`, `sunshine_duration` and `shortwave_radiation` by the end of the hour they cover, so they need shifting back one hour after the timezone conversion. This is confirmed for `shortwave_radiation`: after conversion, the first non-zero value on 2025-06-21 is at 06:00, but sunrise is at 05:25. `cloud_cover`, `temperature_2m`, `is_day` and `snow_depth` are instantaneous and are not shifted.

## Pipeline scripts (run in this order, from the repo root)

Every cut and the reason for it is documented in `docs/corrections.md`. Update it whenever a cleaning rule changes.

1. `scripts/01_clean_weather.py`: DST fix, end-of-hour shift and column drops. Writes `data/processed/weather_hourly.parquet`. Use `--precip {precipitation,rain,both}` to choose which precipitation columns to keep.
2. `scripts/02_clean_taxi.py`: column drops and row rules in DuckDB. Writes `data/processed/taxi_clean.parquet` (~1 GB, 46.6M trips) and `taxi_cleaning_report.csv`. A full run takes a few minutes.
3. `scripts/03_aggregate_zone_hour.py`: groups trips into zone × hour, zero-filled. Writes `data/processed/zone_hour.parquet` (2,303,617 rows = 263 zones × 8,759 hours).
4. `scripts/04_build_model_table.py`: joins weather onto zone-hour. Writes `data/processed/model_table.parquet`, the table to model from.

5. `scripts/05_statistical_tests.ipynb`: handed over to a group member. It currently contains only markdown guidelines, one section per proposal technique (linear regression ladder, heteroskedasticity with residual and QQ plots, F-tests, collinearity via F-tests (no VIF), model comparison, forecast accuracy, FWL), and no code yet. It should read only `model_table.parquet`.
6. `scripts/06_model_selection/`: model building, one notebook per step (`step1_assumptions` … `step7_chosen_model`) so a change only reruns that step. Shared setup (data, split, predictors, helpers) is in `model_utils.py`; steps pass results through JSON files in `data/processed/model_selection/` (step 3, 4 and 5 need step 2; step 6 needs 2 and 5; step 7 needs 6). Works on the top 50 zones with `log(pickups + 1)` and day-clustered standard errors. Random 80/20 train/test split by whole days (seed 860). Steps: residual and QQ plots; collinearity via auxiliary R² and F-tests (no VIF); F-tests of every term (non-significant terms are reported but **kept**, as the professor requires); the full model (14 terms, 170 parameters); BIC stepwise (forward/backward); comparison by adjusted R², AIC, BIC (train) and RMSE/MAE (test). The full model has the lowest test RMSE and is chosen. Predictors include `lag24`/`lag168` (same zone and clock hour 1 and 7 days earlier). Plots are saved to `docs/img/` for `docs/model_selection.md`, which explains every step in plain language. Built on `statsmodels` formulas. Step 5 (stepwise) takes ~5–8 min; the others take seconds to ~1.5 min each.

Steps 1 and 2 don't depend on each other. Step 3 needs step 2, step 4 needs steps 1 and 3, and steps 5 and 6 need step 4.

Install dependencies with `pip install -r requirements.txt`, then run `nbstripout --install` once so notebook outputs are stripped on commit.

`scripts/DONT USE_cleaning.ipynb` is the earlier notebook version of step 1, now replaced by `01_clean_weather.py`.

## Grouping trips by hour (implemented in step 3)

- Aggregate in DuckDB to one row per (`PULocationID`, `date_trunc('hour', tpep_pickup_datetime)`), with `pickups = count(*)`.
- Keep other trip features as **sums** (fare, distance, duration, passengers, tips), so they can be re-aggregated and averaged later.
- Keep categories as **counts** (airport trips, card payments, etc.).
- Join weather and calendar columns afterwards. They're the same for every trip in an hour.
- Same-hour trip summaries (e.g. average fare) are only known after the hour, so using them as predictors leaks information. Use them for exploration or as lagged features.
- When zero-filling zone-hours with no trips, set the sums to 0 and leave averages blank. Count the rows with a value for columns that have nulls (e.g. `passenger_count`).
- Save the result with DuckDB's `COPY ... TO 'data/processed/zone_hour.parquet'` (it doesn't need `pyarrow`). Expect about 2M rows (≈263 zones × 8,759 hours).
- Which extra features to keep is still for the team to decide.

## Environment notes

- Windows. Python libraries: DuckDB 2.0.0, pandas 3.0.1, numpy and matplotlib. **`pyarrow` is not installed**, so pandas can't read or write Parquet without it.
- DuckDB 2.0: relations no longer have `fetchone()`. Use `con.execute(sql).fetchall()` or `.df()`.
- Set `PYTHONIOENCODING=utf-8` when printing from the console, because the weather column names contain `°`.
- The trip CSV is too large for pandas. Query it with DuckDB and only select the columns you need.
