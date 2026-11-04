# Repository guidance

## Entry point

`notebooks/nyc_taxi_analysis.ipynb` is the single runnable workflow. Keep pipeline helpers, configuration, analysis and explanatory text together in that notebook. Do not reintroduce prerequisite scripts or per-technique notebooks. See `README.md` for installation and execution.

## Data and scale

- Never modify raw inputs. Default paths are `data/taxi.csv` and `data/weather.csv`.
- Process trips with DuckDB SQL. Do not materialize the full trip file or a dense zone-panel design matrix in pandas.
- Parse the strictly typed 20-column trip CSV once into a file-backed staging table. Assign each row its first failed rule and compute sequential removal counts from that stage. Deduplicate exactly across all 20 original columns; a hash alone is not an identity check.
- Aggregate to zone-hour before pandas conversion. Build the complete 263-zone clock-hour grid so zero-pickup hours remain explicit.
- The default limits are 2 GB DuckDB memory and four threads, with disk spill under `data/processed/`. The full Python process can use more memory.
- Cache useful hourly/modeling outputs. Validate the manifest against source paths, sizes, modification nanoseconds, configuration and cache version. Metadata matching does not prove identical contents; retain an explicit force-rebuild option.
- Dropping staging tables releases logical data, but may leave the database file allocated. A closed working database can be deleted when disk-space reclamation is needed.
- Keep raw files, processed data, generated outputs and notebook cell outputs out of commits.

## Time handling

Taxi timestamps are New York clock labels. The configured weather source uses fixed EST labels; align interval weather to the hour it describes before combining repeated clock labels. Omit the nonexistent spring 02:00 label. At the autumn transition, combine the repeated 01:00 observations: sum interval amounts and average rates/instantaneous measures as implemented.

For 2025, the complete grid has 8,759 distinct clock labels and 2,303,617 zone-hours. Set `exposure_hours = 2` for November 2, 01:00, and 1 otherwise. The final shifted weather value is unavailable, so the weather-complete modeling grid has 8,758 hours. These are calendar expectations, not observed raw-trip counts. See `docs/corrections.md`.

## Modeling

- Select the top 50 zones using January–August training pickups only. Use September–October for validation and November–December for the final test.
- Forecast with sparse calendar OLS models and a lag-seven-days extension. Use annual Fourier terms rather than month categories that would be unseen in future months.
- Model `log1p(pickups / exposure_hours)`. Fit smearing on training residuals, back-transform to hourly rates, then multiply by the current exposure for count predictions. Exposure-adjust the weekly baseline in the same way; retain its original lag count for audit.
- Select on validation, then score the final test once. Keep competing methods on comparable available rows.
- Evaluate rolling one hour ahead. A seven-day observed lag is available at each forecast time; this is not a single forecast made on November 1 for the entire test period.
- Exclude actual future weather and same-hour trip summaries from forecast inputs. Use actual weather only for the separate explanatory analysis.
- Estimate weather associations on the city-hour series, with `log1p(city_pickups / exposure_hours)`, train-only calendar controls, day-clustered standard errors and HAC sensitivity. Report residual diagnostics, robust block tests and auxiliary regressions. Do not use VIF.
- For FWL, residualize the outcome and all seven weather columns against the full calendar design together. Check its coefficients against the full regression; take reported uncertainty from the full model.
- Completed pickups reflect supply and observed trips. Do not interpret them as latent unmet demand or interpret weather associations as causal effects.

## Verification and documentation

Install development dependencies and run `python -m pytest -q`. Tests must include a fresh-kernel notebook run on a synthetic fixture, with meaningful cleaning, DST, cache and split checks. Do not claim a full-data benchmark when the raw files were not run.

Update `docs/corrections.md` when a cleaning rule changes, and update `docs/modeling/README.md` when the split, predictors or inferential sample changes. Keep documentation about the current workflow; remove obsolete per-notebook guides.
