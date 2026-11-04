# Modeling notes

All analysis lives in [the single notebook](../../notebooks/nyc_taxi_analysis.ipynb). Run it from a fresh kernel. The notebook first produces hourly aggregates, then separates forecasting from explanatory weather analysis.

## Forecast evaluation

The outcome is completed pickups per zone and New York clock hour. The top 50 zones are selected by training pickups only. January–August is training, September–October is validation, and November–December is the held-out test.

Calendar models use sparse least squares on `log1p(pickups / exposure_hours)`:

| Model | Added predictors |
|---|---|
| M0 | Intercept |
| M1 | Hour of day |
| M2 | Day of week |
| M3 | Annual Fourier harmonics |
| M4 | Zone effects |
| M5 | Weekend-specific hourly curve |
| M6 | Log pickup rate in the same zone and clock hour seven days earlier |

The ladder is cumulative. Annual harmonics can be evaluated in future months; categorical month coefficients cannot be estimated for months absent from training. A single year still limits how well seasonality can be established.

At the repeated autumn clock label, `exposure_hours = 2`; it is 1 otherwise. Use a training-only smearing factor to back-transform to hourly pickup rates, then multiply by the current exposure to predict counts. The weekly baseline is the lagged count divided by lagged exposure, multiplied by current exposure. The unadjusted lag count remains available for audit.

All ladder models fit the same training rows with a known seven-day lag, excluding the first week's warmup and the spring transition's unavailable lag. Choose the model by validation MAE, with the weekly baseline itself included as a candidate, and evaluate that selection on the final test once. Coefficients and smearing remain fixed at their training estimates. Compare count-scale errors against the weekly baseline on the same rows. Read the notebook's sample/metric tables for actual row counts and results.

This is **rolling one-hour-ahead evaluation**. The observed seven-day lag is available at each prediction time, including earlier observations within the validation or test period. It is not a two-month forecast from a fixed November origin. Actual future weather and same-hour fare, distance or trip counts are not forecast predictors.

## Weather associations

Use the all-zone city-hour total rather than treating one city-wide weather observation as 50 independent observations. Fit `log1p(city_pickups / exposure_hours)` on calendar controls and seven weather predictors, using training hours only. The notebook reports day-clustered standard errors with HAC sensitivity, residual/fitted and QQ plots, an autocorrelation plot and a Breusch–Pagan diagnostic.

Robust F-tests assess weather blocks. Correlations and auxiliary regressions assess shared information among weather predictors; **do not use VIF**. For FWL, regress the outcome and each of the seven weather variables on the complete calendar design, then regress the residualized outcome on all seven residualized weather columns together. Verify the coefficients against the full regression and obtain reported uncertainty from that full regression.

These coefficients are conditional associations, not causal weather effects. Completed pickups also depend on taxi supply; they do not measure unmet demand. A shared daily curve and one weather location are additional simplifications.

## Sample and output conventions

- Use the complete clock grid, including zero-pickup zone-hours. The 2025 spring gap is absent; the repeated autumn label combines two elapsed hours and is exposure-adjusted.
- The shifted weather endpoint at December 31, 23:00 is unavailable and excluded from the weather-complete sample. Lag features have their own availability checks.
- Make comparisons on the same eligible rows. Fit preprocessing, zone selection and smearing on training data only.
- Keep forecast validation/test metrics separate from train-only explanatory inference.
- Tables and figures are written under `outputs/`. Every reported result should identify its model, sample and uncertainty method.
- Use [the report plan](report_plan.md) to select material for the report. Do not write numerical findings until the real inputs have been run.
