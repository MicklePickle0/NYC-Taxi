# 07_keith_changes: FWL rebuild of 06_model_selection

Three self-contained notebooks. Each runs the whole model-building pipeline in one file, in about a minute, and reads
only `data/processed/model_table.parquet` (build it with `scripts/01_` to `04_`). Nothing is written to disk.

| Notebook | Lags (yesterday, last week) | Model selection |
|---|---|---|
| `07a_lags_stepwise.ipynb` | yes | stepwise by BIC |
| `07b_no_lags_stepwise.ipynb` | no | stepwise by BIC |
| `07c_no_lags_holistic.ipynb` | no | holistic: adjusted R², AIC, BIC, explainability first |

## Changes from 06 (all three)

- **New order:** lags (07a only) → random 80/20 split by whole days (seed 860, as in 06) → FWL deseasonalizing →
  regression → residual plot → QQ plot → F-tests and collinearity → model selection → forecast accuracy (RMSE and MAE
  on the test days).
- **Frisch-Waugh-Lovell replaces the calendar dummies.** Z is the constant, day of week, month, holiday, and the hour
  pattern separately for weekdays and weekends (65 columns). On the training days, Y and every regular predictor
  (zone, zone × weekend, weather, lags) are regressed on Z, and every later step uses the residuals. Test days are
  deseasonalized with the training-day coefficients. Z's parameters still count in adjusted R², AIC and BIC.
- **Collinearity is judged after deseasonalizing:** a predictor is flagged if the *other regular predictors* explain
  more than 80% of it. Overlap with the calendar is no longer a reason to drop a predictor (FWL already removed it), so
  temperature and `is_day`, which 06 dropped, stay in.
- **The selection method picks the model; the test days only evaluate it.** 06 picked by lowest test RMSE.
- `snow_depth` is in cm (was m), so its coefficient reads per cm like snowfall.

## Main results

| | 07a | 07b | 07c |
|---|---|---|---|
| Chosen model | stepwise S10 (171 parameters) | stepwise S8 (169) | M4, every block (171) |
| Adjusted R² (train) | 0.916 | 0.788 | 0.788 |
| Test RMSE / MAE | 33.99 / 19.15 | 60.31 / 34.69 | 60.31 / 34.69 |
| Rain, per mm | +2.1% | +1.5% | +1.5% |

Same hour last week scores RMSE 40.12. Without lags, weather barely changes forecast accuracy: calendar and zone
alone score 60.25. The no-lag models lose to last week because every zone shares one daily curve. See the notes at the
end of 07b and 07c.
