# Model Building and Selection

What `scripts/06_model_selection.ipynb` does, why each step is there, what problems it found and how they were fixed. All numbers come from the notebook's run on the cleaned 2025 data (pipeline steps 1–4).

Run it from the repo root after the pipeline: open the notebook and run all cells (about 3 minutes). It reads only `data/processed/model_table.parquet`.

---

## 1. Setup

### What is modelled

| Choice | Value | Why |
|---|---|---|
| Unit | one zone in one clock hour | The dispatch decision is "how many taxis, where, at what hour". |
| Zones | the 50 busiest, ranked by **training** pickups | The top 60 zones carry ~90% of pickups. The other zones are mostly zero-pickup hours, which a linear model handles badly. The ranking uses training rows only, so the test weeks don't influence which zones are chosen. |
| Target | `log(pickups + 1)` | See Step 1. The `+ 1` keeps zero-pickup hours (log 0 is undefined). |
| Library | `statsmodels` formulas (`smf.ols`) | Dummies, interactions, tests and AIC/BIC all come from the package. |

### Train / validation / test split

Within **every month**: the last 7 days are **test**, the 7 days before are **validation**, the rest is **train**.

- **Why not a plain date split** (e.g. Jan–Sep train, Oct–Dec test, as in the proposal)? The test months would have no training rows, so their month dummy would have no coefficient. With the split inside every month, each month dummy is estimated from that month's own training days.
- **Train** (227,850 rows) fits the models and runs every test.
- **Validation** (100,800 rows) picks the best model.
- **Test** (100,750 rows) is used once, at the very end, so the final error isn't flattered by choosing the model on it.

### Rows dropped

8,550 of 437,950 zone-hours (2%), because a predictor was missing:
- the first week of January (no pickups one week earlier to use as a lag),
- hours whose lag falls on 2025-03-09 02:00 (the skipped daylight-saving hour),
- 2025-12-31 23:00 (no shifted weather value, see `docs/corrections.md`).

### Candidate predictors

| Term | What | Parameters |
|---|---|---|
| `hour` | hour of day (dummies, reference = midnight) | 23 |
| `dow` | day of week (reference = Monday) | 6 |
| `month` | month (reference = January) | 11 |
| `zone` | pickup zone (reference = zone 237, the busiest) | 49 |
| `holiday` | US federal holiday (pandas `USFederalHolidayCalendar`) | 1 |
| `hour_x_weekend` | a separate daily curve for weekends | 23 |
| `zone_x_weekend` | each zone's own weekend shift | 49 |
| `temperature_2m`, `temp_sq` | temperature and its centred square (does demand rise in both cold and heat?) | 1 + 1 |
| `precipitation`, `snowfall`, `snow_depth`, `cloud_cover`, `shortwave_radiation`, `is_day` | weather | 1 each |
| `log_lag24`, `log_lag168` | log(1 + pickups) in the same zone and clock hour **1 day** and **7 days** earlier | 1 + 1 |

**Why the lags.** The proposal plans to use "historic pickup activity". Yesterday's and last week's demand at the same zone and hour are known before the day starts, so a dispatcher could use them in a day-ahead plan. Same-hour summaries (e.g. this hour's average fare) are **not** used, because they are only known after the hour ends.

**Perfect collinearity prevented by design.** Some combinations of dummies add up exactly to another column. OLS then has no unique solution (the "dummy variable trap"):
- Each categorical variable leaves out one reference level. Its dummies are differences from that level.
- There is no separate `weekend` dummy, because it equals `dow_6 + dow_7`.
- The weekend daily curve leaves out weekend midnight. Otherwise its 24 dummies would add up to the weekend flag, which the day-of-week dummies already contain. The weekend zone shift leaves out the reference zone for the same reason.
- `temp_sq` is centred (temperature minus its training mean, squared). An uncentred square rises almost in step with temperature itself; centring removes most of that correlation (r = −0.10).

**Hierarchy rule.** An interaction is only allowed in a model together with its main effects. `hour_x_weekend` needs `hour` and `dow`, `zone_x_weekend` needs `zone` and `dow`, and `temp_sq` needs `temperature_2m`. Without its main effects, an interaction's coefficients mix up the main effect and the interaction, and can't be interpreted.

### Why every test uses day-clustered standard errors

Weather and holidays are identical for all 50 zones in a given hour, and demand in one day's hours moves together (a rainy or busy day affects all of them). The 227,850 rows are therefore **not** 227,850 independent pieces of information about weather: they are closer to ~200 independent days. Ordinary standard errors treat every row as independent, so they come out far too small, and almost anything looks significant.

**Clustered standard errors** (`cov_type="cluster"`, grouped by day) allow any correlation between rows of the same day and any variance pattern. The F-tests then use ~200 days as the effective sample (denominator degrees of freedom = number of days − 1). This is also the fix for heteroskedasticity (Step 1).

---

## 2. Step 1: Checking the assumptions (residual plots, QQ plots)

### Why check

OLS gives unbiased coefficients under weak conditions. Its **standard errors, t-tests and F-tests** need more:
1. **Constant error variance (homoskedasticity).** The usual standard-error formula assumes every row's error has the same spread. If the spread changes with demand, the standard errors are wrong, and so is every p-value and F-test built on them.
2. **Roughly normal errors.** The F and t distributions are exact only for normal errors. With large samples this matters less (see below), but heavy tails still signal hours the model gets badly wrong.
3. **Sensible predictions.** A demand model shouldn't predict negative pickups.

Every later step (collinearity tests, significance tests) relies on these tests, so this is checked first.

### What was done

The model with **all** candidate terms was fitted twice, once on raw `pickups` and once on `log(pickups + 1)`. For each:

| Check | What it shows | What "good" looks like |
|---|---|---|
| Residuals vs fitted (20k random points) | whether errors depend on the prediction | an even horizontal band around 0 |
| Residual SD by fitted-value decile | how the spread changes with demand | a flat line |
| Normal QQ plot (5k points) | the shape of the error distribution | points on the 45° line |
| Breusch-Pagan test | formal test of constant variance (H₀: constant) | p > 0.05 |

**Reading the QQ plot.** Both axes are **z-scores**, not percentages, so negative values are normal. The x-axis is where each residual would sit if the errors were exactly normal (e.g. the 2.5th-percentile residual should be at −1.96). The y-axis is where it actually sits, as (residual − mean) / SD. Points below the line on the left, or above it on the right, mean more extreme errors than a normal distribution allows (heavy tails).

### What was found

| | Raw pickups | log(pickups + 1) |
|---|---|---|
| Residual SD, quietest → busiest decile | 14 → 97 pickups (grows ~7×) | 0.63 → 0.22 (shrinks ~3×) |
| Negative fitted values | **13%** | 0.02% |
| Skew of residuals | 1.41 (long right tail) | −0.38 |
| Excess kurtosis (0 = normal) | 7.6 | 6.6 |
| Breusch-Pagan p | ≈ 0 | ≈ 0 |
| R² | 0.732 | 0.921 |

The raw model violates all three conditions badly: the residual plot is a clear funnel, the spread grows sevenfold with demand, and one prediction in eight is a negative number of pickups.

### What fixed it, and why

**Fix 1: log-transform the target.** Pickups are counts, and count variability grows with the level: a zone averaging 300 pickups an hour swings by tens, a zone averaging 5 swings by a few. A log turns those proportional swings into similar-sized additive ones (a 10% swing is about 0.1 on the log scale at any level), so the variance becomes much more even. Coefficients then mean "% change in pickups", which is how demand effects naturally work (rain raises demand by a percentage, not by the same number of trips in every zone). Back-transformed predictions, `exp(ŷ) − 1`, can never be below 0.

What the log **doesn't** fully fix: the spread now runs the other way (SD 0.63 in the quietest decile vs 0.22 in the busiest). In quiet zone-hours, one or two extra trips is a large proportional change; the diagonal stripes in the plot are hours with exactly 0, 1, 2, ... pickups. The QQ plot still has heavy tails, mostly on the left (hours far below forecast).

**Fix 2: day-clustered standard errors.** These don't assume constant variance (or independence within a day), so the remaining heteroskedasticity no longer invalidates the F-tests. This is why every test from Step 2 on uses them.

**Why the leftover non-normality is acceptable.** With 228k rows (and ~200 day clusters), the central limit theorem makes the coefficient estimates approximately normal even when the errors aren't. So F-tests and confidence intervals stay valid. The heavy tails are still worth reporting: they are holidays and unusual events the model can't anticipate.

**Breusch-Pagan rejects for both** because, with 228k rows, it detects even tiny departures. The plots show the **size** of the problem, which is what matters.

**Alternatives not used:** a square-root target (a middle ground between raw and log), weighted least squares, or a Poisson / negative-binomial count model. The course techniques are OLS-based, so log + robust standard errors was the choice.

---

## 3. Step 2: Removing collinear predictors with F-tests (no VIF)

### Why it matters

**Collinearity** means a predictor is (almost) a linear combination of other predictors, so it carries little information of its own. OLS then can't tell their effects apart:
- standard errors become large, so real effects look insignificant;
- coefficients become unstable: they can change a lot or flip sign when a variable is added or removed;
- interpretation breaks down ("the effect of X holding Y fixed" is meaningless if X and Y always move together).

Collinearity doesn't bias predictions, but it ruins the significance tests in Step 3 and the interpretation of the full model. As the professor requires, it is assessed with **F-tests, not VIF**.

### What was done

1. **Correlation matrix** of the continuous predictors, for reference. Only one pair is strongly correlated: the two lags (r = 0.88). Next is `shortwave_radiation` with `is_day` (0.68).

2. **Auxiliary regressions (Farrar-Glauber F-test).** Each continuous predictor is regressed on every other main-effect predictor (calendar dummies, zone dummies, the other continuous predictors):
   - **Auxiliary R²:** how much of the predictor the others explain. Above 0.8 = serious collinearity, so the predictor is **flagged**.
   - **Overall F:** tests whether the others explain it at all. With 228k rows this always rejects, so the size of R² decides.
   - **Partial F** (`anova_lm`, calendar + zone vs calendar + zone + other continuous predictors) and **partial R²:** whether the overlap is with the **calendar** or with the **other weather/lag variables**.

3. **Joint vs individual F-tests** on the all-candidate model (day-clustered):
   - For each pair with |r| > 0.7: if the joint test that both coefficients are 0 rejects, but an individual test doesn't, the two share one piece of information. Neither adds anything once the other is in. The weaker one is dropped.
   - For each **flagged** predictor: if its own F-test (given every other predictor) isn't significant, its information is already in the model, so it is dropped.

### What was found

| Predictor | R² from calendar + zone | Auxiliary R² (all) | Partial R² (others) | Flagged | Own p (clustered) | Decision |
|---|---|---|---|---|---|---|
| `temperature_2m` | 0.886 | **0.897** | 0.091 | yes | 0.066 | **drop** |
| `shortwave_radiation` | 0.765 | **0.835** | 0.298 | yes | 0.020 | keep (then dropped in Step 3) |
| `is_day` | 0.821 | **0.822** | 0.006 | yes | 0.147 | **drop** |
| `log_lag24` | 0.729 | **0.826** | 0.355 | yes | < 0.001 | keep |
| `log_lag168` | 0.722 | **0.821** | 0.355 | yes | < 0.001 | keep |
| `temp_sq` | 0.448 | 0.545 | 0.175 | no | – | dropped with temperature (hierarchy rule) |
| `snow_depth` | 0.480 | 0.509 | 0.056 | no | – | – |
| `cloud_cover`, `precipitation`, `snowfall` | ≤ 0.12 | ≤ 0.27 | – | no | – | – |

Lag pair (r = 0.88): joint p < 0.001 and both individual p < 0.001. **Keep both:** they are correlated, but each still adds its own information (yesterday captures recent conditions, last week captures the weekday pattern).

### Why dropping fixes it

- **Temperature and `is_day` are almost entirely explained by the calendar.** Month and hour dummies predict 89% of temperature (seasons and the daily cycle) and 82% of `is_day` (daylight depends on hour and month). What is left is a small amount of unusual variation, and it doesn't significantly predict demand once the calendar and lags are in. Dropping them removes a near-duplicate of the month × hour pattern. The calendar dummies already carry that information, so nothing is lost, and the remaining coefficients get smaller standard errors.
- **`temp_sq` goes with temperature**, by the hierarchy rule.

**Caveat for the report:** dropping temperature means it doesn't help **predict zone-hour demand** once the calendar and last week's demand are known. It doesn't mean temperature has no effect on demand. That question is answered by the FWL analysis in `05_statistical_tests.ipynb`.

---

## 4. Step 3: Keeping only significant predictors

### Why

Terms that don't significantly explain demand add noise. They take up parameters, can worsen out-of-sample forecasts, and clutter the interpretation. The full model should contain only terms with evidence behind them.

### What was done

**Backward elimination** from the 14 terms left after Step 2:
1. Fit the model with day-clustered standard errors.
2. Test each term with a clustered F-test (`wald_test_terms`). A dummy block such as `zone` is tested as a whole: H₀ = all 49 zone coefficients are 0. Testing dummies one by one would make the result depend on the reference level.
3. If the least significant term has p > 0.05, drop it and go back to 1. A main effect isn't dropped while an interaction that needs it is still in.

Dropping one at a time matters: removing a term changes the others' p-values, so they are re-tested after every drop.

### What was found

Dropped, in order:

| Order | Term | p |
|---|---|---|
| 1 | `cloud_cover` | 0.281 |
| 2 | `shortwave_radiation` | 0.230 |
| 3 | `snowfall` | 0.169 |
| 4 | `snow_depth` | 0.120 |
| 5 | `holiday` | 0.069 |

All 9 remaining terms are significant at the 1% level (the largest p is 0.002, for precipitation).

- **Weather:** once the calendar and last week's demand are in, rain is the only weather variable that significantly shifts demand.
- **Snow:** 2025 had few snowy hours on training days, so the snow effects are estimated imprecisely.
- **`holiday`:** borderline. Only 5 holidays fall on training days, so there are few independent days to learn from. It may be worth forcing back in for forecasting the Thanksgiving and Christmas weeks.

**Caveat:** p-values after a selection procedure are somewhat optimistic, because the surviving terms were chosen *because* they looked significant. The out-of-sample evaluation in Step 6 is the honest check.

---

## 5. Step 4: The full model

The full model contains every term that passed Steps 2 and 3.

```
log(pickups + 1) ~ hour + dow + month + zone + precipitation + log_lag24 + log_lag168
                   + hour_x_weekend + zone_x_weekend
```

| Term | Parameters |
|---|---|
| Intercept | 1 |
| `hour` | 23 |
| `dow` | 6 |
| `month` | 11 |
| `zone` | 49 |
| `hour_x_weekend` | 23 |
| `zone_x_weekend` | 49 |
| `precipitation`, `log_lag24`, `log_lag168` | 3 |
| **Total** | **165** |

| Fit (training rows) | Value |
|---|---|
| R² | 0.9201 |
| Adjusted R² | 0.9201 |
| AIC | 195,259.8 |
| BIC | 196,965.3 |

Coefficients (day-clustered standard errors):

| Predictor | Coefficient | Clustered SE | p | Meaning |
|---|---|---|---|---|
| `precipitation` | 0.0223 | 0.0072 | 0.002 | each mm of rain in the hour → **+2.3%** pickups (`100 × (e^0.0223 − 1)`) |
| `log_lag24` | 0.2525 | 0.0263 | < 0.001 | elasticity: 1% more pickups at this hour yesterday → +0.25% now |
| `log_lag168` | 0.5747 | 0.0310 | < 0.001 | elasticity: 1% more pickups at this hour last week → +0.57% now |

Last week's demand matters more than yesterday's, because it matches the zone, weekday and hour exactly.

---

## 6. Step 5: Stepwise selection

### Why

The full model is one model. Stepwise selection builds a **sequence of alternative models**, showing which terms matter most and how much each adds. The models can then be compared on several criteria (Step 6).

### What was done

**Bidirectional stepwise selection scored by BIC**, over the 9 terms of the full model:
- At each step, try adding every term that is out and removing every term that is in. Fit each candidate with `smf.ols` and make the move that lowers BIC the most. Stop when no move lowers BIC. The hierarchy rule is enforced.
- **Forward** starts from the intercept only, and each step is saved as a new model (S1, S2, ...). **Backward** starts from the full model.
- `statsmodels` has no stepwise function, so the search loop is written in the notebook. Every fit and BIC value comes from `statsmodels`. 55 models were fitted.

**Why BIC rather than AIC or p-values?** BIC = −2 log-likelihood + k × ln(n). Its penalty per parameter is ln(227,850) ≈ 12.3, vs 2 for AIC. With this many rows almost any term improves the likelihood a little, so AIC would accept nearly everything. BIC's larger penalty is the stricter, more parsimonious choice. Using a criterion also avoids running hundreds of p-value tests.

### What was found

| Step | Adds | BIC |
|---|---|---|
| S1 | `log_lag168` | 264,290 |
| S2 | `log_lag24` | 223,222 |
| S3 | `month` | 218,516 |
| S4 | `hour` | 216,350 |
| S5 | `zone` | 213,923 |
| S6 | `dow` | 211,738 |
| S7 | `hour_x_weekend` | 198,496 |
| S8 | `zone_x_weekend` | 197,286 |
| S9 | `precipitation` (= full model) | 196,965 |

- **Forward ends exactly at the full model**, and backward removes nothing: BIC agrees with the significance tests in Step 3.
- **Most important:** the order shows the lags matter most, followed by the calendar.
- **Largest gains:** the biggest BIC drop after the lags is the weekend daily curve (S7). Weekend demand peaks at different hours from weekday demand.

---

## 7. Step 6: Evaluating the models

### The criteria, and why each

| Criterion | Computed on | What it measures | Better |
|---|---|---|---|
| Adjusted R² | training rows | share of variation explained, with a penalty per parameter (plain R² can only rise when terms are added) | higher |
| AIC | training rows | fit vs complexity, light penalty (2 per parameter); aims at prediction | lower |
| BIC | training rows | fit vs complexity, heavy penalty (ln n per parameter); aims at finding the true model | lower |
| RMSE | **validation** weeks, in pickups | typical forecast error, large errors weigh more | lower |
| MAE | validation weeks, in pickups | average absolute forecast error | lower |
| RMSE (log) | validation weeks, log scale | same as RMSE, before converting back from logs | lower |

The first three only use the data the model was fitted on. RMSE and MAE test the model on **weeks it hasn't seen**, which is what forecasting needs, so validation RMSE picks the model. Predictions are converted back to pickups with `exp(ŷ) − 1`.

The **last-week baseline** (predict the pickups seen in the same zone and hour 7 days earlier) is the minimum a model must beat to be worth using.

### What was found

| Model | Parameters | Adjusted R² | AIC | BIC | RMSE (val) | MAE (val) | RMSE log (val) |
|---|---|---|---|---|---|---|---|
| S1: last week | 2 | 0.8917 | 264,269 | 264,290 | 32.64 | 18.80 | 0.402 |
| **S2: + yesterday** | **3** | 0.9096 | 223,191 | 223,222 | **29.94** | **17.43** | **0.370** |
| S3: + month | 14 | 0.9115 | 218,371 | 218,516 | 32.21 | 18.69 | 0.379 |
| S4: + hour | 37 | 0.9124 | 215,968 | 216,350 | 32.51 | 18.88 | 0.379 |
| S5: + zone | 86 | 0.9136 | 213,034 | 213,923 | 33.29 | 19.08 | 0.377 |
| S6: + day of week | 92 | 0.9144 | 210,787 | 211,738 | 33.98 | 19.47 | 0.377 |
| S7: + weekend curve | 115 | 0.9193 | 197,307 | 198,496 | 33.53 | 19.29 | 0.372 |
| S8: + zone weekend shift | 164 | 0.9200 | 195,591 | 197,286 | 33.22 | 19.13 | 0.371 |
| **S9: + rain = full model** | **165** | **0.9201** | **195,260** | **196,965** | 33.25 | 19.13 | 0.371 |
| Baseline: last week | – | – | – | – | 34.53 | 19.29 | 0.402 |

**The criteria disagree.**
- **In-sample**, every criterion prefers the full model: adjusted R² rises and AIC and BIC fall at every step.
- **Out-of-sample**, S2 (just the two lags, 3 parameters) has the lowest validation RMSE, MAE and log RMSE. Every model beats the baseline.

**Why.** Last week's demand at the same zone and hour already contains the zone × weekday × hour pattern that the dummies estimate. The dummies fit the *average* training-day pattern a little better, but that gain doesn't carry over to other weeks. Month dummies cost the most (S2 → S3: RMSE +2.3), because the training days are the first half of each month and the validation days come later. With 227,850 rows, even BIC's penalty of 12.3 per parameter is too small to reject terms that are statistically real but don't help forecasts. The log-scale RMSE shows the same ranking, so the gap isn't caused by converting predictions back from logs.

**Decision.**
- **S2 is the forecasting model** (lowest validation error).
- **The full model is the explanatory model:** use it for significance and effect sizes (e.g. the rain effect), since every term in it is significant after controlling for the others.

---

## 8. Step 7: Final check on the test weeks

S2 was refitted on train + validation and scored once on the test weeks.

| | RMSE | MAE |
|---|---|---|
| S2 | **34.41** | **19.30** |
| Baseline: last week | 41.13 | 22.38 |

- **Overall:** S2 cuts RMSE by 16% and MAE by 14% compared with the baseline.
- **By zone:** S2 beats the baseline in all 50 zones.
- **Why test errors are higher than validation errors:** the test weeks contain Memorial Day, Thanksgiving and Christmas, when last week's demand is a poor guide.
- **Test residual plots:** same pattern as Step 1. The variance is larger in quiet hours, and the left tail is heavy (hours far below forecast, mostly holidays).
- **Example plot:** zone 161 (Midtown Center) in the June test week. S2 follows the daily cycle and smooths the extremes of last week's curve, but it can't anticipate unusual peaks.

`exp(ŷ) − 1` predicts a median-like value, so it slightly under-predicts mean demand.

---

## 9. Limitations and next steps

- **Demand = completed pickups.** Unmet demand isn't observed, so busy hours are probably underestimated (see `docs/proposal_review.md`).
- **The lags assume a forecast made one day ahead.** For a longer horizon, remove `log_lag24` and rerun Steps 5–7.
- **Actual weather is used.** A live system would only have a forecast.
- **Holidays:** the flag was borderline (p = 0.07) on only 5 training holidays. Forcing it in, or adding a "day before/after holiday" flag, may help in the test weeks.
- **Better forecasts:** interact the lags with the hour of day, add more weeks of lagged demand, or try a count model (Poisson / negative binomial).
- **Selection bias in p-values** (Step 3): report the full model's p-values with that caveat, and rely on the validation and test results for model choice.
