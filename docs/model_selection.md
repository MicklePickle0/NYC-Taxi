# Model Building and Selection

What the notebooks in `scripts/06_model_selection/` do, why each step is there, what they found and how problems were fixed. Every number comes from their run on the cleaned 2025 data.

### Files

Each step is its own notebook, so a small change only reruns that step. Shared setup (loading the data, the train/test split, the predictors and helper functions) is in `model_utils.py`. Steps pass their results on through small files in `data/processed/model_selection/`.

| Notebook | Step | Needs | Run time |
|---|---|---|---|
| `step1_assumptions.ipynb` | 1. Residual and QQ plots | – | ~15 s |
| `step2_collinearity.ipynb` | 2. Collinearity | – | ~1.5 min |
| `step3_significance.ipynb` | 3. Significance | step 2 | ~10 s |
| `step4_full_model.ipynb` | 4. Full model, incl. the full `statsmodels` coefficient table | step 2 | ~10 s |
| `step5_stepwise.ipynb` | 5. Stepwise selection | step 2 | ~5–8 min |
| `step6_compare.ipynb` | 6. Model comparison | steps 2 and 5 | ~1 min |
| `step7_chosen_model.ipynb` | 7. Closer look at the chosen model | step 6 | ~10 s |

The first time, build the data with pipeline steps 1–4 (see `CLAUDE.md`), then run the notebooks in order. After that, rerun only the step you changed and the steps that need it. The plots below are saved by the notebooks to `docs/img/`, so rerunning updates them.

---

## 1. Setup

### What is modelled

| Choice | Value | Why |
|---|---|---|
| What one row is | one zone in one hour | The dispatch decision is "how many taxis, where, at what hour". |
| Zones | the 50 busiest | They carry most of the pickups. Quiet zones are mostly hours with zero pickups, which a linear regression handles badly. |
| What is predicted | `log(pickups + 1)` | See Step 1. The `+ 1` is there because log(0) doesn't exist. |

### Train / test split (80/20, random)

**73 of the 365 days of 2025 (20%) are picked at random as test days; the other 80% are training days.** A fixed random seed means everyone who reruns the notebook gets the same split.

- **Train** (345,400 rows) is used to fit the models and run the tests.
- **Test** (84,000 rows) is used to measure prediction error on days the models have never seen, and to pick the best model.

**Why whole days, not individual rows?** In a given hour all 50 zones share the same weather, and neighbouring hours are very similar. If single rows were split at random, almost every test row would have near-copies in the training data, and the test error would look better than it really is. Keeping each day entirely in train or entirely in test avoids this. Every month still has training days (between 3 and 10 days per month went to test).

### Predictors

| Predictor | What it is |
|---|---|
| hour, day of week, month, zone | when and where the pickup happens (each as a set of 0/1 columns) |
| holiday | 1 on a US federal holiday |
| weekend daily pattern | weekends have different busy hours from weekdays |
| zone weekend shift | some zones are busier on weekends, others quieter |
| temperature (and its square) | the square tests whether demand rises in both cold *and* heat |
| rain, new snow, snow on the ground, cloud cover, sunlight, daylight (`is_day`) | weather |
| yesterday's and last week's pickups | pickups in the same zone and clock hour 1 day and 7 days earlier |

**Why yesterday's and last week's pickups?** The proposal plans to use "historic pickup activity". Both values are known before the day starts, so a dispatcher could use them in a next-day plan.

### Why every test uses "standard errors clustered by day"

The weather is the same for all 50 zones in a given hour, and a rainy or busy day affects all of its hours. So the 345,400 training rows aren't 345,400 independent observations of weather; they are closer to ~290 independent days. Normal regression tests assume every row is independent, which would make almost everything look significant.

Clustering by day tells the tests to treat rows from the same day as related, so the p-values are honest. It also fixes the uneven error spread found in Step 1.

---

## 2. Step 1: Checking the assumptions (residual plot, QQ plot)

### Why check

A regression's tests (p-values, F-tests) rely on two assumptions about the **errors** (actual demand minus predicted demand):
1. **The errors have the same spread at every demand level.** A quiet zone and a busy zone should be predicted about equally well. If the errors grow with demand, the tests are wrong.
2. **The errors are roughly normally distributed** (bell-shaped).

A demand model should also never predict a negative number of pickups.

All later steps rely on these tests, so this check comes first.

### What was done

The model with all predictors was fitted twice: once predicting raw `pickups`, once predicting `log(pickups + 1)`. For each, two plots:

| Plot | What it shows | Good looks like |
|---|---|---|
| Residuals vs fitted | whether the errors get bigger or smaller with the predicted demand | an even band around 0 |
| Normal QQ plot | whether the errors are normally distributed | points on the dashed line |

**Reading the QQ plot:** both axes are measured in standard deviations from the average, not percentages, so negative values are expected. Points bending away from the line at the ends mean some errors are more extreme than normal.

### What was found

![Residual plots and normal QQ plots for raw pickups (top) and log(pickups + 1) (bottom)](img/step1_assumptions.png)

| | Raw pickups | log(pickups + 1) |
|---|---|---|
| Residual plot | funnel: the busier the hour, the bigger the errors | much more even band |
| QQ plot | bends far away on the right | closer to the line; still bends at the ends |
| Negative predictions | **13%** | 0.01% |
| R² | 0.72 | 0.92 |

The raw model fails both checks, and one prediction in eight is a negative number of pickups.

### What fixed it, and why

**Fix 1: predict the log of pickups.** Busy zones naturally vary more in absolute numbers. A zone averaging 300 pickups an hour swings by tens; a zone averaging 5 swings by a few. The log turns these into **percentage** changes, which are similar in size for busy and quiet zones, so the errors even out. As a bonus:
- each effect reads as "% change in pickups", which is how demand really behaves (rain raises demand by a percentage, not by the same number of trips everywhere);
- predictions converted back from the log can never be negative.

What remains: quiet hours now have somewhat larger errors (one or two extra trips is a big percentage there), and a few hours have far less demand than predicted, mostly holidays.

**Fix 2: standard errors clustered by day.** They don't need the errors to have the same spread, so the leftover unevenness no longer makes the tests wrong. With 345k rows, the tests also remain reliable even though the errors aren't perfectly normal.

---

## 3. Step 2: Removing predictors that repeat each other (collinearity)

### Why it matters

**Collinearity** means two or more predictors carry the same information. For example, daylight (`is_day`) is almost fully determined by the hour and the month. The model then can't tell which one is having the effect, so their coefficients become unreliable: uncertain, and sometimes with the wrong sign. As the professor requires, this is checked with **F-tests, not VIF**.

### What was done

1. **Correlation** between the numeric predictors.
2. **How much of each predictor the others explain:** regress the predictor on all the other predictors and take the R². Above 0.8 = **flagged** as collinear.
3. **F-tests to decide:**
   - **Strongly correlated pairs** (correlation above 0.7): test the pair together and one at a time. Significant together but not one at a time means they share one piece of information, so the weaker one is dropped.
   - **Flagged predictors:** test each on its own. If it isn't significant, its information is already carried by the other predictors, so it is dropped.

### What was found

| Predictor | Explained by calendar + zone | Explained by all others | Flagged | Own F-test p | Decision |
|---|---|---|---|---|---|
| temperature | 86% | 87% | yes | 0.54 | **drop** |
| daylight (`is_day`) | 82% | 82% | yes | 0.14 | **drop** |
| sunlight | 76% | 83% | yes | 0.03 | keep |
| yesterday's pickups | 72% | 82% | yes | < 0.001 | keep |
| last week's pickups | 72% | 82% | yes | < 0.001 | keep |
| temperature squared | 41% | 50% | no | – | dropped with temperature |
| rain, new snow, snow on the ground, cloud cover | ≤ 41% | ≤ 46% | no | – | keep |

- **Yesterday's and last week's pickups** are strongly correlated (0.88), but each is significant on its own, so both are kept.
- **Temperature and daylight are dropped.** They are almost fully explained by the calendar: temperature follows the season and time of day, and daylight follows the hour and month. Beyond that, they add nothing significant. The squared temperature goes with temperature.

**Note for the report:** dropping temperature means it doesn't help *predict* demand once the calendar is known. It doesn't mean temperature has no effect. That question is answered by the FWL analysis in `05_statistical_tests.ipynb`.

---

## 4. Step 3: Which predictors are significant (all are kept)

### What was done

Each predictor was tested with an F-test: "is its effect zero?" A group of 0/1 columns, such as the 49 zone columns, is tested as a whole.

### Why non-significant predictors are kept

1. **"Not significant" doesn't mean "no effect".** It means the data can't measure the effect precisely enough. Snow is the example: 2025 had few snowy days, so snow's effect is uncertain, even though snow surely matters for taxi demand.
2. **Removing a predictor distorts the others.** Rain and cloud cover go together. Remove cloud cover, and the rain effect partly picks up the effect of clouds.
3. **It makes the remaining results look better than they are.** Predictors that survive a removal process were kept *because* they looked significant, so their p-values come out too small.
4. **The cut-off is arbitrary.** Sunlight has p = 0.06. A slightly different sample could give 0.049 and a different model.

So every predictor stays in, and its significance is reported.

### What was found

| Predictor | p | Significant at 5%? |
|---|---|---|
| last week's pickups, yesterday's pickups, zone, hour, day of week, weekend pattern, zone weekend shift, month | < 0.001 | yes |
| rain | < 0.001 | yes |
| sunlight | 0.06 | no, kept |
| holiday | 0.11 | no, kept |
| cloud cover | 0.18 | no, kept |
| snow on the ground | 0.90 | no, kept |
| new snow | 0.92 | no, kept |

Why the five aren't significant:
- **Snow:** few snowy hours in 2025.
- **Holiday:** only 7 holidays in the training days.
- **Sunlight and cloud cover:** sunlight just misses the 5% cut-off; cloud cover has no detectable effect once rain, the calendar and last week's demand are known.

---

## 5. Step 4: The full model

The full model contains every predictor left after Step 2, significant or not: 14 predictors, **170 parameters** (the intercept, plus all the 0/1 columns for hour, day, month, zone and the weekend terms, plus 8 numeric predictors).

| Fit on training data | Value |
|---|---|
| R² | 0.916 |
| Adjusted R² | 0.916 |
| AIC | 311,874.1 |
| BIC | 313,702.0 |

| Predictor | Effect on pickups | Range (95% confidence) | Significant? |
|---|---|---|---|
| Rain | **+2.2%** per mm in the hour | +1.0% to +3.3% | yes |
| Last week's pickups | 10% more last week → about 5.5% more now | | yes |
| Yesterday's pickups | 10% more yesterday → about 2.6% more now | | yes |
| Holiday | −15.5% | −31% to +4% | no, kept |
| Sunlight | about −0.6% per 100 W/m² | includes 0 | no, kept |
| Cloud cover | about −1.8% from clear to fully overcast | includes 0 | no, kept |
| Snow on the ground | about −0.8% per 10 cm | −12% to +12% | no, kept |
| New snow | −1.1% per cm | −21% to +24% | no, kept |

**How to explain a non-significant result:** give the estimate, the range, and say the data can't rule out zero. For example: "Holidays lower demand by an estimated 15%, but the effect is not statistically significant, because only seven holidays fall in the training data." The snow ranges are very wide because 2025 had so few snowy hours.

**Last week's demand matters more than yesterday's,** because it is the same weekday as well as the same zone and hour.

---

## 6. Step 5: Stepwise selection

### Why

The full model is one model. Stepwise selection builds a **series of models** that adds one predictor at a time. This shows which predictors matter most and gives several models to compare in Step 6.

### What was done

- Start from an empty model.
- At each step, add the predictor that improves **BIC** the most. BIC measures how well the model fits, with a penalty for every extra parameter, so a predictor is only added if it improves the fit enough to be worth it. Each step is saved as a new model (S1, S2, ...).
- Stop when no predictor improves BIC. Removing a predictor was also allowed at every step.
- Separately, start from the full model and remove predictors while that improves BIC (backward).

### What was found

| Model | Adds | BIC |
|---|---|---|
| S1 | last week's pickups | 405,474 |
| S2 | yesterday's pickups | 345,658 |
| S3 | day of week | 342,379 |
| S4 | hour | 339,863 |
| S5 | weekend daily pattern | 327,767 |
| S6 | zone | 318,424 |
| S7 | zone weekend shift | 316,871 |
| S8 | holiday | 315,594 |
| S9 | month | 314,286 |
| S10 | rain | 313,828 |
| S11 | sunlight | 313,782 |
| S12 | cloud cover | 313,680 |

- **What matters most:** past demand comes first, followed by the calendar and zone. The weather barely moves BIC.
- **Stepwise stops at S12:** adding new snow or snow on the ground doesn't improve BIC. **Backward** starts from the full model and removes exactly those two snow terms, ending at the same model.
- **BIC keeps sunlight and cloud cover even though they aren't significant.** That's because BIC counts all 345k rows as independent, so tiny improvements count. The F-tests in Step 3 know that hours on the same day are related, so they are the better guide to significance.

---

## 7. Step 6: Comparing the models

### The measures

| Measure | Data | Meaning | Better |
|---|---|---|---|
| Adjusted R² | train | share of demand explained, with a penalty for extra parameters | higher |
| AIC, BIC | train | fit with a penalty for extra parameters (BIC's penalty is larger) | lower |
| RMSE | test | typical prediction error, in pickups per zone-hour; big misses count more | lower |
| MAE | test | average prediction error, in pickups per zone-hour | lower |

Adjusted R², AIC and BIC measure the fit on the data the model learned from. RMSE and MAE measure predictions on days the model hasn't seen, which is what matters for planning, so **test RMSE picks the model**.

**Baseline:** predict each zone-hour with the pickups at the same hour last week. A model that can't beat this isn't worth using.

### What was found

![Adjusted R², BIC and test RMSE for each stepwise model](img/step6_model_comparison.png)

| Model | Parameters | Adjusted R² | AIC | BIC | RMSE | MAE |
|---|---|---|---|---|---|---|
| S1: last week | 2 | 0.890 | 405,453 | 405,474 | 38.71 | 21.78 |
| S2: + yesterday | 3 | 0.907 | 345,626 | 345,658 | 34.36 | 19.36 |
| S3: + day of week | 9 | 0.908 | 342,283 | 342,379 | 34.89 | 19.61 |
| S4: + hour | 32 | 0.909 | 339,519 | 339,863 | 35.36 | 19.79 |
| S5: + weekend pattern | 55 | 0.912 | 327,176 | 327,767 | 35.27 | 19.80 |
| S6: + zone | 104 | 0.914 | 317,306 | 318,424 | 35.79 | 19.94 |
| S7: + zone weekend shift | 153 | 0.915 | 315,226 | 316,871 | 35.33 | 19.77 |
| S8: + holiday | 154 | 0.915 | 313,938 | 315,594 | 34.41 | 19.39 |
| S9: + month | 165 | 0.916 | 312,512 | 314,286 | 34.13 | 19.25 |
| S10: + rain | 166 | 0.916 | 312,043 | 313,828 | 34.07 | 19.20 |
| S11: + sunlight | 167 | 0.916 | 311,987 | 313,782 | 34.10 | 19.22 |
| S12: + cloud cover (= backward stepwise) | 168 | 0.916 | **311,874** | **313,680** | 34.06 | 19.20 |
| **Full model** | **170** | **0.916** | 311,874 | 313,702 | **34.05** | **19.20** |
| Baseline: last week | – | – | – | – | 40.12 | 22.00 |

- **On the training data:** S12 (the full model without the two snow terms) has the lowest AIC and BIC, only just ahead of the full model. Adjusted R² is 0.916 for both.
- **On the test data, the full model has the lowest error** (RMSE 34.05, MAE 19.20), practically tied with S12 (34.06).
- **What does the work:** the two lags alone (S2) already cut the error from 40.1 (baseline) to 34.4. The calendar, zone and weather predictors add a small further improvement. From S9 onwards the models differ by less than 0.1 pickups.

**Decision: the full model is chosen.** It has the lowest test error, and it is also the model used to explain how each factor (rain, holidays, snow, ...) affects demand, so one model answers both questions.

---

## 8. Step 7: A closer look at the chosen model

| On the test days | RMSE | MAE |
|---|---|---|
| Full model | **34.1** | **19.2** |
| Baseline: last week | 40.1 | 22.0 |

- **Overall:** the model's typical error is **15% smaller** than the baseline's.
- **By zone:** the model beats the baseline in **49 of the 50 zones**.
- **Holidays:** three holidays fell in the random test days (Feb 17, Jul 4, Nov 27).
- **Zone 237 (Upper East Side South), June test days:** the model follows the daily cycle closely. On June 23 it under-predicts the midday peak, which was unusually high that day.

![Zone 237, June test days: actual pickups vs the full model vs last week's pickups](img/step7_test_week.png)

---

## 9. Limitations

- **Demand = completed pickups.** People who couldn't find a taxi aren't counted, so busy hours are probably underestimated.
- **The model assumes it is used one day ahead,** since it needs yesterday's demand.
- **Actual weather is used.** A real system would only have a weather forecast.
- **The test days also choose the model.** With an 80/20 split there is no separate set for choosing, so the chosen model's test error may be slightly optimistic. The top models differ by less than 0.1 pickups, so this doesn't change the conclusion.
- **Temperature and daylight were dropped in Step 2.** The reason is that they repeat the calendar *and* add nothing significant beyond it. If the team prefers, they can be added back and reported as collinear with the calendar.
