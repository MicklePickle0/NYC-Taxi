# Cleaning and time alignment

The single analysis notebook implements these rules and writes a cleaning report on each rebuild. Raw files are never changed. Historical full-file counts from an earlier pipeline are not current evidence: this checkout has no raw inputs, and actual counts must come from a new run.

## Trips

The CSV is parsed with an explicit schema covering all 20 original columns. Invalid schema or unparseable values should fail visibly rather than silently disappear. All original columns remain available through exact duplicate removal, even if they are not forecast features.

Rules apply in this order. A row contributes to the sequential removal count only for its first failed rule. The report reconciles raw rows with retained rows plus these mutually exclusive removals.

| Order | Remove when | Reason or exception |
|---|---|---|
| 1 | Pickup is missing or outside 2025 | The analysis needs a valid pickup clock label. |
| 2 | Pickup zone is missing or outside 1–263 | Unknown/outside zones cannot enter the fixed zone grid. |
| 3 | Pickup is labeled March 9, 02:00 | That New York clock hour did not exist in 2025. |
| 4 | A nonmissing fare or distance is nonfinite | Reject NaN/infinite numerical measures. |
| 5 | Payment is no-charge (3), voided (6), or an unsupported dispute (4) | Keep disputed trips only when distance is positive and fare is nonnegative. |
| 6 | Fare is negative | Keep Flex Fare (`payment_type = 0`) when distance is positive. |
| 7 | Dropoff is missing/nonfinite, duration cannot be computed, or dropoff precedes pickup | The time ordering is invalid. |
| 8 | Duration is zero or over six hours | Exclude invalid or implausibly long meter records. |
| 9 | Distance is missing or nonpositive | Keep trips lasting at least one minute with a positive fare. |
| 10 | Distance exceeds 100 miles | Exclude implausible distance records. |
| 11 | Distance divided by duration exceeds 80 mph | Exclude implausible speeds. |
| 12 | All 20 original typed columns equal another surviving row | Keep one exact copy. Deduplication does not rely on hash uniqueness. |

Flex Fare, zero fares and unknown dropoff zones do not independently invalidate a pickup. Do not impute missing passengers, rate codes or other optional fields to manufacture trips. Completed pickups are the outcome; fare and distance are cleaning checks, not same-hour forecast inputs.

An exception must be supported by known values: a disputed trip needs a known nonnegative fare and positive distance; a zero-distance exception needs a known positive fare; a negative-fare exception needs payment type 0. Ordinary positive-distance pickups can retain a missing fare. SQL nulls do not count as satisfying an exception.

The trip stage is aggregated in SQL and dropped after the observed hourly aggregates are built. No full cleaned trip Parquet or full trip dataframe is required.

## Weather and clock hours

The default weather file has three metadata lines followed by its column header and fixed UTC−5 (EST) time labels. The notebook converts those labels to New York clock time. This is a source-specific assumption; use the expected file or inspect and update the configuration before substituting another export.

Precipitation, snowfall and shortwave radiation describe the preceding interval. Shift those values back one elapsed hour while the time axis is unambiguous, then aggregate to the clock labels used for trips. Temperature, snow depth, cloud cover and the daylight indicator are instantaneous and are not shifted.

| Situation in 2025 | Treatment |
|---|---|
| March 9, 02:00 | Exclude this nonexistent New York clock hour. |
| November 2, 01:00 | Combine the two elapsed hours sharing that label. Sum precipitation/snow amounts; average temperature, snow depth, cloud cover and shortwave radiation; take the maximum daylight indicator. |
| December 31, 23:00 | The source lacks the following endpoint needed for shifted interval values. Keep the missingness visible and exclude this label from the weather-complete modeling sample. |

The repeated autumn label contains two hours of exposure: `exposure_hours = 2` there and 1 elsewhere. Forecast targets and the city explanatory target use pickups per exposure hour. Predicted rates and the weekly baseline are converted to counts using the current exposure before scoring. A clock-time aggregation cannot reconstruct which repeated hour a taxi timestamp belongs to.

The full 2025 pickup grid contains 263 × 8,759 = **2,303,617 rows**, including explicit zero-pickup cells. The weather-complete grid contains 8,758 city clock labels; lag-dependent models may use fewer rows when history is unavailable. Calendar counts are independent of the actual number of raw trips.

The seven explanatory weather predictors are temperature, precipitation, snowfall, snow depth, daylight, cloud cover and shortwave radiation. Apparent temperature, sunshine duration and rain are omitted to reduce redundant measures. Missing weather is not imputed.
