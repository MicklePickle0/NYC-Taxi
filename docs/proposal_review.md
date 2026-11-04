# Design rationale and limits

The current workflow is implemented in the single notebook. Earlier proposals and per-technique skeletons are superseded by the decisions below.

| Issue | Current decision |
|---|---|
| Pickup counts do not observe people who could not find a taxi | Call the outcome completed pickups. Discuss supply constraints and unmet demand as limitations. |
| Holding out future months leaves month categories unestimated | Use a chronological Jan–Aug / Sep–Oct / Nov–Dec split and annual Fourier terms. One year remains weak evidence for recurring seasonality. |
| Predictors must exist at forecast time | Exclude actual future weather and same-hour trip summaries from the forecast. Evaluate rolling one hour ahead with a seven-day observed lag. |
| Many panel rows share the same weather | Estimate weather associations on city-hour totals, using train-only calendar controls and clustered/HAC uncertainty. |
| Sequentially subtracting group averages is not FWL | Jointly residualize against the full calendar design and include all seven residualized weather variables together. Verify against the full regression. |
| The full trip CSV is larger than a typical laptop's comfortable pandas input | Parse once into disk-backed DuckDB, aggregate in SQL and materialize only hourly/modeling data. |
| Clock labels do not uniquely identify elapsed time at the autumn DST transition | Combine the repeated label and model pickups per exposure hour. Convert predictions back to observed counts for scoring. |
| A weather coefficient is not a causal estimate | Describe adjusted associations, with uncertainty and limitations. |

Remaining limitations include a single weather location, yellow taxis only, a daily curve shared across zones in the basic ladder, and no direct measure of taxi availability, wait time or local events. The notebook is a reproducible analysis of observed pickups, not an optimized dispatch system.

See [modeling notes](modeling/README.md), [cleaning rules](corrections.md) and [performance notes](performance_review.md) for implementation details. Actual cleaning counts, model rankings and full-data performance must be established by running the real inputs.
