# Implementation plan

1. Build the single notebook's configuration, disk-backed typed staging pipeline, cleaning audit, exact deduplication, complete hourly grid, weather alignment and cache manifest.
2. Add train-only zone selection and hourly modeling samples, exposure-aware lags, sparse model ladder, validation selection and final test evaluation.
3. Add city-hour explanatory inference, residual diagnostics, robust weather tests, auxiliary regressions and the joint FWL check; export report artifacts.
4. Remove superseded scripts/notebooks/guides and update the entry documentation, data rules, modeling notes and performance limits.
5. Run meaningful synthetic pipeline checks and fresh-kernel execution, inspect the resulting notebook and repository diff, strip outputs, then commit/push to the existing PR branch.

Do not claim the raw dataset was processed or benchmarked unless a real-data run supplies evidence.
