# Single-notebook consolidation

Deliver one useful, fresh-kernel notebook for the complete 2025 taxi workflow, designed for more than 50 million input trips. Remove the obsolete pipeline scripts, notebook skeletons and per-technique guides. Keep the existing pull request as the review location.

The notebook must parse trips once into disk-backed DuckDB, audit sequential cleaning cuts, deduplicate exact original rows, aggregate before pandas conversion and reuse manifest-validated hourly caches. Preserve the full clock grid, align fixed-EST interval weather and account for the repeated autumn hour's two-hour exposure.

Forecasting uses train-selected zones, chronological train/validation/test periods, sparse calendar models, a seven-day lag and count-scale evaluation after training-only retransformation. Actual future weather is excluded. Weather explanation uses train-only city-hour inference, robust diagnostics/tests and jointly residualized FWL. No causal or latent-demand claims.

Acceptance requires focused synthetic checks and a fresh-kernel notebook run, concise current documentation, stripped notebook outputs and an updated existing PR. Full-data benchmark claims require a separate run with the raw files, which are absent here.
