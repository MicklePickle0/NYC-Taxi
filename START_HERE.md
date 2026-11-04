# Start here

## 1. Where to start
Open `scripts/05_statistical_tests.ipynb`. It reads one file, `data/processed/model_table.parquet`. If you don't have that file, build it first (from the repo root):

```
pip install -r requirements.txt
nbstripout --install
python scripts/01_clean_weather.py
python scripts/02_clean_taxi.py
python scripts/03_aggregate_zone_hour.py
python scripts/04_build_model_table.py
```

## 2. Where everything is
| Path | What |
|---|---|
| `scripts/01_` to `04_` | Data pipeline: clean weather, clean trips, group by zone × hour, join weather. |
| `scripts/05_statistical_tests.ipynb` | The analysis. Work here. |
| `scripts/06_model_selection/` | Model building, one notebook per step (`step1_` to `step7_`): assumption checks, collinearity and significance F-tests, full model, stepwise, evaluation. Shared setup in `model_utils.py`. |
| `docs/corrections.md` | Every cleaning cut and why. |
| `docs/model_selection.md` | What the model-selection notebooks do and why, step by step, with results and plots. |
| `CLAUDE.md` | Full notes on the data, the DST fix and the pipeline. |
| `scripts/DONT USE_cleaning.ipynb` | Old version of step 1. Ignore it. |

## 3. Where to put data
Put the two raw files in `data/`:
- `data/taxi.csv` (trips, ~5.3 GB)
- `data/weather.csv` (weather)

The scripts write to `data/processed/`. Never edit the raw files. `data/` is not in git (too big for GitHub), so get the raw files from the team.

## 4. Why everything except 05 is a .py file
Steps 1 to 4 are a fixed pipeline: you run them once, top to bottom, with no plots. As `.py` files they run in one command, give clean diffs in git, and can't break because cells ran out of order. Step 5 is analysis with plots and write-up, which is what notebooks are good for.
