"""Execute the unmodified notebook from a fresh kernel using complete synthetic sources.

This is a correctness smoke test, not a throughput benchmark of the real dataset.
"""
import csv
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd

from test_notebook import NOTEBOOK, RAW_COLUMNS


def write_synthetic_sources(root):
    data = root / "data"
    data.mkdir(parents=True)
    rng = np.random.default_rng(42)
    fixed_hours = pd.date_range("2025-01-01", periods=8760, freq="h")
    wx = pd.DataFrame({
        "time": fixed_hours.strftime("%Y-%m-%dT%H:%M"),
        "temperature_2m (°C)": 10 + 15*np.sin(np.arange(8760)/1400) + rng.normal(0, 2, 8760),
        "precipitation (mm)": rng.exponential(.2, 8760),
        "snowfall (cm)": (rng.random(8760) < .02)*rng.uniform(0, 2, 8760),
        "snow_depth (m)": rng.uniform(0, .02, 8760),
        "is_day ()": ((fixed_hours.hour >= 7) & (fixed_hours.hour < 19)).astype(int),
        "cloud_cover (%)": rng.uniform(0, 100, 8760),
        "shortwave_radiation (W/m²)": rng.uniform(0, 600, 8760),
    })
    path = data / "weather.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("latitude,longitude,elevation,utc_offset_seconds,timezone\n")
        handle.write("40.74,-74.04,51,-18000,GMT-5\n\n")
        wx.to_csv(handle, index=False)
    hours = fixed_hours[fixed_hours != pd.Timestamp("2025-03-09 02:00")]
    with (data / "taxi.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RAW_COLUMNS)
        for hour in hours:
            for zone in (1, 2, 3):
                count = 1 + zone + (hour.hour >= 12) + int(rng.integers(0, 3))
                for trip in range(count):
                    pickup = hour + pd.Timedelta(minutes=trip)
                    writer.writerow([
                        1, pickup, pickup + pd.Timedelta(minutes=10), 1, 2, 1,
                        "N", zone, 2, 1, 10, 0, .5, 2, 0, 1, 16, 2.5, 0, 0,
                    ])


def test_restart_and_run_all_on_synthetic_sources(tmp_path):
    write_synthetic_sources(tmp_path)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    env = dict(os.environ, NYC_TAXI_ROOT=str(tmp_path), NYC_TAXI_TOP_ZONES="3",
               MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "mpl"),
               IPYTHONDIR=str(tmp_path / "ipython"),
               JUPYTER_RUNTIME_DIR=str(tmp_path / "jupyter"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    NotebookClient(notebook, timeout=600, kernel_name="python3").execute(
        cwd=str(NOTEBOOK.parents[1]), env=env,
    )
    tables = tmp_path / "outputs/tables"
    for filename in ["cleaning_report.csv", "forecast_ladder.csv", "validation_scores.csv",
                     "weather_associations.csv", "robust_f_tests.csv", "fwl_check.csv",
                     "final_test_scores.csv", "test_errors_by_zone.csv", "test_predictions.csv"]:
        assert (tables / filename).stat().st_size > 0
    assert np.isfinite(pd.read_csv(tables / "final_test_scores.csv")["MAE"]).all()
    fwl = pd.read_csv(tables / "fwl_check.csv")
    np.testing.assert_allclose(fwl["joint_FWL_original_units"], fwl["full_model_original_units"], atol=1e-8)
    # Cache reuse must not rewrite checkpoints or silently use changed source metadata.
    definitions = "\n".join(c.source for c in notebook.cells
                            if c.cell_type == "code" and "definitions" in c.metadata.get("tags", []))
    import sys
    import types
    module = types.ModuleType("notebook_cache_test")
    sys.modules[module.__name__] = module
    exec(definitions, module.__dict__)
    cfg = module.Config(root=tmp_path, top_zones=3)
    checkpoint = tmp_path / "data/processed/model_sample.parquet"
    original_mtime = checkpoint.stat().st_mtime_ns
    _, _, cached_audit = module.run_pipeline(cfg)
    assert checkpoint.stat().st_mtime_ns == original_mtime
    # Appending an exact duplicate changes signature/raw count but cannot increase pickups.
    raw = tmp_path / "data/taxi.csv"
    with raw.open("rb") as handle:
        handle.readline()
        duplicate = handle.readline()
    with raw.open("ab") as handle:
        handle.write(duplicate)
    _, _, changed_audit = module.run_pipeline(cfg)
    assert changed_audit["raw_rows"] == cached_audit["raw_rows"] + 1
    assert changed_audit["kept_rows"] == cached_audit["kept_rows"]
    assert changed_audit["rules"]["exact_duplicate"] == 1
