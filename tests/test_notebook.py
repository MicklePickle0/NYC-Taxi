"""Behavioral checks for the single runnable NYC taxi analysis notebook.

Only cells tagged ``definitions`` are executed, so these fixtures never read the
large project CSV or fit the final analysis models.
"""

import csv
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "nyc_taxi_analysis.ipynb"
RAW_COLUMNS = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID",
    "payment_type", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
    "improvement_surcharge", "total_amount", "congestion_surcharge", "Airport_fee",
    "cbd_congestion_fee",
]
WEATHER_COLUMNS = {"temperature_2m", "precipitation", "snowfall", "snow_depth", "is_day", "cloud_cover", "shortwave_radiation"}


def test_extra_source_columns_fail_instead_of_being_silently_ignored_by_dedup(api, tmp_path):
    cfg = api.Config(root=tmp_path)
    cfg.raw_taxi.parent.mkdir(parents=True)
    with cfg.raw_taxi.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*RAW_COLUMNS, "new_trip_identifier"])
        writer.writerow([*_trip(), "a"])
        writer.writerow([*_trip(), "b"])
    con = api.connect(cfg)
    try:
        with pytest.raises(ValueError, match="20 original"):
            api.build_trip_aggregates(con, cfg)
    finally:
        con.close()


def _load_notebook_definitions():
    assert NOTEBOOK.is_file(), f"Authoritative analysis notebook is missing: {NOTEBOOK}"
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "definitions" in cell.get("metadata", {}).get("tags", [])
    ]
    assert sources, "The notebook needs executable code cells tagged 'definitions'."
    module = types.ModuleType("nyc_taxi_notebook_test_definitions")
    sys.modules[module.__name__] = module
    exec(compile("\n\n".join(sources), str(NOTEBOOK), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def api():
    return _load_notebook_definitions()


def _trip(**changes):
    row = dict(zip(RAW_COLUMNS, [
        1, "2025-01-08 10:00:00", "2025-01-08 10:10:00", 1,
        2.0, 1, "N", 1, 2, 1, 10.0, 0.0, 0.5, 2.0, 0.0,
        1.0, 16.0, 2.5, 0.0, 0.0,
    ], strict=True))
    row.update(changes)
    return [row[column] for column in RAW_COLUMNS]


def test_nonfinite_dropoff_cannot_bypass_duration_cleaning(api, tmp_path):
    cfg = api.Config(root=tmp_path)
    cfg.raw_taxi.parent.mkdir(parents=True)
    with cfg.raw_taxi.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RAW_COLUMNS)
        writer.writerow(_trip(tpep_dropoff_datetime="infinity"))
        writer.writerow(_trip())
    con = api.connect(cfg)
    try:
        audit = api.build_trip_aggregates(con, cfg)
        assert audit["kept_rows"] == 1
        assert audit["rules"]["dropoff_before_pickup"] == 1
    finally:
        con.close()


@pytest.fixture(scope="module")
def cleaned_taxi(api, tmp_path_factory):
    root = tmp_path_factory.mktemp("taxi_cleaning")
    cfg = api.Config(root=root)
    cfg.raw_taxi.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        _trip(),
        _trip(),  # exact copy across all 20 original columns
        _trip(extra=1.0),  # distinct original trip, even though extra is dropped
        _trip(PULocationID=2, payment_type=0, fare_amount=-5.0,
              tpep_pickup_datetime="2025-01-08 11:00:00",
              tpep_dropoff_datetime="2025-01-08 11:10:00"),
        _trip(PULocationID=3, payment_type=2, fare_amount=-5.0),
        _trip(PULocationID=4, trip_distance=0.0, fare_amount=4.0,
              tpep_pickup_datetime="2025-01-08 12:00:00",
              tpep_dropoff_datetime="2025-01-08 12:01:00"),
        _trip(PULocationID=5, trip_distance=0.0, fare_amount=2.0,
              tpep_pickup_datetime="2025-01-08 13:00:00",
              tpep_dropoff_datetime="2025-01-08 13:00:30"),
        _trip(PULocationID=264),
        _trip(tpep_pickup_datetime="2025-03-09 02:15:00",
              tpep_dropoff_datetime="2025-03-09 02:25:00"),
        _trip(PULocationID=265, fare_amount=-3.0,
              tpep_pickup_datetime="2024-12-31 23:00:00",
              tpep_dropoff_datetime="2024-12-31 23:10:00"),
    ]
    with cfg.raw_taxi.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RAW_COLUMNS)
        writer.writerows(rows)
    raw_before = cfg.raw_taxi.read_bytes()
    con = api.connect(cfg)
    try:
        audit = api.build_trip_aggregates(con, cfg)
        yield con, audit, cfg, raw_before
    finally:
        con.close()


def test_connection_uses_a_persistent_database_and_configured_resources(api, tmp_path):
    cfg = api.Config(root=tmp_path, memory_limit="256MB", threads=2)
    assert cfg.raw_taxi == tmp_path / "data" / "taxi.csv"
    assert cfg.raw_weather == tmp_path / "data" / "weather.csv"
    con = api.connect(cfg)
    try:
        database_files = [row[2] for row in con.execute("PRAGMA database_list").fetchall()]
        assert any(path and Path(path).is_file() for path in database_files)
        assert con.execute("SELECT current_setting('threads')").fetchall()[0][0] == 2
    finally:
        con.close()


def test_trip_cleaning_keeps_valid_exceptions_and_deduplicates_original_rows(cleaned_taxi):
    con, _, _, _ = cleaned_taxi
    rows = con.execute("""
        SELECT zone, hour, pickups, fare_sum, fare_n, distance_sum, distance_n
        FROM observed ORDER BY zone, hour
    """).fetchall()
    assert rows == [
        (1, datetime(2025, 1, 8, 10), 2, 20.0, 2, 4.0, 2),
        (2, datetime(2025, 1, 8, 11), 1, 0.0, 0, 2.0, 1),
        (4, datetime(2025, 1, 8, 12), 1, 4.0, 1, 0.0, 0),
    ]


def test_cleaning_audit_reconciles_overlapping_rules_without_modifying_raw(cleaned_taxi):
    con, audit, cfg, raw_before = cleaned_taxi
    assert audit["raw_rows"] == 10
    assert audit["kept_rows"] == 4
    assert audit["removed_rows"] == 6
    assert audit["raw_rows"] == audit["kept_rows"] + audit["removed_rows"]
    assert isinstance(audit["rules"], dict)
    assert all(isinstance(count, int) and count >= 0 for count in audit["rules"].values())
    assert sum(audit["rules"].values()) == 6
    assert sorted(count for count in audit["rules"].values() if count) == [1, 1, 1, 1, 1, 1]
    assert con.execute("SELECT sum(pickups) FROM observed").fetchall()[0][0] == 4
    assert cfg.raw_taxi.read_bytes() == raw_before


def test_null_values_do_not_satisfy_trip_cleaning_exceptions(api, tmp_path):
    cfg = api.Config(root=tmp_path)
    cfg.raw_taxi.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        _trip(PULocationID=1, payment_type=4, fare_amount=None),
        _trip(PULocationID=2, trip_distance=None, fare_amount=None),
        _trip(PULocationID=3, trip_distance=0.0, fare_amount=None),
        _trip(PULocationID=4, payment_type=None, fare_amount=-5.0),
        # Missing fare does not invalidate an ordinary positive-distance pickup.
        _trip(PULocationID=5, payment_type=1, fare_amount=None),
    ]
    with cfg.raw_taxi.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RAW_COLUMNS)
        writer.writerows(rows)
    con = api.connect(cfg)
    try:
        audit = api.build_trip_aggregates(con, cfg)
        assert con.execute("""
            SELECT zone, pickups, fare_sum, fare_n, distance_sum, distance_n
            FROM observed ORDER BY zone
        """).fetchall() == [(5, 1, 0.0, 0, 2.0, 1)]
        assert audit["raw_rows"] == 5
        assert audit["kept_rows"] == 1
        assert audit["removed_rows"] == 4
        assert audit["rules"]["non_trip_payment"] == 1
        assert audit["rules"]["zero_distance"] == 2
        assert audit["rules"]["negative_fare"] == 1
        assert sum(audit["rules"].values()) == 4
    finally:
        con.close()


@pytest.fixture(scope="module")
def cleaned_weather(api, tmp_path_factory):
    path = tmp_path_factory.mktemp("weather_cleaning") / "open-meteo.csv"
    special = {
        "2025-01-01 01:00": {"precipitation": 0.7, "snowfall": 0.2, "shortwave_radiation": 30.0},
        "2025-03-09 02:00": {"temperature_2m": 12.0, "precipitation": 0.2},
        "2025-03-09 03:00": {"temperature_2m": 13.0, "precipitation": 0.3},
        "2025-11-02 00:00": {"temperature_2m": 10.0, "snow_depth": 2.0,
                              "cloud_cover": 20.0, "is_day": 0},
        "2025-11-02 01:00": {"temperature_2m": 14.0, "snow_depth": 4.0,
                              "cloud_cover": 60.0, "is_day": 1,
                              "precipitation": 1.0, "snowfall": 0.1, "shortwave_radiation": 100.0},
        "2025-11-02 02:00": {"precipitation": 2.0, "snowfall": 0.2, "shortwave_radiation": 300.0},
    }
    header = [
        "time", "temperature_2m (°C)", "precipitation (mm)", "rain (mm)",
        "snowfall (cm)", "snow_depth (m)", "is_day ()", "apparent_temperature (°C)",
        "sunshine_duration (s)", "cloud_cover (%)", "shortwave_radiation (W/m²)",
    ]
    names = [column.split(" (")[0] for column in header[1:]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["latitude", "longitude", "elevation", "utc_offset_seconds", "timezone", "timezone_abbreviation"])
        writer.writerow([40.74, -74.04, 51, -18000, "GMT-5", "GMT-5"])
        writer.writerow([])
        writer.writerow(header)
        for time in pd.date_range("2025-01-01", periods=8760, freq="h"):
            values = dict(zip(names, [10.0, 0.0, 0.0, 0.0, 0.0, 0, 9.0, 0.0, 20.0, 0.0], strict=True))
            values.update(special.get(time.strftime("%Y-%m-%d %H:%M"), {}))
            writer.writerow([time.isoformat(timespec="minutes"), *(values[name] for name in names)])
    raw_before = path.read_bytes()
    result = api.clean_weather(path)
    assert path.read_bytes() == raw_before
    return result.set_index("time")


def test_weather_converts_fixed_est_to_clock_hours_and_shifts_hourly_totals(cleaned_weather):
    weather = cleaned_weather
    assert len(weather) == 8759
    assert weather.index.is_unique
    assert pd.Timestamp("2025-03-09 02:00") not in weather.index
    assert WEATHER_COLUMNS <= set(weather.columns)
    first = weather.loc[pd.Timestamp("2025-01-01 00:00")]
    assert first["precipitation"] == pytest.approx(0.7)
    assert first["snowfall"] == pytest.approx(0.2)
    assert first["shortwave_radiation"] == pytest.approx(30.0)
    spring = weather.loc[pd.Timestamp("2025-03-09 03:00")]
    assert spring["temperature_2m"] == pytest.approx(12.0)
    assert spring["precipitation"] == pytest.approx(0.3)


def test_fall_weather_sums_amounts_but_averages_states_and_radiation(cleaned_weather):
    fall = cleaned_weather.loc[pd.Timestamp("2025-11-02 01:00")]
    assert fall["temperature_2m"] == pytest.approx(12.0)
    assert fall["snow_depth"] == pytest.approx(3.0)
    assert fall["cloud_cover"] == pytest.approx(40.0)
    assert fall["is_day"] == 1
    assert fall["precipitation"] == pytest.approx(3.0)
    assert fall["snowfall"] == pytest.approx(0.3)
    assert fall["shortwave_radiation"] == pytest.approx(200.0)


def test_weather_preserves_the_final_unobserved_shifted_hour_as_missing(cleaned_weather):
    last = cleaned_weather.loc[pd.Timestamp("2025-12-31 23:00")]
    assert last[["precipitation", "snowfall", "shortwave_radiation"]].isna().all()
    assert last[["temperature_2m", "snow_depth", "cloud_cover", "is_day"]].notna().all()
    assert cleaned_weather.iloc[:-1][list(WEATHER_COLUMNS)].notna().all().all()


@pytest.fixture(scope="module")
def model_frames(api, tmp_path_factory):
    root = tmp_path_factory.mktemp("model_panel")
    cfg = api.Config(root=root, top_zones=1)
    con = api.connect(cfg)
    con.execute("""
        CREATE TABLE observed (
            zone INTEGER, hour TIMESTAMP, pickups BIGINT, fare_sum DOUBLE,
            fare_n BIGINT, distance_sum DOUBLE, distance_n BIGINT
        )
    """)
    observations = [
        (1, "2025-01-01 00:00", 10),
        (2, "2025-01-01 00:00", 4),
        (263, "2025-01-01 00:00", 2),
        (1, "2025-01-08 00:00", 3),
        (1, "2025-03-02 02:00", 7),
        (1, "2025-03-09 03:00", 11),
        (1, "2025-03-16 02:00", 5),
        (1, "2025-03-16 03:00", 6),
        (1, "2025-08-25 00:00", 13),
        (1, "2025-09-01 00:00", 3),
        (2, "2025-09-01 00:00", 1000),
        (1, "2025-10-25 00:00", 17),
        (1, "2025-11-01 00:00", 4),
        (263, "2025-11-01 00:00", 2000),
    ]
    con.executemany("INSERT INTO observed VALUES (?, ?, ?, ?, ?, ?, ?)", [
        (zone, hour, pickups, pickups * 10.0, pickups, pickups * 2.0, pickups)
        for zone, hour, pickups in observations
    ])
    hours = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
    hours = hours[hours != pd.Timestamp("2025-03-09 02:00")]
    weather = pd.DataFrame({
        "time": hours, "temperature_2m": 10.0, "precipitation": 0.0, "snowfall": 0.0,
        "snow_depth": 0.0, "is_day": 1, "cloud_cover": 20.0, "shortwave_radiation": 100.0,
    })
    weather.loc[weather["time"] == pd.Timestamp("2025-12-31 23:00"), ["precipitation", "snowfall", "shortwave_radiation"]] = np.nan
    try:
        panel, city = api.prepare_panel(con, cfg, weather)
        yield panel, city, con
    finally:
        con.close()


def test_panel_zero_fills_hours_and_ranks_zones_using_training_demand_only(model_frames):
    panel, city, con = model_frames
    assert con.execute("SELECT count(*), count(DISTINCT zone), sum(pickups) FROM zone_hour").fetchall() == [(2_303_617, 263, 3085)]
    assert con.execute("SELECT count(*) FROM zone_hour WHERE hour = TIMESTAMP '2025-12-31 23:00'").fetchall() == [(263,)]
    assert set(panel["zone"]) == {1}
    assert len(panel) == 8758
    assert len(city) == 8758
    assert not panel.duplicated(["zone", "hour"]).any()
    assert city["hour"].is_unique
    assert panel["hour"].min() == pd.Timestamp("2025-01-01 00:00")
    assert panel["hour"].max() == pd.Timestamp("2025-12-31 22:00")
    assert pd.Timestamp("2025-03-09 02:00") not in set(panel["hour"])
    assert panel.loc[panel["hour"] == pd.Timestamp("2025-01-01 01:00"), "pickups"].item() == 0
    city_by_hour = city.set_index("hour")
    assert city_by_hour.loc[pd.Timestamp("2025-01-01 00:00"), "city_pickups"] == 16
    assert city_by_hour.loc[pd.Timestamp("2025-01-01 01:00"), "city_pickups"] == 0
    assert city_by_hour.loc[pd.Timestamp("2025-09-01 00:00"), "city_pickups"] == 1003
    assert city_by_hour.loc[pd.Timestamp("2025-11-01 00:00"), "city_pickups"] == 2004
    assert WEATHER_COLUMNS <= set(panel.columns)
    assert WEATHER_COLUMNS <= set(city.columns)
    assert panel[list(WEATHER_COLUMNS)].notna().all().all()
    assert city[list(WEATHER_COLUMNS)].notna().all().all()


def test_seven_day_baseline_joins_clock_labels_before_the_chronological_split(model_frames):
    panel, _, _ = model_frames
    by_hour = panel.set_index("hour")
    assert pd.isna(by_hour.loc[pd.Timestamp("2025-01-01 00:00"), "base_lag168"])
    assert by_hour.loc[pd.Timestamp("2025-01-08 00:00"), "base_lag168"] == 10
    assert pd.isna(by_hour.loc[pd.Timestamp("2025-03-16 02:00"), "base_lag168"])
    assert by_hour.loc[pd.Timestamp("2025-03-16 03:00"), "base_lag168"] == 11
    assert by_hour.loc[pd.Timestamp("2025-09-01 00:00"), "base_lag168"] == 13
    assert by_hour.loc[pd.Timestamp("2025-11-01 00:00"), "base_lag168"] == 17
    assert set(panel.loc[panel["base_lag168"].isna(), "split"]) == {"train"}


def test_splits_and_calendar_fields_follow_the_documented_clock_conventions(model_frames):
    panel, city, _ = model_frames
    for frame in (panel, city):
        assert set(frame["split"]) == {"train", "val", "test"}
        by_hour = frame.set_index("hour")
        assert by_hour.loc[pd.Timestamp("2025-08-31 23:00"), "split"] == "train"
        assert by_hour.loc[pd.Timestamp("2025-09-01 00:00"), "split"] == "val"
        assert by_hour.loc[pd.Timestamp("2025-10-31 23:00"), "split"] == "val"
        assert by_hour.loc[pd.Timestamp("2025-11-01 00:00"), "split"] == "test"
        assert frame.loc[frame["split"] == "train", "hour"].max() < frame.loc[frame["split"] == "val", "hour"].min()
        assert frame.loc[frame["split"] == "val", "hour"].max() < frame.loc[frame["split"] == "test", "hour"].min()
        wednesday = by_hour.loc[pd.Timestamp("2025-01-01 00:00")]
        assert (wednesday["hod"], wednesday["dow"], wednesday["month"], wednesday["weekend"], wednesday["wknd_hod"]) == (0, 3, 1, 0, 0)
        sunday = by_hour.loc[pd.Timestamp("2025-03-16 03:00")]
        assert (sunday["hod"], sunday["dow"], sunday["month"], sunday["weekend"], sunday["wknd_hod"]) == (3, 7, 3, 1, 3)
        assert pd.Timestamp(sunday["date"]).date().isoformat() == "2025-03-16"


def test_sparse_ols_matches_an_independent_dense_least_squares_solution(api):
    dense = np.array([
        [1.0, -2.0, 1.0],
        [1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 1.0],
        [1.0, 3.0, 0.0],
    ])
    target = np.array([1.0, 0.5, 2.5, 2.0, 4.0, 2.8])
    expected_beta = np.linalg.lstsq(dense, target, rcond=None)[0]
    expected_residual = target - dense @ expected_beta
    fit = api.sparse_ols(sparse.csr_matrix(dense), target)
    np.testing.assert_allclose(fit["beta"], expected_beta, atol=1e-8, rtol=1e-8)
    assert fit["rss"] == pytest.approx(expected_residual @ expected_residual)
    assert fit["smear"] == pytest.approx(np.exp(expected_residual).mean())
    assert fit["iterations"] > 0


def test_forecast_design_scales_lag_rates_using_training_rows_only(api):
    frame = pd.DataFrame({
        "hour": pd.date_range("2025-01-01", periods=6, freq="h"),
        "hod": [0, 1, 2, 3, 4, 5], "dow": [3, 3, 3, 3, 3, 3],
        "zone": [1, 2, 1, 2, 1, 2], "wknd_hod": [0, 0, 0, 0, 0, 0],
        "base_lag168": [0, 1, 3, 14, 15, 31],
        "lag_exposure": [1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
    })
    training = np.array([True, True, True, True, False, False])
    design, names, blocks = api.forecast_design(frame, training)
    assert sparse.isspmatrix_csr(design)
    assert design.shape == (6, len(names))
    # Training log rates are 0, log(2), log(4), log(8); the held-out
    # log(16), log(32) values must not alter their mean or population SD.
    np.testing.assert_allclose(
        design[:, names.index("log_lag7_scaled")].toarray().ravel(),
        np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]) / np.sqrt(5.0),
        atol=1e-10,
    )
    assert blocks["M0"] == [names.index("Intercept")]
    assert set(blocks["M5"]) < set(blocks["M6"])
    assert names.index("log_lag7_scaled") in blocks["M6"]


def test_backtransform_applies_smearing_then_restores_hour_exposure(api):
    predictions = api.to_pickups(
        np.log(np.array([3.0, 3.0, 0.5])),
        smear=1.5,
        exposure=np.array([1.0, 2.0, 2.0]),
    )
    np.testing.assert_allclose(predictions, [3.5, 7.0, 0.0])


def test_metrics_keep_zero_demand_hours_and_leave_zero_total_wape_undefined(api):
    result = api.metrics([0, 0, 4], [0, 2, 2])
    assert result["MAE"] == pytest.approx(4.0 / 3.0)
    assert result["RMSE"] == pytest.approx(np.sqrt(8.0 / 3.0))
    assert result["WAPE"] == pytest.approx(1.0)
    assert result["bias"] == pytest.approx(0.0)
    zero = api.metrics([0, 0], [1, 3])
    assert zero["MAE"] == pytest.approx(2.0)
    assert zero["RMSE"] == pytest.approx(np.sqrt(5.0))
    assert np.isnan(zero["WAPE"])
    assert zero["bias"] == pytest.approx(2.0)
