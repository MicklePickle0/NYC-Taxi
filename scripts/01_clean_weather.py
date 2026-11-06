"""Step 1 of 4: clean Open-Meteo hourly weather and align it to NYC clock time.

Usage:
    python scripts/01_clean_weather.py                        # keep precipitation, drop rain (default)
    python scripts/01_clean_weather.py --precip rain          # keep rain instead of precipitation
    python scripts/01_clean_weather.py --precip both          # keep both

Reads the raw Open-Meteo CSV in data/ (never modified). Its name differs between laptops, so the first of
data/weather.csv, data/weather_hourly.csv or data/open-meteo-*.csv that exists is used.
Writes data/processed/weather_hourly.parquet.
Every cut and correction is documented in docs/corrections.md (Weather section).
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "data" / "processed" / "weather_hourly.parquet"


def raw_weather():
    """The raw weather CSV: named differently on different laptops."""
    found = [DATA / "weather.csv", DATA / "weather_hourly.csv", *sorted(DATA.glob("open-meteo-*.csv"))]
    found = [f for f in found if f.exists()]
    if not found:
        raise FileNotFoundError(f"no raw weather CSV in {DATA} (expected weather.csv or open-meteo-*.csv)")
    return found[0]


RAW = raw_weather()

# apparent_temperature: near-duplicate of temperature_2m.
# sunshine_duration: mostly repeats is_day/season; shortwave_radiation and cloud_cover cover brightness.
DROP = ["apparent_temperature", "sunshine_duration"]

# Open-Meteo labels these by the END of the hour they cover; shift back one hour to match the pickup hour.
END_OF_HOUR = ["precipitation", "rain", "snowfall", "shortwave_radiation"]

def total(s):
    return s.sum(min_count=1)  # plain "sum" would turn a missing hour into 0


# How to combine the two real hours that share the 2025-11-02 01:00 label after the DST conversion:
# sum amounts, average instantaneous values and rates (W/m², %).
MERGE = {
    "temperature_2m": "mean", "snow_depth": "mean", "cloud_cover": "mean", "shortwave_radiation": "mean",
    "is_day": "max", "precipitation": total, "rain": total, "snowfall": total,
}


def clean_weather(precip="precipitation"):
    wx = pd.read_csv(RAW, skiprows=3)  # 3 metadata lines before the header
    wx.columns = [c.split(" (")[0] for c in wx.columns]  # drop units, e.g. "temperature_2m (°C)"

    # precipitation = rain + snowfall (as water); rain and snowfall can replace it if precipitation
    # turns out to be collinear with snowfall.
    drop = DROP + {"precipitation": ["rain"], "rain": ["precipitation"], "both": []}[precip]
    wx = wx.drop(columns=drop)

    # Shift while still on Open-Meteo's regular fixed-offset grid (no DST gaps yet).
    # The last hour (2025-12-31 23:00) becomes NaN: its value would sit in 2026-01-01 00:00.
    shift = [c for c in END_OF_HOUR if c in wx]
    wx[shift] = wx[shift].shift(-1)

    # Open-Meteo returned fixed UTC-5 (EST) all year despite timezone=America/New_York,
    # so convert to real NYC clock time to match the taxi pickup timestamps.
    wx["time"] = (pd.to_datetime(wx["time"])
                    .dt.tz_localize("Etc/GMT+5")  # fixed UTC-5 (Etc sign is inverted)
                    .dt.tz_convert("America/New_York")
                    .dt.tz_localize(None))

    # Fall-back hour (2025-11-02 01:00) now appears twice; trips merge it into one label, so do the same.
    wx = wx.groupby("time", as_index=False).agg({c: f for c, f in MERGE.items() if c in wx})

    assert pd.Timestamp("2025-03-09 02:00") not in set(wx["time"])  # spring-forward hour gone
    assert wx["time"].is_unique and len(wx) == 8759
    return wx


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precip", choices=["precipitation", "rain", "both"], default="precipitation",
                    help="which precipitation column(s) to keep (default: precipitation)")
    args = ap.parse_args()

    print(f"reading {RAW.name}")
    wx = clean_weather(args.precip)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.register("wx", wx)
    con.execute(f"COPY wx TO '{OUT.as_posix()}' (FORMAT parquet)")  # DuckDB writes Parquet without pyarrow
    print(f"{len(wx)} hours, columns: {', '.join(wx.columns)}")
    print(f"missing values:\n{wx.isna().sum()[wx.isna().sum() > 0].to_string()}")
    print(f"wrote {OUT}")
