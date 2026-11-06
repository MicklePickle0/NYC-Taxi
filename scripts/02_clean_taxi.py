"""Step 2 of 4: clean the combined 2025 yellow taxi trip data.

Usage:
    python scripts/02_clean_taxi.py

Reads the merged trip CSV in data/ (never modified): data/taxi.csv, or data/combined_data_final.csv
(the name it has on some laptops).
Writes data/processed/taxi_clean.parquet          (kept trips, kept columns)
       data/processed/taxi_cleaning_report.csv    (rows flagged and removed by each rule)
Every removed column and row rule is documented in docs/corrections.md (Taxi section).
"""
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW_NAMES = ["taxi.csv", "combined_data_final.csv"]  # the merged CSV is named differently on different laptops
RAW = next((ROOT / "data" / n for n in RAW_NAMES if (ROOT / "data" / n).exists()), ROOT / "data" / RAW_NAMES[0])
OUT = ROOT / "data" / "processed" / "taxi_clean.parquet"
REPORT = ROOT / "data" / "processed" / "taxi_cleaning_report.csv"
EXPECTED_ROWS = 48_722_602  # TLC total for Jan-Dec 2025 (from the proposal)

# Explicit types so DuckDB doesn't guess from a sample (late-2025 rows have blanks).
TYPES = {
    "VendorID": "INTEGER", "tpep_pickup_datetime": "TIMESTAMP", "tpep_dropoff_datetime": "TIMESTAMP",
    "passenger_count": "DOUBLE", "trip_distance": "DOUBLE", "RatecodeID": "DOUBLE",
    "store_and_fwd_flag": "VARCHAR", "PULocationID": "INTEGER", "DOLocationID": "INTEGER",
    "payment_type": "INTEGER", "fare_amount": "DOUBLE", "extra": "DOUBLE", "mta_tax": "DOUBLE",
    "tip_amount": "DOUBLE", "tolls_amount": "DOUBLE", "improvement_surcharge": "DOUBLE",
    "total_amount": "DOUBLE", "congestion_surcharge": "DOUBLE", "Airport_fee": "DOUBLE",
    "cbd_congestion_fee": "DOUBLE",
}

# Removed columns (store_and_fwd_flag, VendorID, mta_tax, improvement_surcharge, extra, tolls_amount,
# tip_amount, total_amount, congestion_surcharge, cbd_congestion_fee, passenger_count): see docs/corrections.md.
KEEP = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID",
        "trip_distance", "fare_amount", "payment_type", "RatecodeID", "Airport_fee"]

MAX_DURATION_HOURS = 6     # longer trips are meter errors (most are ~24 h meters left running)
MAX_DISTANCE_MILES = 100   # beyond any realistic NYC taxi trip
MAX_SPEED_MPH = 80         # distance / duration above this is physically impossible in NYC

# Each rule is (name, condition). A row is removed by the FIRST rule it fails, so the
# "removed" counts add up to the total; "flagged" counts each rule on its own (rules overlap).
RULES = [
    ("pickup_outside_2025", "tpep_pickup_datetime IS NULL OR tpep_pickup_datetime < TIMESTAMP '2025-01-01' "
                            "OR tpep_pickup_datetime >= TIMESTAMP '2026-01-01'"),
    ("unknown_pickup_zone", "PULocationID IS NULL OR PULocationID NOT BETWEEN 1 AND 263"),  # 264/265 = unknown/outside NYC
    # 3 no charge, 6 voided. 4 dispute: keep the real ride (distance > 0, fare >= 0); its negative
    # rows are reversal copies (89% have a positive twin), so keeping them would double-count.
    ("non_trip_payment", "payment_type IN (3, 6) "
                         "OR (payment_type = 4 AND NOT (trip_distance > 0 AND fare_amount >= 0))"),
    # Refunds and corrections. Exception: Flex Fare (payment_type 0, fixed upfront price) rows with a
    # real distance are standalone trips (no positive twin), so they are kept with fare set to NULL.
    ("negative_fare", "fare_amount < 0 AND NOT (payment_type = 0 AND trip_distance > 0)"),
    ("dropoff_before_pickup", "tpep_dropoff_datetime IS NULL OR duration_s < 0"),
    ("impossible_duration", f"duration_s = 0 OR duration_s > {MAX_DURATION_HOURS * 3600}"),
    # Cancellations. Zero-distance trips lasting >= 1 minute with a positive fare are real rides where the
    # odometer/GPS didn't record miles (or negotiated/flat fares), so they're kept with distance set to NULL.
    ("zero_distance", "(trip_distance IS NULL OR trip_distance <= 0) AND NOT (duration_s >= 60 AND fare_amount > 0)"),
    ("extreme_distance", f"trip_distance > {MAX_DISTANCE_MILES}"),
    ("impossible_speed", f"duration_s > 0 AND trip_distance / (duration_s / 3600.0) > {MAX_SPEED_MPH}"),
    ("exact_duplicate", "dup_rank > 1"),  # identical on all 20 original columns; first copy is kept
]

# Alternative to removing negative fares (not used unless the group decides otherwise):
# treat them as real pickups with a sign error. WARNING: most negative cash/dispute rows are
# reversal copies of a trip already in the data, so this double-counts those trips. Delete the "negative_fare" rule above and
# replace "fare_amount" in KEEP with "abs(fare_amount) AS fare_amount".
# RULES = [r for r in RULES if r[0] != "negative_fare"]
# KEEP = [("abs(fare_amount) AS fare_amount" if c == "fare_amount" else c) for c in KEEP]


def main():
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{Path(tempfile.gettempdir()).as_posix()}/duckdb_taxi'")  # spill to disk if RAM runs out

    types = ", ".join(f"'{c}': '{t}'" for c, t in TYPES.items())
    all_cols = ", ".join(TYPES)
    if not RAW.exists():
        raise FileNotFoundError(f"no trip CSV in {RAW.parent} (expected one of: {', '.join(RAW_NAMES)})")
    print(f"reading {RAW.name}...")
    con.execute(f"""
        CREATE TABLE trips AS
        SELECT {", ".join(c for c in KEEP if c != "RatecodeID")},
               CASE WHEN RatecodeID = 99 THEN NULL ELSE RatecodeID::INTEGER END AS RatecodeID,  -- 99 = unknown
               date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) AS duration_s,
               hash({all_cols}) AS row_hash
        FROM read_csv('{RAW.as_posix()}', header = true, types = {{{types}}})
    """)
    n = con.execute("SELECT count(*) FROM trips").fetchall()[0][0]
    print(f"raw rows: {n:,}" + ("" if n == EXPECTED_ROWS else f"  (WARNING: expected {EXPECTED_ROWS:,})"))

    flags = ", ".join(f"coalesce({cond}, false) AS f_{name}" for name, cond in RULES)
    reason = " ".join(f"WHEN f_{name} THEN '{name}'" for name, _ in RULES)
    print("applying rules...")
    con.execute(f"""
        CREATE TABLE flagged AS
        SELECT *, CASE {reason} END AS reason
        FROM (SELECT *, {flags}
              FROM (SELECT *, row_number() OVER (PARTITION BY row_hash) AS dup_rank FROM trips))
    """)

    report = con.execute(" UNION ALL ".join(
        f"SELECT '{name}' AS rule, count(*) FILTER (f_{name}) AS flagged, "
        f"count(*) FILTER (reason = '{name}') AS removed FROM flagged" for name, _ in RULES
    )).df()
    kept = n - report["removed"].sum()  # computed before the breakdown rows are added
    for code, label in [(3, "no charge"), (4, "dispute"), (6, "voided")]:  # breakdown of non_trip_payment
        r = con.execute(f"SELECT count(*), count(*) FILTER (reason = 'non_trip_payment') FROM flagged "
                        f"WHERE payment_type = {code}").fetchall()[0]
        report.loc[len(report)] = [f"  non_trip_payment: {code} {label}", r[0], r[1]]
    report.loc[len(report)] = ["TOTAL removed", None, report["removed"].iloc[:len(RULES)].sum()]
    report.loc[len(report)] = ["kept", None, kept]
    report["removed_pct"] = (100 * report["removed"] / n).round(3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT, index=False)
    print(report.to_string(index=False))

    print("writing Parquet...")
    # Kept negative fares (Flex Fare) and kept zero distances aren't real values, so leave them missing
    # rather than imputing.
    fixes = {"fare_amount": "CASE WHEN fare_amount >= 0 THEN fare_amount END AS fare_amount",
             "trip_distance": "CASE WHEN trip_distance > 0 THEN trip_distance END AS trip_distance"}
    out_cols = ", ".join(fixes.get(c, c.split(" AS ")[-1]) for c in KEEP)
    con.execute(f"COPY (SELECT {out_cols} FROM flagged WHERE reason IS NULL) TO '{OUT.as_posix()}' (FORMAT parquet)")
    print(f"wrote {OUT}\nwrote {REPORT}")


if __name__ == "__main__":
    main()
