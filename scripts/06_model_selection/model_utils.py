"""Shared setup for the model-selection notebooks in this folder (step1_ ... step7_).

Each notebook starts with:
    import model_utils as mu
    train, test = mu.load()

load() reads data/processed/model_table.parquet, draws the random 80/20 train/test split (whole days, fixed seed),
picks the 50 busiest zones and builds every predictor. Steps pass their results to later steps through small JSON
files in data/processed/model_selection/ (mu.save / mu.read), so a step can be rerun on its own.
"""
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[2]
MODEL_TABLE = ROOT / "data" / "processed" / "model_table.parquet"
RESULTS_DIR = ROOT / "data" / "processed" / "model_selection"  # results passed between steps
FIG_DIR = ROOT / "docs" / "img"                                # plots shown in docs/model_selection.md

TOP_ZONES = 50        # busiest zones by training pickups
TEST_SHARE = 0.2      # 80/20 split, by whole days
ALPHA = 0.05          # significance level
AUX_R2_FLAG = 0.8     # a predictor more than 80% explained by the others is flagged as collinear
PAIR_CORR_FLAG = 0.7  # pairs with a correlation above this are tested together
SEED = 860

LOG_TARGET = "np.log1p(pickups)"
CALENDAR = ["hour", "dow", "month", "holiday"]
CONTINUOUS = ["temperature_2m", "temp_sq", "precipitation", "snowfall", "snow_depth", "cloud_cover",
              "shortwave_radiation", "is_day", "log_lag24", "log_lag168"]
INTERACTIONS = ["hour_x_weekend", "zone_x_weekend"]
ALL_TERMS = CALENDAR + ["zone"] + CONTINUOUS + INTERACTIONS
PARENTS = {"hour_x_weekend": ["hour", "dow"], "zone_x_weekend": ["zone", "dow"], "temp_sq": ["temperature_2m"]}

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 110,
})
pd.set_option("display.width", 160, "display.max_columns", 30, "display.float_format", "{:,.4f}".format)

# Set by load(): the training data (default for ols) and the formula term for each predictor.
TRAIN = None
TERMS = {}
TERM_NAME = {}


def load():
    """Load the data, split it 80/20 by random days, build the predictors. Returns (train, test)."""
    global TRAIN, TERMS, TERM_NAME
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    test_days = pd.DataFrame({"date": np.random.default_rng(SEED).choice(
        days, size=round(TEST_SHARE * len(days)), replace=False)})

    con = duckdb.connect()
    con.register("test_days", test_days)
    df = con.execute(f"""
        WITH m AS (SELECT * FROM '{MODEL_TABLE.as_posix()}'),
        s AS (
            SELECT *, CASE WHEN hour::DATE IN (SELECT date::DATE FROM test_days) THEN 'test'
                           ELSE 'train' END AS split
            FROM m
        ),
        top AS (
            SELECT zone FROM s WHERE split = 'train'
            GROUP BY zone ORDER BY sum(pickups) DESC LIMIT {TOP_ZONES}
        )
        SELECT s.zone, s.hour, s.split, s.pickups,
               hour(s.hour) AS hour_of_day, isodow(s.hour) AS dow, month(s.hour) AS month,
               s.hour::DATE AS date, dayofyear(s.hour) AS day,
               s.temperature_2m, s.precipitation, s.snowfall, s.snow_depth, s.cloud_cover, s.shortwave_radiation,
               s.is_day, l24.pickups AS lag24, l168.pickups AS lag168
        FROM s
        JOIN top USING (zone)
        LEFT JOIN m l24  ON l24.zone  = s.zone AND l24.hour  = s.hour - INTERVAL 24 HOUR
        LEFT JOIN m l168 ON l168.zone = s.zone AND l168.hour = s.hour - INTERVAL 168 HOUR
        ORDER BY s.zone, s.hour
    """).df()

    # Rows with a missing lag or weather: first week of January, hours next to the DST gap, last hour of the year.
    df = df.dropna().reset_index(drop=True)
    df = df.astype({c: "float64" for c in df.columns if str(df[c].dtype) in ("Float64", "Int64")})
    df["date"] = pd.to_datetime(df["date"])

    holidays = USFederalHolidayCalendar().holidays("2025-01-01", "2025-12-31")
    ref_zone = int(df[df["split"] == "train"].groupby("zone")["pickups"].sum().idxmax())
    weekend = df["dow"] >= 6
    df["holiday"] = df["date"].isin(holidays).astype(float)
    df["hour_wknd"] = np.where(weekend, df["hour_of_day"], 0)                     # weekend hour, 0 on weekdays
    df["zone_wknd"] = np.where(weekend & (df["zone"] != ref_zone), df["zone"], 0)  # weekend zone, 0 on weekdays
    df["log_lag24"] = np.log1p(df["lag24"])
    df["log_lag168"] = np.log1p(df["lag168"])
    df["temp_sq"] = (df["temperature_2m"] - df.loc[df["split"] == "train", "temperature_2m"].mean()) ** 2

    TERMS = {"hour": "C(hour_of_day)", "dow": "C(dow)", "month": "C(month)",
             "zone": f"C(zone, Treatment({ref_zone}))", "hour_x_weekend": "C(hour_wknd)",
             "zone_x_weekend": "C(zone_wknd)", **{t: t for t in ["holiday"] + CONTINUOUS}}
    TERM_NAME = {f: t for t, f in TERMS.items()}

    train, test = (df[df["split"] == s].reset_index(drop=True) for s in ["train", "test"])
    TRAIN = train
    print(f"train {len(train):,} rows, test {len(test):,} rows ({len(test) / len(df):.0%} test)")
    return train, test


def formula(terms, target=LOG_TARGET):
    return f"{target} ~ " + (" + ".join(TERMS[t] for t in terms) if terms else "1")


def ols(terms, data=None, target=LOG_TARGET, clustered=False):
    """Linear regression on the training data. clustered=True: standard errors clustered by day (used for tests)."""
    data = TRAIN if data is None else data
    model = smf.ols(formula(terms, target), data)
    if clustered:
        return model.fit(cov_type="cluster", cov_kwds={"groups": data["day"]}, use_t=True)
    return model.fit()


def term_tests(res):
    """F-test per term: are all of the term's coefficients zero?"""
    t = res.wald_test_terms(skip_single=False, scalar=True).summary_frame()
    t = t.drop(index="Intercept").rename(index=TERM_NAME)
    return t.rename(columns={"P>F": "p", "df constraint": "coefficients tested"})[["coefficients tested", "F", "p"]]


def with_hierarchy(terms):
    """Remove a combined term if a term it is built from is missing."""
    terms = list(terms)
    while any(t in PARENTS and not set(PARENTS[t]) <= set(terms) for t in terms):
        terms = [t for t in terms if not (t in PARENTS and not set(PARENTS[t]) <= set(terms))]
    return terms


def diagnostics(axes, fitted, resid, label):
    """Residuals-vs-fitted plot and normal QQ plot on two axes."""
    fitted, resid = pd.Series(np.asarray(fitted)), pd.Series(np.asarray(resid))
    s = fitted.sample(20_000, random_state=SEED).index
    ax = axes[0]
    ax.scatter(fitted[s], resid[s], s=4, alpha=0.25, color=BLUE, linewidths=0)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set(title=f"{label}: residuals vs fitted", xlabel="fitted", ylabel="residual")

    ax = axes[1]
    sm.qqplot(resid.sample(5_000, random_state=SEED), fit=True, line="45", ax=ax,
              marker=".", markersize=3, alpha=0.5, markerfacecolor=BLUE, markeredgecolor=BLUE)
    ax.get_lines()[1].set(color=INK_2, linewidth=1.2, linestyle="--")
    x, y = ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata()
    ax.set(xlim=(x.min() - 0.3, x.max() + 0.3), ylim=(y.min() - 0.3, y.max() + 0.3))  # zoom to the points
    ax.set(title=f"{label}: normal QQ plot", xlabel="theoretical quantile", ylabel="standardised residual")


def save_figure(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")


def save(name, obj):
    """Save a step's result for later steps."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(obj, indent=2))


def read(name, made_by):
    """Read an earlier step's result."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run {made_by} first")
    return json.loads(path.read_text())
