"""Ridge baseline: walk-forward, every feature variation, all three stocks.

Setup
    label        next trading day's return (close to close); its sign gives up/down
    rows         one per trading day where every pool feature exists, so all variations
                 are scored on exactly the same days for a given stock
    walk-forward expanding window, retrained once a year; test years 2015 through Oct 2026
    each fit     (all learned from that year's training rows only)
                 1. clip each feature to its 1st-99th percentile (a few days like April 2020 oil are extreme)
                 2. standardize (Ridge's penalty treats every coefficient the same, so units matter)
                 3. clip extreme training labels at 3 standard deviations, so a few huge days don't dominate
                 4. pick the penalty strength (alpha) on the last year of the training window,
                    then refit on the whole window
    baselines    always guess "up" (direction), predict the training-window average return (size)
    periods      2015-2019 is where we compare variations; 2020-2026 is the check on the winners

Outputs
    results/ridge_baseline_results.csv      one row per stock x variation x period
    results/ridge_baseline_predictions.csv  every test-day prediction (for later analysis)

Run from the repo root: python src/ridge_baseline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import MACRO_CATALOG, MODEL_START, STOCK_CATALOG, load_table, macro_features, stock_features

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
STOCKS = ["BRK-B", "PEP", "TSLA"]
TEST_YEARS = range(2015, 2027)
ALPHAS = np.logspace(-2, 7, 19)  # up to 1e7: at the top, every coefficient is shrunk to about zero (= predict the average)
VALIDATION_DAYS = 252

STOCK_POOL = [k for k, v in STOCK_CATALOG.items() if v[1] == "pool"]
MACRO_POOL = [k for k, v in MACRO_CATALOG.items() if v[1] == "pool"]

CORE = ["gap_overnight", "ret_days2to5", "vol_21d", "volume_vs_20d", "days_to_next_session", "report_before_next_close"]
MACRO_MARKET = ["vix", "vix_chg_5d", "vix_minus_realized", "sp500_ret_1d", "sp500_ret_21d", "baa_spread_chg_21d"]
MACRO_RATES_ECON = ["yield10_chg_21d", "curve_slope", "dollar_chg_21d", "oil_chg_21d", "claims_chg_4w"]

# name: (features, what it tests)
VARIATIONS = {
    "V1 price moves (5)":       (["gap_overnight", "ret_intraday", "ret_days2to5", "ret_days6to21", "ret_days22to63"],
                                 "Do price moves alone say anything?"),
    "V2 risk and position (5)": (["vol_21d", "vol_5d_vs_63d", "range_today", "from_52w_high", "corr_63d"],
                                 "Does risk and position say anything?"),
    "V3 core (6)":              (CORE, "Most independent feature of each type"),
    "V4 core, attention instead of volume (6)":
                                ([f if f != "volume_vs_20d" else "attention" for f in CORE], "Volume vs attention head to head"),
    "V5 core, quarter trend instead of reversal (6)":
                                ([f if f != "ret_days2to5" else "ret_days22to63" for f in CORE], "Short reversal vs quarter trend"),
    "V6 core + attention (7)":  (CORE + ["attention"], "Do volume and attention add up?"),
    "V7 all stock features (17)": (STOCK_POOL, "Does Ridge's penalty cope with everything?"),
    "M1 core + market macro (12)": (CORE + MACRO_MARKET, "Does market context add to the core?"),
    "M2 core + rates and economy (11)": (CORE + MACRO_RATES_ECON, "Does the rates/economy context add?"),
    "M3 core + all macro (17)": (CORE + MACRO_POOL, "Does all macro add?"),
}


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to percentiles learned on the training rows."""

    def __init__(self, lower=1, upper=99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        self.lo_ = np.nanpercentile(X, self.lower, axis=0)
        self.hi_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


def make_model(alpha):
    return Pipeline([("clip", PercentileClipper()), ("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def clip_labels(y):
    s = y.std()
    return y.clip(y.mean() - 3 * s, y.mean() + 3 * s)


def fit_one_year(X_train, y_train):
    """Pick alpha on the last year of the training window, then refit on all of it."""
    X_in, y_in = X_train[:-VALIDATION_DAYS], y_train[:-VALIDATION_DAYS]
    X_val, y_val = X_train[-VALIDATION_DAYS:], y_train[-VALIDATION_DAYS:]
    errors = []
    for a in ALPHAS:
        m = make_model(a).fit(X_in, clip_labels(y_in))
        errors.append(np.mean((m.predict(X_val) - y_val) ** 2))
    best = ALPHAS[int(np.argmin(errors))]
    return make_model(best).fit(X_train, clip_labels(y_train)), best


def dataset(ticker, macro):
    table = load_table(ticker)
    f = stock_features(ticker).merge(macro, on="date", how="left")
    f["ret_next"] = table["ret_next"].to_numpy()
    f = f.replace([np.inf, -np.inf], np.nan)
    needed = STOCK_POOL + MACRO_POOL + ["ret_next"]
    keep = (f["date"] >= MODEL_START[ticker]) & f[needed].notna().all(axis=1)
    return f.loc[keep].reset_index(drop=True)


def walk_forward(data, cols):
    rows = []
    for year in TEST_YEARS:
        train = data[data["date"] < f"{year}-01-01"]
        test = data[data["date"].dt.year == year]
        if test.empty:
            continue
        model, alpha = fit_one_year(train[cols].to_numpy(), train["ret_next"])
        pred = model.predict(test[cols].to_numpy())
        rows.append(pd.DataFrame({
            "date": test["date"].to_numpy(), "actual": test["ret_next"].to_numpy(), "pred": pred,
            "train_mean": train["ret_next"].mean(), "alpha": alpha,
        }))
    return pd.concat(rows, ignore_index=True)


def score(p):
    up = p["actual"] > 0
    n = len(p)
    acc = np.mean((p["pred"] > 0) == up)
    base = np.mean(up)  # always guess "up"
    return {
        "days": n,
        "accuracy": acc,
        "always_up": base,
        "edge_pts": 100 * (acc - base),
        "margin_pts": 100 * 1.96 * np.sqrt(base * (1 - base) / n),  # rough 95% noise band for a hit rate
        "pred_up_share": np.mean(p["pred"] > 0),
        "mae": np.mean(np.abs(p["pred"] - p["actual"])),
        "mae_baseline": np.mean(np.abs(p["train_mean"] - p["actual"])),
        "median_alpha": float(np.median(p["alpha"])),
    }


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    macro = macro_features()
    results, preds = [], []
    for ticker in STOCKS:
        data = dataset(ticker, macro)
        print(f"{ticker}: {len(data)} rows, {data['date'].min().date()} to {data['date'].max().date()}")
        for name, (cols, question) in VARIATIONS.items():
            p = walk_forward(data, cols)
            p.insert(0, "variation", name)
            p.insert(0, "ticker", ticker)
            preds.append(p)
            for period, mask in [("2015-2026", p["date"].dt.year >= 2015),
                                 ("2015-2019 (compare)", p["date"].dt.year <= 2019),
                                 ("2020-2026 (check)", p["date"].dt.year >= 2020)]:
                results.append({"ticker": ticker, "variation": name, "period": period, "question": question,
                                **score(p[mask])})
    res = pd.DataFrame(results)
    res.to_csv(RESULTS / "ridge_baseline_results.csv", index=False)
    pd.concat(preds).to_csv(RESULTS / "ridge_baseline_predictions.csv", index=False)

    pd.set_option("display.width", 220)
    show = res[res["period"] == "2015-2019 (compare)"][
        ["ticker", "variation", "days", "accuracy", "always_up", "edge_pts", "margin_pts", "pred_up_share", "mae", "mae_baseline", "median_alpha"]]
    print("\n2015-2019, where variations get compared:")
    print(show.round(4).to_string(index=False))
