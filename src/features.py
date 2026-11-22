"""Feature catalog and builders.

Every feature is grouped by what it's meant to capture, so features that overlap sit
in the same group. Each one also has a role:

    pool  in the feature pool the experiment variations draw from
    held  a near-duplicate of a pool feature; held out, added back only if testing says so
    new   a candidate alternative, not in the pool yet

Every value on a row uses only data known by that day's close.
Build the modeling tables first: python src/build_modeling_table.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "data" / "processed" / "modeling"
SUBMISSIONS = ROOT / "data" / "raw" / "sec" / "submissions"

# name: (group, role, what it is)
STOCK_CATALOG = {
    # Recent move: which way the stock just went
    "gap_overnight":        ("Recent move", "pool", "Yesterday's close to today's open"),
    "ret_intraday":         ("Recent move", "pool", "Today's open to today's close"),
    "ret_days2to5":         ("Recent move", "pool", "Return over days 2-5 back (short reversal)"),
    "ret_1d":               ("Recent move", "held", "Today's full return (= gap + intraday)"),
    "close_in_range":       ("Recent move", "held", "Where the close sits in today's high-low range"),
    # Trend and position: the longer direction and where price sits.
    # rsi_14, up_volume_share_20d, downside_share_63d and vs_sp500_21d were first filed under other
    # groups, but measured overlap shows they mostly capture trend, so they live here.
    "ret_days6to21":        ("Trend and position", "pool", "Return over days 6-21 back"),
    "ret_days22to63":       ("Trend and position", "pool", "Return over days 22-63 back"),
    "from_52w_high":        ("Trend and position", "pool", "Distance below the 52-week high"),
    "efficiency_21d":       ("Trend and position", "pool", "21-day net move / sum of daily moves (1 = straight line)"),
    "vs_sp500_21d":         ("Trend and position", "held", "21-day return minus the S&P 500's (redundant for Tesla)"),
    "dist_ma50":            ("Trend and position", "held", "Distance from the 50-day average price"),
    "rsi_14":               ("Trend and position", "held", "14-day relative strength index"),
    "up_volume_share_20d":  ("Trend and position", "held", "Share of 20-day volume traded on up days"),
    "downside_share_63d":   ("Trend and position", "held", "Share of 63-day volatility that came from down days"),
    # Volatility: how big moves are
    "vol_21d":              ("Volatility", "pool", "Std of daily returns, last 21 days"),
    "vol_5d_vs_63d":        ("Volatility", "pool", "5-day vol / 63-day vol (building up or calming)"),
    "range_today":          ("Volatility", "pool", "Today's high-low range / close"),
    "parkinson_vol_21d":    ("Volatility", "held", "21-day vol estimated from high-low ranges"),
    # Attention and activity: how much people are trading and watching
    "volume_vs_20d":        ("Attention", "pool", "Today's volume / prior 20-day average"),
    "attention":            ("Attention", "pool", "5-day net move z-score + earnings proximity"),
    # Market relationship: how tightly the stock moves with the S&P 500
    "corr_63d":             ("Market link", "pool", "63-day correlation with the S&P 500"),
    "beta_63d":             ("Market link", "held", "63-day beta to the S&P 500"),
    # Calendar and events: scheduled things
    "days_to_next_session": ("Calendar and events", "pool", "Calendar days until the next session (weekend gap)"),
    "report_before_next_close": ("Calendar and events", "pool", "1 if a report lands before tomorrow's close"),
    "days_since_report":    ("Calendar and events", "pool", "Trading days since the last report"),
    "last_report_reaction": ("Calendar and events", "pool", "Last report-day return / normal volatility (post-earnings drift)"),
    "is_friday":            ("Calendar and events", "held", "1 on Fridays"),
    "options_expiry":       ("Calendar and events", "new",  "1 on monthly options expiration (3rd Friday)"),
    "turn_of_month":        ("Calendar and events", "new",  "1 on the last trading day and first 3 of a month"),
}

MACRO_CATALOG = {
    "vix":                  ("Fear and volatility", "pool", "VIX close"),
    "vix_chg_5d":           ("Fear and volatility", "pool", "VIX change over 5 days"),
    "sp500_vol_21d":        ("Fear and volatility", "held", "S&P 500 realized volatility, 21 days"),
    "vix_minus_realized":   ("Fear and volatility", "pool", "VIX minus S&P realized vol (fear beyond what's happening)"),
    "sp500_from_52w_high":  ("Fear and volatility", "held", "S&P 500 distance below its 52-week high (moves with VIX)"),
    "sp500_ret_1d":         ("Market direction", "pool", "S&P 500 return today"),
    "sp500_ret_21d":        ("Market direction", "pool", "S&P 500 return, 21 days"),
    "yield10_chg_21d":      ("Rates", "pool", "10-year yield change, 21 days"),
    "curve_slope":          ("Rates", "pool", "10-year minus 2-year yield"),
    "yield2_chg_21d":       ("Rates", "held", "2-year yield change, 21 days"),
    "real_yield_chg_21d":   ("Rates", "held", "10-year real yield change, 21 days"),
    "slope_chg_21d":        ("Rates", "new",  "Curve slope change, 21 days"),
    "inflation_exp_chg_21d": ("Rates", "new", "10-year breakeven inflation change, 21 days"),
    "baa_spread_chg_21d":   ("Credit", "pool", "Baa spread change, 21 days"),
    "baa_spread":           ("Credit", "held", "Baa spread level"),
    "dollar_chg_21d":       ("Economy and costs", "pool", "Dollar index change, 21 days"),
    "oil_chg_21d":          ("Economy and costs", "pool", "Oil $ change over 21 days / 63-day average price"),
    "claims_chg_4w":        ("Economy and costs", "pool", "Jobless claims change, 4 weeks"),
}

MODEL_START = {"BRK-B": "2009-01-01", "PEP": "2009-01-01", "TSLA": "2010-10-01"}


def load_table(ticker):
    return pd.read_csv(TABLES / f"{ticker}.csv", parse_dates=["date"])


def report_times(ticker):
    """Earnings-type filings (8-K item 2.02, 10-Q, 10-K), as New York acceptance times."""
    blocks = [json.load(open(SUBMISSIONS / f"{ticker}.json"))["filings"]["recent"]]
    blocks += [json.load(open(p)) for p in sorted(glob.glob(str(SUBMISSIONS / f"{ticker}-*.json")))]
    times = []
    for b in blocks:
        items = b.get("items", [""] * len(b["form"]))
        for form, accepted, item in zip(b["form"], b["acceptanceDateTime"], items):
            if form in ("10-Q", "10-K") or (form == "8-K" and "2.02" in (item or "")):
                times.append(pd.Timestamp(accepted).tz_convert("America/New_York").tz_localize(None))
    return sorted(set(times))


def reaction_days(dates, ticker):
    """Row index of each report's reaction day: the first 4pm close at or after the filing."""
    closes = dates + pd.Timedelta(hours=16)
    idx = sorted({int(closes.searchsorted(t)) for t in report_times(ticker)})
    return [i for i in idx if i < len(dates)]


def stock_features(ticker):
    d = load_table(ticker)
    c, o, h, l, v = d.adj_close, d.adj_open, d.adj_high, d.adj_low, d.volume
    r = c.pct_change()
    lr = np.log(c).diff()
    sig63 = r.rolling(63).std()
    m = d.sp500_close.pct_change()
    f = pd.DataFrame(index=d.index)

    f["gap_overnight"] = o / c.shift(1) - 1
    f["ret_intraday"] = c / o - 1
    f["ret_days2to5"] = c.shift(1) / c.shift(5) - 1
    f["ret_1d"] = r
    f["close_in_range"] = ((c - l) / (h - l)).where(h > l)
    gain, loss = r.clip(lower=0).rolling(14).mean(), (-r.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss)

    f["ret_days6to21"] = c.shift(5) / c.shift(21) - 1
    f["ret_days22to63"] = c.shift(21) / c.shift(63) - 1
    f["from_52w_high"] = c / c.rolling(252).max() - 1
    f["dist_ma50"] = c / c.rolling(50).mean() - 1
    f["efficiency_21d"] = (np.log(c / c.shift(21))).abs() / lr.abs().rolling(21).sum()

    f["vol_21d"] = r.rolling(21).std()
    f["vol_5d_vs_63d"] = r.rolling(5).std() / sig63
    f["range_today"] = (h - l) / c
    f["parkinson_vol_21d"] = np.sqrt((np.log(h / l) ** 2).rolling(21).mean() / (4 * np.log(2)))
    f["downside_share_63d"] = np.sqrt((r.clip(upper=0) ** 2).rolling(63).mean()) / np.sqrt((r ** 2).rolling(63).mean())

    # Reports: trading days until the next reaction day, and since the last one
    react = reaction_days(d.date, ticker)
    nxt = pd.Series(np.nan, index=d.index)
    last = pd.Series(np.nan, index=d.index)
    for i in react:
        nxt.loc[:i] = nxt.loc[:i].fillna(i)          # earlier rows: this is their next reaction day
        last.loc[i:] = i                              # later rows: this is their latest one
    # After the last report inside the data, the next one is past the cutoff: treat it as a quarter away
    days_until = (nxt - d.index).where(nxt.notna(), 63.0)
    days_since = d.index - last

    f["volume_vs_20d"] = v / v.shift(1).rolling(20).mean()
    f["attention"] = (np.log(c / c.shift(5))).abs() / (sig63 * np.sqrt(5)) + np.exp(-days_until / 5)
    f["up_volume_share_20d"] = (v * (r > 0)).rolling(20).sum() / v.rolling(20).sum()

    f["vs_sp500_21d"] = c / c.shift(21) - d.sp500_close / d.sp500_close.shift(21)
    f["beta_63d"] = r.rolling(63).cov(m) / m.rolling(63).var()
    f["corr_63d"] = r.rolling(63).corr(m)

    f["days_to_next_session"] = (d.date.shift(-1) - d.date).dt.days
    f["report_before_next_close"] = (days_until == 1).astype(float)
    f["days_since_report"] = days_since
    z = pd.Series(np.nan, index=d.index)
    for i in react:
        if i > 0:
            z.loc[i] = r.iloc[i] / sig63.iloc[i - 1]
    f["last_report_reaction"] = z.ffill()
    f["is_friday"] = (d.date.dt.dayofweek == 4).astype(float)
    f["options_expiry"] = ((d.date.dt.dayofweek == 4) & d.date.dt.day.between(15, 21)).astype(float)
    month = d.date.dt.to_period("M")
    pos = d.groupby(month).cumcount()
    from_end = d.groupby(month).cumcount(ascending=False)
    f["turn_of_month"] = ((pos <= 2) | (from_end == 0)).astype(float)

    f.insert(0, "date", d.date)
    return f[list(["date"] + list(STOCK_CATALOG))]


def macro_features(ticker="BRK-B"):
    """Same for every stock; built from one stock's table (its FRED and S&P columns)."""
    d = load_table(ticker)
    spx = d.sp500_close
    m = spx.pct_change()
    f = pd.DataFrame({"date": d.date})
    f["vix"] = d.VIXCLS
    f["vix_chg_5d"] = d.VIXCLS - d.VIXCLS.shift(5)
    f["sp500_vol_21d"] = m.rolling(21).std()
    f["vix_minus_realized"] = d.VIXCLS - 100 * np.sqrt(252) * f["sp500_vol_21d"]
    f["sp500_ret_1d"] = m
    f["sp500_ret_21d"] = spx / spx.shift(21) - 1
    f["sp500_from_52w_high"] = spx / spx.rolling(252).max() - 1
    f["yield10_chg_21d"] = d.DGS10 - d.DGS10.shift(21)
    f["curve_slope"] = d.T10Y2Y
    f["yield2_chg_21d"] = d.DGS2 - d.DGS2.shift(21)
    f["real_yield_chg_21d"] = d.DFII10 - d.DFII10.shift(21)
    f["slope_chg_21d"] = d.T10Y2Y - d.T10Y2Y.shift(21)
    f["inflation_exp_chg_21d"] = d.T10YIE - d.T10YIE.shift(21)
    f["baa_spread_chg_21d"] = d.BAA10Y - d.BAA10Y.shift(21)
    f["baa_spread"] = d.BAA10Y
    f["dollar_chg_21d"] = d.DTWEXBGS / d.DTWEXBGS.shift(21) - 1
    # WTI went negative in April 2020, so a plain % change breaks; scale the $ change by the 63-day average price
    oil = d.DCOILWTICO
    f["oil_chg_21d"] = (oil - oil.shift(21)) / oil.rolling(63).mean().abs()
    f["claims_chg_4w"] = d.ICSA / d.ICSA.shift(20) - 1
    return f[["date"] + list(MACRO_CATALOG)]
