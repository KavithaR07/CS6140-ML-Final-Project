"""Build one modeling table per stock: one row per trading day.

Each row holds what was known at that day's close, plus the next-day labels.

    prices      the stock's own open/high/low/close/adj_close/volume, plus
                dividend-adjusted open/high/low, and the S&P 500 close
    macro       every FRED series as of that day, using only values already released
    labels      next trading day's return and direction (close to close),
                plus an open-to-close version as a side label

Features (returns over 1/5/21/63 days, volatility, etc.) are built from this
table in the next step. Rows start in 2008 so those lookbacks have history;
modeling rows start in 2009 (Tesla: its first trading day, 2010-06-29).

Run from the repo root:
    python src/build_modeling_table.py
Output: data/processed/modeling/<TICKER>.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "modeling"

STOCKS = ["BRK-B", "PEP", "TSLA"]
TABLE_START = "2008-01-01"
END_DATE = "2026-10-30"  # last trading day in the raw data

# Calendar days between a FRED value's date and when it can be used.
# 0 = known at that day's close. Weekly series come out days after the period ends.
RELEASE_DELAY_DAYS = {
    "VIXCLS": 0,       # VIX close is known at the close
    "DGS10": 1,        # Treasury yields, published the next day
    "DGS2": 1,
    "T10Y2Y": 1,
    "DFII10": 1,
    "T10YIE": 1,
    "BAA10Y": 1,       # Moody's Baa spread, published the next day
    "DCOILWTICO": 2,   # WTI oil
    "ICSA": 5,         # jobless claims: week ends Saturday, released Thursday
    "DTWEXBGS": 7,     # dollar index, published about a week behind
}


def load_prices(name):
    df = pd.read_csv(RAW / "prices" / f"{name}.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def fred_as_of(trading_days, series, delay_days):
    """For each trading day, the latest value released on or before that day."""
    df = pd.read_csv(RAW / "fred" / f"{series}.csv", parse_dates=["observation_date"])
    df = df.rename(columns={series: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])  # blank = market closed that day
    df["released"] = df["observation_date"] + pd.Timedelta(days=delay_days)
    df = df.sort_values("released")

    merged = pd.merge_asof(
        pd.DataFrame({"date": trading_days}),
        df[["released", "value"]],
        left_on="date",
        right_on="released",
        direction="backward",  # latest release on or before the trading day
    )
    return merged["value"].to_numpy()


def build(ticker, sp500):
    px = load_prices(ticker)
    px = px[(px["date"] >= TABLE_START) & (px["date"] <= END_DATE)].reset_index(drop=True)

    # open/high/low/close are split-adjusted only; adj_close also includes dividends.
    # Scale open/high/low the same way so they line up with adj_close.
    factor = px["adj_close"] / px["close"]
    for col in ["open", "high", "low"]:
        px[f"adj_{col}"] = px[col] * factor

    # S&P 500 close on the same day
    px = px.merge(sp500[["date", "adj_close"]].rename(columns={"adj_close": "sp500_close"}),
                  on="date", how="left")

    # FRED, as of each trading day
    for series, delay in RELEASE_DELAY_DAYS.items():
        px[series] = fred_as_of(px["date"], series, delay)

    # Labels: what happens on the NEXT trading day
    next_close = px["adj_close"].shift(-1)
    px["ret_next"] = next_close / px["adj_close"] - 1            # close to close
    px["up_next"] = (px["ret_next"] > 0).astype("Int64")
    px.loc[px["ret_next"].isna(), "up_next"] = pd.NA               # last row has no next day
    px["ret_next_open_to_close"] = px["close"].shift(-1) / px["open"].shift(-1) - 1  # side label

    px.insert(0, "ticker", ticker)
    return px


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sp500 = load_prices("GSPC")
    for ticker in STOCKS:
        table = build(ticker, sp500)
        table.to_csv(OUT / f"{ticker}.csv", index=False)
        print(f"{ticker}: {len(table)} rows, {table['date'].min().date()} to {table['date'].max().date()}, "
              f"{table.shape[1]} columns -> data/processed/modeling/{ticker}.csv")
