"""Download the raw data for the project.

Every source is saved in the shape it comes in. Nothing is cleaned, filled,
aligned or joined here; that happens in the cleaning step.

    data/raw/prices/<TICKER>.csv              daily prices (Yahoo Finance)
    data/raw/fred/<SERIES>.csv                market and macro series (FRED)
    data/raw/sec/companyfacts/<TICKER>.json   quarterly and annual financials (SEC EDGAR)
    data/raw/sec/submissions/<TICKER>*.json   filing history, for earnings dates (SEC EDGAR)

Prices and FRED stop at END_DATE so a rerun gives the same numbers.
SEC files are a snapshot of everything filed up to the day you run this.

Run from the repo root:
    python src/download_data.py

SEC asks scripts to identify themselves. Set SEC_USER_AGENT to your name and
email before running, e.g. in PowerShell:
    $env:SEC_USER_AGENT = "Your Name you@example.com"

Uses only the Python standard library.
"""
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

END_DATE = "2026-10-31"  # last day kept for prices and FRED

# ticker: SEC CIK number
STOCKS = {"BRK-B": 1067983, "PEP": 77476, "TSLA": 1318605}

# Yahoo symbol: file name
INDEXES = {"^GSPC": "GSPC"}  # S&P 500

FRED_SERIES = [
    "VIXCLS",      # VIX close
    "DGS10",       # 10-year Treasury yield
    "DGS2",        # 2-year Treasury yield
    "T10Y2Y",      # 10-year minus 2-year
    "BAA10Y",      # Moody's Baa corporate bond yield minus 10-year Treasury
    "DFII10",      # 10-year real yield (TIPS)
    "T10YIE",      # 10-year breakeven inflation
    "DTWEXBGS",    # trade-weighted US dollar index
    "DCOILWTICO",  # WTI crude oil price
    "ICSA",        # initial jobless claims (weekly)
]

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
SEC_UA = os.environ.get("SEC_USER_AGENT", "CS6140 course project research-script")
YAHOO_UA = "Mozilla/5.0"


def fetch(url, user_agent=None):
    """GET a URL and return the raw bytes, retrying twice on failure."""
    headers = {"User-Agent": user_agent} if user_agent else {}
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)


def save_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def download_prices(symbol, name):
    """Full daily history from Yahoo's chart endpoint (the same one yfinance uses)."""
    end = datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
           f"?period1=-1325635200&period2={int(end.timestamp())}&interval=1d")
    result = json.loads(fetch(url, YAHOO_UA))["chart"]["result"][0]
    offset = result["meta"]["gmtoffset"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"]["adjclose"][0]["adjclose"]

    path = os.path.join(RAW, "prices", f"{name}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = 0
    with open(path, "w", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["date", "open", "high", "low", "close", "adj_close", "volume"])
        for i, ts in enumerate(result["timestamp"]):
            # Windows can't convert pre-1970 timestamps directly, so add them to the epoch
            day = (datetime(1970, 1, 1) + timedelta(seconds=ts + offset)).date().isoformat()
            if day > END_DATE:
                continue
            values = [quote["open"][i], quote["high"][i], quote["low"][i], quote["close"][i], adj[i], quote["volume"][i]]
            out.writerow([day] + ["" if v is None else v for v in values])
            rows += 1
    print(f"  prices/{name}.csv: {rows} days")


def download_fred(series):
    """FRED's CSV exactly as served, cut at END_DATE."""
    lines = fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}").decode().splitlines()
    kept = [lines[0]] + [line for line in lines[1:] if line[:10] <= END_DATE]
    save_bytes(os.path.join(RAW, "fred", f"{series}.csv"), ("\n".join(kept) + "\n").encode())
    print(f"  fred/{series}.csv: {len(kept) - 1} rows")


def download_sec(ticker, cik):
    """Company facts plus the full filing history back to 2008."""
    base = "https://data.sec.gov"
    facts = fetch(f"{base}/api/xbrl/companyfacts/CIK{cik:010d}.json", SEC_UA)
    save_bytes(os.path.join(RAW, "sec", "companyfacts", f"{ticker}.json"), facts)
    time.sleep(0.2)

    # The main submissions file holds the most recent ~1000 filings; older ones are in extra pages.
    subs = fetch(f"{base}/submissions/CIK{cik:010d}.json", SEC_UA)
    save_bytes(os.path.join(RAW, "sec", "submissions", f"{ticker}.json"), subs)
    pages = 0
    for page in json.loads(subs)["filings"].get("files", []):
        if page["filingTo"] >= "2008-01-01":
            time.sleep(0.2)
            number = page["name"].rsplit("-", 1)[-1]  # e.g. "001.json"
            save_bytes(os.path.join(RAW, "sec", "submissions", f"{ticker}-{number}"),
                       fetch(f"{base}/submissions/{page['name']}", SEC_UA))
            pages += 1
    print(f"  sec/{ticker}: companyfacts + submissions ({pages} older pages)")


if __name__ == "__main__":
    print("Yahoo Finance prices")
    for ticker in STOCKS:
        download_prices(ticker, ticker)
    for symbol, name in INDEXES.items():
        download_prices(symbol, name)

    print("FRED series")
    for series in FRED_SERIES:
        download_fred(series)

    print("SEC EDGAR")
    for ticker, cik in STOCKS.items():
        download_sec(ticker, cik)

    print(f"Done. Prices and FRED cut at {END_DATE}; SEC is a snapshot as of {datetime.now():%Y-%m-%d}.")
