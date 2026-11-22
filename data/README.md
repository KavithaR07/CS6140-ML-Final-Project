# Data

Raw data for the project. Everything in `raw/` is saved exactly the way the source gives it. Nothing has been cleaned, filled in, lined up by date or joined yet. That's the cleaning step.

## Rebuilding it

From the repo root:

```
python src/download_data.py
```

- Standard library only, no installs needed.
- Prices and FRED series stop at **2026-10-31**, so a rerun gives the same numbers.
- SEC files are a snapshot of everything filed up to the download day (last run 2026-11-15). A rerun will pick up newer filings, so cleaning should drop anything filed after 2026-10-31. Berkshire's Q3 2026 10-Q (filed 2026-11-09) is already in there.
- SEC asks scripts to identify themselves. Set `SEC_USER_AGENT` to your name and email before running. In PowerShell: `$env:SEC_USER_AGENT = "Your Name you@example.com"`

## Running the pipeline

From the repo root, in this order:

```
python src/download_data.py          # raw data (already in data/raw/, only rerun to refresh it)
python src/build_modeling_table.py   # -> data/processed/modeling/<TICKER>.csv
python src/eda_modeling_table.py     # EDA charts -> results/figures/modeling_*.png
python src/feature_overlap_chart.py  # feature overlap charts -> results/figures/feature_overlap_*.png
python src/ridge_baseline.py         # Ridge walk-forward -> results/ridge_baseline_*.csv
```

Needs pandas, numpy, matplotlib and scikit-learn (all in `requirements.txt`). The Ridge run takes about half a minute. `results/ridge_baseline_predictions.csv` (every test-day prediction, about 10 MB) isn't committed; the last step recreates it.

### `processed/modeling/`

One table per stock, one row per trading day. Berkshire and Pepsi start 2008-01-02 and Tesla 2010-06-29, so the 63-day lookbacks have history before modeling starts; all end 2026-10-30.

- `open, high, low, close, adj_close, volume`, plus dividend-adjusted `adj_open, adj_high, adj_low` (open/high/low scaled by `adj_close / close`)
- `sp500_close`
- the 10 FRED series as known at that day's close. Each value only appears once it was released: VIX the same day, daily rates and the Baa spread the next day, oil 2 days, jobless claims 5, the dollar index 7.
- labels: `ret_next` (next trading day's close-to-close return), `up_next` (1 if it went up), `ret_next_open_to_close` (side label)

Features are built from these tables by `src/features.py`. Its catalog is the source of truth for every feature's group (what it captures) and role (in the pool, held out as a near-duplicate, or a new candidate).

## What's in `raw/`

```
raw/
  prices/              daily prices, one CSV per ticker (Yahoo Finance)
    BRK-B.csv            Berkshire Hathaway B shares, from 1996-05-09
    PEP.csv              PepsiCo, from 1972-06-01
    TSLA.csv             Tesla, from its IPO on 2010-06-29
    GSPC.csv             S&P 500 index, from 1927-12-30
  fred/                market and macro series, one CSV per series (FRED)
  sec/
    companyfacts/      every number each company has reported in its 10-Qs and 10-Ks (SEC EDGAR)
    submissions/       each company's filing history, used for earnings dates (SEC EDGAR)
```

Price CSV columns: `date, open, high, low, close, adj_close, volume`. `close` is adjusted for splits. `adj_close` is also adjusted for dividends, so use `adj_close` for returns. Berkshire, Tesla and the index don't pay dividends, so their two columns match. Pepsi's don't.

FRED CSVs are two columns, `observation_date` and the series ID. Blank values are days that market was closed. For the rate series that includes bond holidays when stocks still traded.

## How each part gets used

There are three kinds of data here: unsupervised inputs for the GMM, labeled base data for the three stocks, and feature sources that get turned into predictor columns.

### 1. Unsupervised: GMM regime inputs

These are market-wide, so they're the same for every stock. The GMM clusters days into market regimes (calm, choppy, stressed and so on) using only these six inputs. The inputs stay fixed in every experiment. The GMM is fit on data from 2005 on, refit inside each walk-forward training window, and only sees data from before each test period. It picks 2 to 4 regimes by silhouette, stability across restarts, and whether each regime can be named.

| GMM input | Built from | Note |
|---|---|---|
| VIX level | `fred/VIXCLS.csv` | as is |
| S&P 500 volatility over the last 21 trading days | `prices/GSPC.csv` | std of daily returns |
| S&P 500 return over the last 63 trading days | `prices/GSPC.csv` | |
| Baa credit spread | `fred/BAA10Y.csv` | the level is fine, it stays in a steady range |
| 10-year yield change over the last 63 trading days | `fred/DGS10.csv` | use the change, not the level, since rate levels jumped to a new range after 2022 |
| Yield curve slope (10y minus 2y) | `fred/T10Y2Y.csv` | lowest was -0.19 in 2005-2021 but -1.08 after 2022, so the first deep inversion is new territory |

Standardize the inputs using each training window's data only. The output is a probability of each regime for every day. It isn't saved here yet; it comes out of the GMM step.

### 2. Labeled base data: the three stocks

`prices/BRK-B.csv`, `prices/PEP.csv`, `prices/TSLA.csv`

The labels and the technical features both come from these files.

- **Label (target):** the next trading day's direction (up or down), from `adj_close`. A weekly version (Friday close to Friday close, one row per week) is a stretch goal.
- **Rows:** one per trading day from 2009 on (Tesla from mid-2010). The files go back further so the 63-day lookbacks have history at the start.
- **Three different kinds of stock on purpose:** Berkshire moves with the market, Pepsi moves mostly on its own reports, and Tesla moves on headlines.

### 3. Feature sources: predictor columns

Rule for everything here: a feature for day t has to be known by day t's close.

The feature sets compared in the experiments: technical; technical + macro; technical + fundamentals; all three. Each one gets run with and without the regime columns.

**Technical** (from the stock's own price file): returns over the last 1, 5, 21 and 63 days, volume vs its 20-day average, 21-day volatility, and similar.

**Macro** (FRED, same for every stock): the GMM inputs above in raw form, plus:

| File | What it is | Frequency | Why it matters |
|---|---|---|---|
| `DGS2` | 2-year Treasury yield | daily | tracks where the market expects the Fed to go |
| `DFII10` | 10-year real yield (TIPS) | daily, from 2003 | rate after inflation, which is what valuations react to |
| `T10YIE` | 10-year breakeven inflation | daily, from 2003 | the bond market's inflation forecast, no release delay |
| `DTWEXBGS` | trade-weighted US dollar | daily values, published about a week behind | a strong dollar shrinks overseas earnings |
| `DCOILWTICO` | WTI crude oil | daily | input cost and inflation driver |
| `ICSA` | initial jobless claims | weekly | fastest read on the real economy |

**Fundamentals** (SEC, different for each stock). `companyfacts` has the quarterly numbers and `submissions` gives the earnings dates. Raw numbers only change once a quarter, so turn them into features that change daily: price / trailing earnings, price / book, price / sales, growth rates, days since the last report.

- **PepsiCo:** reports revenue, net income, operating income, assets, liabilities and equity cleanly every quarter.
- **Berkshire:** since 2018 its net income includes paper gains and losses on its stock portfolio, so it mostly tracks the market. Use net income minus `GainLossOnInvestments` (operating earnings), and `StockholdersEquity` for price / book. It doesn't report diluted EPS the normal way, so compute per-share numbers ourselves.
- **Tesla:** it lost money in 32 of its first 60 quarters (last loss Q2 2019), so price / earnings doesn't work for half the history. Use price / sales and growth rates. Quarterly revenue went from $21M in Q1 2010 to $28.0B, so levels aren't comparable across years.

**Regime** (from the GMM step): the regime probabilities as columns, added to a feature set or left out.

### Not downloaded yet (stretch)

- **Wikipedia daily page views:** free official API, data from July 2015. Tesla's page was "Tesla Motors" until Feb 2017, so add both titles together.
- **Google Trends:** no open official API. Daily values only come in windows of about 9 months, each scaled 0-100 to its own peak, so use ratios inside a window (like today vs the past 4 weeks) to avoid leaking the future.

## Things to handle in cleaning

1. **Bond holidays.** The bond market closes on some days the stock market is open (Columbus Day, Veterans Day), which leaves blanks in the FRED rate series. Carry the previous value forward.
2. **Weekly release delays.** `ICSA` rows are dated the Saturday the week ends but come out the following Thursday. The last row, 2026-10-31, wasn't public until Nov 5. `DTWEXBGS` is published about a week behind. Only join a value from the day it was actually released.
3. **Join SEC numbers on the `filed` date, not the period end date.** Otherwise the model knows earnings before the market did.
4. **SEC has quarterly and year-to-date values under the same field.** Keep the 3-month ones by checking the `start` and `end` dates. Q4 usually only appears as a full-year number, so Q4 = full year minus the first nine months.
5. **Revenue field names.** Pepsi switched from `SalesRevenueNet` (through 2018) to `Revenues` (from 2016). Berkshire and Tesla use `Revenues` the whole way.
6. **Earnings dates** = 8-K filings with item 2.02 plus 10-Q/10-K filing dates. Treat the filing day and the next trading day as the reaction window.
   - Berkshire's recent 10-Qs hit EDGAR on Monday mornings before the open (it posts results on its own site the Saturday before), so the reaction is Monday. Berkshire doesn't always file an 8-K, so use the 10-Q dates.
   - Tesla files two item 2.02 8-Ks a quarter: deliveries around the 2nd of the month after the quarter ends, then earnings about three weeks later. Both move the stock.
7. **Drop SEC filings after 2026-10-31** (see above).

## Sources and terms

- SEC EDGAR, and the Federal Reserve, Energy Department and Labor Department series on FRED, are public domain.
- Yahoo Finance data, VIX (Cboe) and the Baa spread (Moody's) belong to their providers and are covered by their terms. They're included here only for this course project.
