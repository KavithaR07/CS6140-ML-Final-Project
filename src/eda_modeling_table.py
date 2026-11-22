"""EDA charts from the per-stock modeling tables (run build_modeling_table.py first).

Saves to results/figures/:
    modeling_growth_log.png        growth of $1, log scale
    modeling_macro_inputs.png      VIX, 10-year yield, curve slope, Baa spread, with stressed periods shaded
    modeling_next_day_returns.png  distribution of next-day returns per stock
    modeling_autocorrelation.png   does today's move predict tomorrow's direction or size?
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "data" / "processed" / "modeling"
FIGS = ROOT / "results" / "figures"
START = "2009-01-01"

# Colors: one per stock, kept the same in every chart. S&P 500 is the gray reference.
COLOR = {"BRK-B": "#2a78d6", "PEP": "#eb6834", "TSLA": "#1baf7a", "S&P 500": "#52514e"}
NAME = {"BRK-B": "Berkshire", "PEP": "Pepsi", "TSLA": "Tesla", "S&P 500": "S&P 500"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titlelocation": "left", "legend.frameon": False,
})


def title(fig, main, sub):
    fig.text(0.01, 0.985, main, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.935, sub, fontsize=10, color=INK2, va="top")


tables = {t: pd.read_csv(TABLES / f"{t}.csv", parse_dates=["date"]) for t in ["BRK-B", "PEP", "TSLA"]}
tables = {t: d[d["date"] >= START].reset_index(drop=True) for t, d in tables.items()}
FIGS.mkdir(parents=True, exist_ok=True)

# 1. Growth of $1, log scale -------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.2))
series = {t: d.set_index("date")["adj_close"] for t, d in tables.items()}
series["S&P 500"] = tables["BRK-B"].set_index("date")["sp500_close"]
ends = {}
for name, s in series.items():
    g = s / s.iloc[0]
    ax.plot(g.index, g.values, color=COLOR[name], lw=1.6 if name != "S&P 500" else 1.3,
            ls="-" if name != "S&P 500" else "--", label=NAME[name])
    ends[name] = (g.index[-1], g.iloc[-1])
# End labels, nudged apart so lines that finish close together don't overlap
placed = []
for name, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
    ly = np.log10(y)
    if placed and ly - placed[-1] < 0.1:
        ly = placed[-1] + 0.1
    placed.append(ly)
    ax.annotate(f"{NAME[name]}  ${y:,.2f}" if y < 10 else f"{NAME[name]}  ${y:,.0f}", (x, y), xytext=(x + pd.Timedelta(days=60), 10 ** ly),
                textcoords="data", va="center", color=INK2, fontsize=9)
ax.set_xticks(pd.to_datetime([f"{yr}-01-01" for yr in range(2010, 2027, 2)]))
ax.set_xticklabels([str(yr) for yr in range(2010, 2027, 2)])
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500])
ax.set_yticklabels(["$0.50", "$1", "$2", "$5", "$10", "$20", "$50", "$100", "$200", "$500"])
ax.set_ylabel("Value of $1 invested (log scale)")
ax.set_xlim(right=series["BRK-B"].index[-1] + pd.Timedelta(days=900))
ax.legend(loc="upper left")
title(fig, "Growth of $1, 2009 to Oct 2026",
      "Dividends reinvested. Tesla starts at its June 2010 IPO. Log scale, so equal steps are equal percentage moves.")
fig.subplots_adjust(top=0.84, left=0.08, right=0.98, bottom=0.08)
fig.savefig(FIGS / "modeling_growth_log.png", dpi=200)
plt.close(fig)

# 2. Market and macro inputs, stressed periods shaded ------------------------
m = tables["BRK-B"].set_index("date")
stress = m["VIXCLS"] > 25
fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
panels = [("VIXCLS", "VIX (market fear gauge)", None),
          ("DGS10", "10-year Treasury yield (%)", None),
          ("T10Y2Y", "Yield curve slope: 10-year minus 2-year (%)", 0),
          ("BAA10Y", "Baa corporate bond spread over Treasuries (%)", None)]
for ax, (col, label, ref) in zip(axes.flat, panels):
    lo, hi = m[col].min(), m[col].max()
    ax.fill_between(m.index, lo, hi, where=stress, color=GRID, alpha=0.9, lw=0, step="mid")
    ax.plot(m.index, m[col], color="#2a78d6", lw=1.2)
    if ref is not None:
        ax.axhline(ref, color=AXIS, lw=1)
    ax.set_title(label)
    ax.set_ylim(lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))
title(fig, "The market and macro inputs, 2009 to Oct 2026",
      "Gray bands: days VIX closed above 25 (stressed market). Values as known at each day's close.")
fig.subplots_adjust(top=0.84, left=0.06, right=0.98, bottom=0.06, hspace=0.35, wspace=0.18)
fig.savefig(FIGS / "modeling_macro_inputs.png", dpi=200)
plt.close(fig)

# 3. Next-day return distributions ------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
bins = np.arange(-10, 10.5, 0.5)
for ax, (t, d) in zip(axes, tables.items()):
    r = d["ret_next"].dropna() * 100
    w = np.ones(len(r)) / len(r) * 100
    ax.hist(r.clip(-10, 10), bins=bins, weights=w, color=COLOR[t], edgecolor=SURFACE, lw=0.6)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(NAME[t])
    ax.set_xlabel("Next-day return (%)")
    ax.text(0.97, 0.95, f"up {np.mean(r > 0):.1%} of days\ntypical move ±{r.std():.1f}%\nbeyond ±10%: {np.mean(r.abs() > 10):.1%}",
            transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=9)
axes[0].set_ylabel("Share of days (%)")
title(fig, "Next-day returns: the label we predict",
      "Each bar is a 0.5% slice. Moves past ±10% are piled into the end bars.")
fig.subplots_adjust(top=0.80, left=0.06, right=0.99, bottom=0.14, wspace=0.08)
fig.savefig(FIGS / "modeling_next_day_returns.png", dpi=200)
plt.close(fig)

# 4. Autocorrelation: direction vs size -------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
lags = np.arange(1, 11)
for ax, (t, d) in zip(axes, tables.items()):
    r = d["adj_close"].pct_change().dropna()
    direction = [r.autocorr(k) for k in lags]
    size = [r.abs().autocorr(k) for k in lags]
    ax.bar(lags - 0.2, direction, width=0.38, color=AXIS, label="return (direction)")
    ax.bar(lags + 0.2, size, width=0.38, color=COLOR[t], label="size of move, |return|")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(NAME[t])
    ax.set_xticks(lags)
    ax.set_xlabel("Days apart")
    ax.legend(loc="upper right", fontsize=8.5)
axes[0].set_ylabel("Correlation")
title(fig, "Does today's move say anything about later days?",
      "Correlation between a day's move and the move k days later, 2009 to Oct 2026.")
fig.subplots_adjust(top=0.80, left=0.06, right=0.99, bottom=0.14, wspace=0.08)
fig.savefig(FIGS / "modeling_autocorrelation.png", dpi=200)
plt.close(fig)

print("Saved 4 charts to results/figures/")
