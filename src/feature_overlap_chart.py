"""Overlap charts for feature selection: features grouped by what they capture.

Cells are the strength of the rank correlation between two features (sign ignored),
averaged over Berkshire, Pepsi and Tesla for the stock features. Overlapping features
should show up as dark squares inside a group, with light cells between groups.

Saves results/figures/feature_overlap_stock.png and feature_overlap_macro.png,
and prints each feature's strongest overlap.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import MACRO_CATALOG, MODEL_START, STOCK_CATALOG, macro_features, stock_features

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
# Sequential blue ramp (light = little overlap, dark = a lot)
RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ROLE_STYLE = {"pool": (INK, "normal", ""), "held": (MUTED, "italic", "  (held out)"), "new": (INK2, "normal", "  (new)")}

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def abs_corr(frames):
    mats = [f.drop(columns="date").corr(method="spearman").abs() for f in frames]
    return sum(mats) / len(mats)


def draw(corr, catalog, path, main, sub):
    names = list(catalog)
    groups = [catalog[n][0] for n in names]
    n = len(names)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    fig, ax = plt.subplots(figsize=(12, 10 if n > 20 else 8.5))
    ax.imshow(corr.loc[names, names].to_numpy(), cmap=cmap, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j] if False else corr.loc[names[i], names[j]]
            if i != j and val >= 0.5:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="#ffffff")
    # group boundaries
    edges = [i for i in range(1, n) if groups[i] != groups[i - 1]]
    for e in edges:
        ax.axhline(e - 0.5, color=SURFACE, lw=3)
        ax.axvline(e - 0.5, color=SURFACE, lw=3)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([nm + ROLE_STYLE[catalog[nm][1]][2] for nm in names], fontsize=8)
    for lab, nm in zip(ax.get_yticklabels(), names):
        color, style, _ = ROLE_STYLE[catalog[nm][1]]
        lab.set_color(color); lab.set_fontstyle(style)
        if catalog[nm][1] == "pool":
            lab.set_fontweight("bold")
    for lab, nm in zip(ax.get_xticklabels(), names):
        lab.set_color(ROLE_STYLE[catalog[nm][1]][0])
    # group names on the right
    starts = [0] + edges
    ends = edges + [n]
    for s, e in zip(starts, ends):
        ax.text(n - 0.3, (s + e - 1) / 2, groups[s], va="center", ha="left", fontsize=9.5, fontweight="bold", color=INK)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.01, 0.985, main, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.955, sub, fontsize=9.5, color=INK2, va="top")
    cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.025, pad=0.16)
    cb.set_label("Overlap (rank correlation, sign ignored)", color=INK2)
    cb.outline.set_visible(False)
    fig.subplots_adjust(top=0.92, left=0.2, right=0.92, bottom=0.14)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def strongest(corr, catalog):
    rows = []
    for nm in catalog:
        others = corr.loc[nm].drop(nm)
        partner = others.idxmax()
        rows.append((catalog[nm][0], catalog[nm][1], nm, partner, others.max()))
    return pd.DataFrame(rows, columns=["group", "role", "feature", "strongest overlap", "overlap"])


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    stock = [stock_features(t).loc[lambda f: f.date >= MODEL_START[t]].dropna() for t in ["BRK-B", "PEP", "TSLA"]]
    per_stock = {t: f.drop(columns="date").corr(method="spearman").abs() for t, f in zip(["BRK-B", "PEP", "TSLA"], stock)}
    sc = abs_corr(stock)
    draw(sc, STOCK_CATALOG, FIGS / "feature_overlap_stock.png", "Stock features, grouped by what they capture",
         "Overlap averaged over Berkshire, Pepsi and Tesla, 2009 to Oct 2026. Bold = in the 15-feature pool. "
         "Values shown where overlap is 0.5 or more.")
    macro = macro_features().loc[lambda f: f.date >= "2009-01-01"].dropna()
    mc = abs_corr([macro])
    draw(mc, MACRO_CATALOG, FIGS / "feature_overlap_macro.png", "Macro features, grouped by what they capture",
         "Same for every stock, 2009 to Oct 2026. Bold = in the 10-feature pool. Values shown where overlap is 0.5 or more.")
    pd.set_option("display.width", 200)
    s = strongest(sc, STOCK_CATALOG)
    for t, c in per_stock.items():
        s[t] = [c.loc[f].drop(f).max() for f in s.feature]
    print(s.round(2).to_string(index=False))
    print()
    print(strongest(mc, MACRO_CATALOG).round(2).to_string(index=False))
    print("\nrows used:", {t: len(f) for t, f in zip(["BRK-B", "PEP", "TSLA"], stock)}, "macro:", len(macro))
