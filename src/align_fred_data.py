from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FRED_FOLDER = PROJECT_ROOT / "data" / "raw" / "fred"
PRICE_FOLDER = PROJECT_ROOT / "data" / "raw" / "prices"

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed" / "fred_aligned"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Timestamp("2010-06-29")
CUTOFF_DATE = pd.Timestamp("2026-10-30")

PUBLICATION_LAGS = {
    "BAA10Y": 1,
    "DCOILWTICO": 2,
    "DFII10": 1,
    "DGS10": 1,
    "DGS2": 1,
    "DTWEXBGS": 7,
    "ICSA": 5,
    "T10Y2Y": 1,
    "T10YIE": 1,
    "VIXCLS": 1,
}

prices = pd.read_csv(PRICE_FOLDER / "TSLA.csv")

prices["date"] = pd.to_datetime(
    prices["date"],
    errors="coerce"
)

trading_calendar = pd.DataFrame({
    "available_date": (
        prices.loc[
            prices["date"].between(START_DATE, CUTOFF_DATE),
            "date"
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
}).reset_index(drop=True)

print("\nFRED PUBLICATION DELAY ALIGNMENT")

print("Modeling start date:", START_DATE.date())
print("Modeling cutoff date:", CUTOFF_DATE.date())
print("Trading dates:", len(trading_calendar))

for indicator, lag_days in PUBLICATION_LAGS.items():

    file = FRED_FOLDER / f"{indicator}.csv"

    df = pd.read_csv(file)

    df["observation_date"] = pd.to_datetime(
        df["observation_date"],
        errors="coerce"
    )

    df[indicator] = pd.to_numeric(
        df[indicator],
        errors="coerce"
    )
    df = df.dropna(
        subset=["observation_date", indicator]
    ).copy()

    HISTORY_START = START_DATE - pd.Timedelta(days=30)

    df = df[
        (df["observation_date"] >= HISTORY_START) &
        (df["observation_date"] <= CUTOFF_DATE)
    ].copy()

    df["estimated_release_date"] = (
        df["observation_date"] +
        pd.Timedelta(days=lag_days)
    )

    df = df.sort_values(
        "estimated_release_date"
    ).reset_index(drop=True)

    aligned = pd.merge_asof(
        df,
        trading_calendar,
        left_on="estimated_release_date",
        right_on="available_date",
        direction="forward",
        allow_exact_matches=False
    )
    aligned = aligned.dropna(
        subset=["available_date"]
    ).copy()

    output_file = (
        OUTPUT_FOLDER /
        f"{indicator}_aligned.csv"
    )

    aligned.to_csv(
        output_file,
        index=False
    )

    invalid_alignment = (
        aligned["available_date"] <=
        aligned["estimated_release_date"]
    ).sum()

    future_dates = (
        aligned["available_date"] >
        CUTOFF_DATE
    ).sum()

    duplicate_dates = aligned[
        "available_date"
    ].duplicated().sum()

    print(f"\nIndicator: {indicator}")
    print("Publication lag (calendar days):", lag_days)
    print("Aligned observations:", len(aligned))
    print("Earliest available date:", aligned["available_date"].min())
    print("Latest available date:", aligned["available_date"].max())
    print("Invalid alignments:", invalid_alignment)
    print("Dates after cutoff:", future_dates)
    print("Duplicate availability dates:", duplicate_dates)

    if invalid_alignment == 0 and future_dates == 0:
        print("Status: PASSED")
    else:
        print("Status: NEEDS REVIEW")

    print("Saved to:", output_file)

print("\nFRED publication delay alignment completed!")
