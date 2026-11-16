import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEC_FOLDER = PROJECT_ROOT / "data" / "processed" / "sec"
PRICE_FOLDER = PROJECT_ROOT / "data" / "raw" / "prices"

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed" / "sec_aligned"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

CUTOFF_DATE = pd.Timestamp("2026-10-30")

print("\nHISTORICAL SEC DATA ALIGNMENT")

for ticker in ["BRK-B", "PEP", "TSLA"]:

    sec_file = SEC_FOLDER / f"{ticker}.json"
    price_file = PRICE_FOLDER / f"{ticker}.csv"

    with open(sec_file, "r", encoding="utf-8") as f:
        sec_data = json.load(f)

    prices = pd.read_csv(price_file)
    prices["date"] = pd.to_datetime(prices["date"])

    prices = prices[
        prices["date"] <= CUTOFF_DATE
    ].sort_values("date")

    trading_dates = prices["date"].drop_duplicates()

    records = []

    for metric_name, metric_data in sec_data.get(
        "facts", {}
    ).get("us-gaap", {}).items():

        for unit, observations in metric_data.get(
            "units", {}
        ).items():

            for observation in observations:

                filed_date = pd.to_datetime(
                    observation.get("filed"),
                    errors="coerce"
                )

                if pd.isna(filed_date):
                    continue

                if filed_date > CUTOFF_DATE:
                    continue

                records.append({
                    "metric": metric_name,
                    "unit": unit,
                    "period_end": observation.get("end"),
                    "filed": filed_date,
                    "value": observation.get("val")
                })

    financials = pd.DataFrame(records)

    if financials.empty:
        print(f"\n{ticker}: No financial records found.")
        continue

    trading_calendar = pd.DataFrame({
        "available_date": trading_dates
    })

    financials = financials.sort_values("filed")

    aligned = pd.merge_asof(
        financials,
        trading_calendar,
        left_on="filed",
        right_on="available_date",
        direction="forward",
        allow_exact_matches=False
    )

    aligned = aligned.dropna(subset=["available_date"])

    output_file = OUTPUT_FOLDER / f"{ticker}_sec_aligned.csv"

    aligned.to_csv(output_file, index=False)

    print(f"\nCompany: {ticker}")
    print("Financial observations:", len(financials))
    print("Aligned observations:", len(aligned))
    print("Earliest available date:", aligned["available_date"].min())
    print("Latest available date:", aligned["available_date"].max())
    print("Saved to:", output_file)

print("\nHistorical SEC alignment completed!")