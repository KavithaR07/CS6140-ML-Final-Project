from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FRED_DIR = ROOT / "data" / "processed" / "fred_cleaned"

print("\nFINAL FRED VALIDATION")
print("=" * 60)

files = sorted(FRED_DIR.glob("*_cleaned.csv"))

print("FRED files found:", len(files))

for file in files:
    df = pd.read_csv(file)

    date_cols = [
        "observation_date",
        "estimated_release_date",
        "available_date"
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    missing_dates = df[date_cols].isna().sum().sum()
    duplicates = df["available_date"].duplicated().sum()

    early_availability = (
        df["available_date"] < df["estimated_release_date"]
    ).sum()

    print(f"\nIndicator: {file.stem}")
    print("Rows:", len(df))
    print("Missing dates:", missing_dates)
    print("Duplicate available dates:", duplicates)
    print("Available before estimated release:", early_availability)
    print("Dates sorted:", df["available_date"].is_monotonic_increasing)

print("\nFRED VALIDATION COMPLETED")