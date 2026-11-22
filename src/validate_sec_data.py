from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEC_DIR = ROOT / "data" / "processed" / "sec_aligned"

STOCKS = ["TSLA", "PEP", "BRK-B"]

print("\nSEC DATA VALIDATION")
print("=" * 55)

for ticker in STOCKS:
    file = SEC_DIR / f"{ticker}_sec_aligned.csv"

    if not file.exists():
        print(f"\n{ticker}: File not found")
        continue

    df = pd.read_csv(file)

    print(f"\nStock: {ticker}")
    print("Total rows:", len(df))
    print("Columns:", list(df.columns))

    for col in ["period_end", "filed", "available_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        print(f"Missing/invalid {col}:", df[col].isna().sum())

    print("Missing financial values:", df["value"].isna().sum())

    print("Exact duplicate rows:", df.duplicated().sum())

    keys = ["metric", "unit", "period_end", "filed", "value"]
    print("Duplicate financial records:", df.duplicated(subset=keys).sum())

    print(
        "Available before filing:",
        (df["available_date"] < df["filed"]).sum()
    )

    print(
        "Filing before period end:",
        (df["filed"] < df["period_end"]).sum()
    )

print("\nSEC VALIDATION COMPLETED")
