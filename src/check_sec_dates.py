from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEC_DIR = ROOT / "data" / "processed" / "sec_cleaned"

for ticker in ["TSLA", "PEP", "BRK-B"]:
    file = SEC_DIR / f"{ticker}_sec_cleaned.csv"

    df = pd.read_csv(file)

    df["period_end"] = pd.to_datetime(df["period_end"])
    df["filed"] = pd.to_datetime(df["filed"])

    unusual = df[df["filed"] < df["period_end"]]

    print(f"\n{'=' * 60}")
    print(f"{ticker}: {len(unusual)} unusual records")
    print("=" * 60)

    if not unusual.empty:
        print(unusual.to_string(index=False))