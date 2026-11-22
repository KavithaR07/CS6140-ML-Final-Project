from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "data" / "processed" / "sec_aligned"
OUTPUT_DIR = ROOT / "data" / "processed" / "sec_cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STOCKS = ["TSLA", "PEP", "BRK-B"]

print("\nSEC DUPLICATE CLEANING")
print("=" * 55)

for ticker in STOCKS:
    input_file = INPUT_DIR / f"{ticker}_sec_aligned.csv"

    if not input_file.exists():
        print(f"{ticker}: File not found")
        continue

    df = pd.read_csv(input_file)

    original_rows = len(df)

    df = df.drop_duplicates().copy()

    removed = original_rows - len(df)

    output_file = OUTPUT_DIR / f"{ticker}_sec_cleaned.csv"
    df.to_csv(output_file, index=False)

    print(f"\nStock: {ticker}")
    print(f"Original rows: {original_rows}")
    print(f"Exact duplicates removed: {removed}")
    print(f"Cleaned rows: {len(df)}")
    print(f"Duplicates remaining: {df.duplicated().sum()}")
    print(f"Saved to: {output_file.relative_to(ROOT)}")

print("\nSEC duplicate cleaning completed!")