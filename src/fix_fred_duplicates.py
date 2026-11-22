from pathlib import Path
import pandas as pd

INPUT_DIR = Path("data/processed/fred_aligned")
OUTPUT_DIR = Path("data/processed/fred_cleaned")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nFRED DUPLICATE DATE CLEANING")
print("=" * 50)

for file in sorted(INPUT_DIR.glob("*_aligned.csv")):

    df = pd.read_csv(file)

    df["observation_date"] = pd.to_datetime(
        df["observation_date"], errors="coerce"
    )

    df["estimated_release_date"] = pd.to_datetime(
        df["estimated_release_date"], errors="coerce"
    )

    df["available_date"] = pd.to_datetime(
        df["available_date"], errors="coerce"
    )

    original_rows = len(df)

    # Reject records with invalid dates
    df = df.dropna(
        subset=[
            "observation_date",
            "estimated_release_date",
            "available_date"
        ]
    )

    df = df[
        df["estimated_release_date"] <= df["available_date"]
    ].copy()

 
    df = df.sort_values(
        [
            "available_date",
            "estimated_release_date",
            "observation_date"
        ]
    )


    df = df.drop_duplicates(
        subset=["available_date"],
        keep="last"
    )

    df = df.sort_values("available_date")

    duplicate_count = df["available_date"].duplicated().sum()

    output_file = OUTPUT_DIR / file.name.replace(
        "_aligned.csv", "_cleaned.csv"
    )

    df.to_csv(output_file, index=False)

    print(f"\nIndicator: {file.stem}")
    print(f"Original rows: {original_rows}")
    print(f"Cleaned rows: {len(df)}")
    print(f"Duplicate dates remaining: {duplicate_count}")
    print(f"Saved to: {output_file}")

    assert duplicate_count == 0, "Duplicate dates still exist!"

print("\nFRED duplicate cleaning completed successfully!")