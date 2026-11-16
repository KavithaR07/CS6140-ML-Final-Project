from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"

def check_data_quality(folder):
    for file in sorted(folder.glob("*.csv")):
        print(f"\n{'=' * 50}")
        print(f"Dataset: {file.name}")
        df = pd.read_csv(file)
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print("\nMissing values:")
        print(df.isna().sum())
        date_column = "date" if "date" in df.columns else "observation_date"

        if date_column in df.columns:
            print(f"\nDuplicate dates: {df[date_column].duplicated().sum()}")
        print("=" * 50)

print("STOCK PRICE DATA QUALITY")
check_data_quality(RAW_DATA / "prices")

print("\nFRED ECONOMIC DATA QUALITY")
check_data_quality(RAW_DATA / "fred")

CLEANED_DATA = PROJECT_ROOT / "data" / "processed" / "fred"
CLEANED_DATA.mkdir(parents=True, exist_ok=True)

for file in sorted((RAW_DATA / "fred").glob("*.csv")):

    df = pd.read_csv(file)
    df["observation_date"] = pd.to_datetime(
        df["observation_date"]
    )
    df = df.sort_values("observation_date")
    df = df.ffill()

    output_file = CLEANED_DATA / file.name
    df.to_csv(output_file, index=False)

    print(f"\nCleaned: {file.name}")
    print("Remaining missing values:")
    print(df.isna().sum())

print("\nFRED cleaning completed!")
import numpy as np

print("\nSTOCK DATA VALIDATION")

for file in sorted((RAW_DATA / "prices").glob("*.csv")):

    df = pd.read_csv(file)

    print(f"\nDataset: {file.name}")
    print("\nData types:")
    print(df.dtypes)

    dates = pd.to_datetime(df["date"], errors="coerce")

    print("Invalid dates:", dates.isna().sum())
    print("Duplicate dates:", dates.duplicated().sum())
    print("Dates sorted:", dates.is_monotonic_increasing)

    print("Start date:", dates.min())
    print("End date:", dates.max())
    numeric_columns = [
        "open", "high", "low",
        "close", "adj_close", "volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    price_columns = [
        "open", "high", "low",
        "close", "adj_close"
    ]

    invalid_prices = (df[price_columns] <= 0).sum().sum()

    print("\nInvalid prices:", invalid_prices)
    negative_volume = (df["volume"] < 0).sum()

    print("Negative volume:", negative_volume)
    infinite_values = np.isinf(
        df[numeric_columns].to_numpy()
    ).sum()

    print("Infinite values:", infinite_values)
    df = df.sort_values("date")

    df["daily_return"] = df["adj_close"].pct_change(
        fill_method=None
    )

    outliers = (df["daily_return"].abs() > 0.20).sum()

    print("Daily returns above 20%:", outliers)

print("\nStock data validation completed!")
print("\nDAILY RETURN OUTLIER DETAILS")

for file in sorted((RAW_DATA / "prices").glob("*.csv")):

    df = pd.read_csv(file)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["daily_return"] = df["adj_close"].pct_change(
        fill_method=None
    )
    outliers = df[df["daily_return"].abs() > 0.20].copy()

    print(f"\nDataset: {file.name}")
    print(f"Number of outliers: {len(outliers)}")

    if not outliers.empty:
        outliers["daily_return_pct"] = (
            outliers["daily_return"] * 100
        )

        print(
            outliers[
                ["date", "adj_close", "daily_return_pct"]
            ].to_string(index=False)
        )

print("\nOutlier inspection completed!")

print("\nCOMMON TRADING DATE VALIDATION")

stock_files = ["BRK-B.csv", "PEP.csv", "TSLA.csv", "GSPC.csv"]

date_sets = []

for filename in stock_files:

    df = pd.read_csv(RAW_DATA / "prices" / filename)

    df["date"] = pd.to_datetime(df["date"])

    date_sets.append(set(df["date"]))

common_dates = set.intersection(*date_sets)

print("Total common trading dates:", len(common_dates))

if common_dates:
    print("Common start date:", min(common_dates))
    print("Common end date:", max(common_dates))

print("\nCommon date validation completed!")

print("\nFRED DATA VALIDATION")

for file in sorted((RAW_DATA / "fred").glob("*.csv")):

    df = pd.read_csv(file)

    date_column = "observation_date"
    value_column = df.columns[1]

    df[date_column] = pd.to_datetime(
        df[date_column], errors="coerce"
    )

    df[value_column] = pd.to_numeric(
        df[value_column], errors="coerce"
    )

    print(f"\nDataset: {file.name}")

    print("Start date:", df[date_column].min())
    print("End date:", df[date_column].max())

    print("Invalid dates:", df[date_column].isna().sum())

    print("Invalid or missing numeric values:",
          df[value_column].isna().sum())

    print("Infinite values:",
          np.isinf(df[value_column]).sum())

    print("Duplicate dates:",
          df[date_column].duplicated().sum())

print("\nFRED data validation completed!")

print("\nFRED MODELING PERIOD COVERAGE")

start_date = pd.Timestamp("2010-06-29")
end_date = pd.Timestamp("2026-10-30")

for file in sorted((RAW_DATA / "fred").glob("*.csv")):

    df = pd.read_csv(file)

    df["observation_date"] = pd.to_datetime(
        df["observation_date"]
    )
    df = df[
        (df["observation_date"] >= start_date) &
        (df["observation_date"] <= end_date)
    ]

    value_column = df.columns[1]

    print(f"\nDataset: {file.name}")
    print("Observations:", len(df))
    print("Missing values:", df[value_column].isna().sum())

print("\nFRED modeling-period coverage completed!")

import json

print("\nSEC FINANCIAL DATA VALIDATION")

sec_folder = RAW_DATA / "sec" / "companyfacts"

for file in sorted(sec_folder.glob("*.json")):

    print(f"\nDataset: {file.name}")

    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("Company name:", data.get("entityName"))
        print("CIK:", data.get("cik"))

        facts = data.get("facts", {})
        us_gaap = facts.get("us-gaap", {})

        print("Financial metrics:", len(us_gaap))
        print("Has financial data:", len(us_gaap) > 0)

    except (json.JSONDecodeError, OSError) as error:
        print("Error reading file:", error)

print("\nSEC financial data validation completed!")

print("\nSEC FINANCIAL FILING DATE VALIDATION")

for file in sorted(sec_folder.glob("*.json")):

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nCompany: {data.get('entityName')}")

    us_gaap = data.get("facts", {}).get("us-gaap", {})

    total_records = 0
    missing_filed_dates = 0
    invalid_filed_dates = 0
    filed_before_period_end = 0

    for metric_data in us_gaap.values():

        for observations in metric_data.get("units", {}).values():

            for record in observations:

                total_records += 1

                filed = record.get("filed")
                period_end = record.get("end")

                if not filed:
                    missing_filed_dates += 1
                    continue

                filed_date = pd.to_datetime(
                    filed, errors="coerce"
                )

                if pd.isna(filed_date):
                    invalid_filed_dates += 1
                    continue

                if period_end:
                    end_date = pd.to_datetime(
                        period_end, errors="coerce"
                    )

                    if (
                        pd.notna(end_date)
                        and filed_date < end_date
                    ):
                        filed_before_period_end += 1

    print("Total financial records:", total_records)
    print("Missing filing dates:", missing_filed_dates)
    print("Invalid filing dates:", invalid_filed_dates)
    print(
        "Filed before period end:",
        filed_before_period_end
    )

print("\nSEC filing date validation completed!")

print("\nSEC FILING DATE INVESTIGATION")

for file in sorted(sec_folder.glob("*.json")):

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nCompany: {data.get('entityName')}")

    us_gaap = data.get("facts", {}).get("us-gaap", {})

    flagged_records = []

    for metric_name, metric_data in us_gaap.items():

        for unit, observations in metric_data.get("units", {}).items():

            for record in observations:

                filed = pd.to_datetime(
                    record.get("filed"), errors="coerce"
                )

                period_end = pd.to_datetime(
                    record.get("end"), errors="coerce"
                )

                if (
                    pd.notna(filed)
                    and pd.notna(period_end)
                    and filed < period_end
                ):

                    flagged_records.append({
                        "metric": metric_name,
                        "unit": unit,
                        "period_end": period_end.date(),
                        "filed": filed.date(),
                        "form": record.get("form"),
                        "accession": record.get("accn")
                    })

    if flagged_records:
        print(pd.DataFrame(flagged_records).to_string(index=False))
    else:
        print("No unusual filing dates found.")

print("\nSEC filing date investigation completed!")

print("\nSEC FILING CUTOFF VALIDATION")

cutoff_date = pd.Timestamp("2026-10-30")

for file in sorted(sec_folder.glob("*.json")):

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    company_name = data.get("entityName")

    us_gaap = data.get("facts", {}).get("us-gaap", {})

    total_records = 0
    after_cutoff = 0
    on_or_before_cutoff = 0

    for metric_data in us_gaap.values():

        for observations in metric_data.get("units", {}).values():

            for record in observations:

                filed_date = pd.to_datetime(
                    record.get("filed"), errors="coerce"
                )

                if pd.isna(filed_date):
                    continue

                total_records += 1

                if filed_date > cutoff_date:
                    after_cutoff += 1
                else:
                    on_or_before_cutoff += 1

    print(f"\nCompany: {company_name}")
    print("Total dated records:", total_records)
    print("Filed on/before cutoff:", on_or_before_cutoff)
    print("Filed after cutoff:", after_cutoff)

print("\nSEC filing cutoff validation completed!")

print("\nSEC CUTOFF FILTERING")

SEC_OUTPUT = PROJECT_ROOT / "data" / "processed" / "sec"
SEC_OUTPUT.mkdir(parents=True, exist_ok=True)

cutoff_date = pd.Timestamp("2026-10-30")

for file in sorted(sec_folder.glob("*.json")):

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    removed_records = 0

    for taxonomy, metrics in data.get("facts", {}).items():
        for metric_name, metric_data in metrics.items():
            for unit, observations in metric_data.get("units", {}).items():

                filtered = []

                for record in observations:
                    filed_date = pd.to_datetime(
                        record.get("filed"), errors="coerce"
                    )

                    if pd.notna(filed_date) and filed_date <= cutoff_date:
                        filtered.append(record)
                    else:
                        removed_records += 1

                metric_data["units"][unit] = filtered

    output_file = SEC_OUTPUT / file.name

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"\nCompany: {data.get('entityName')}")
    print("Excluded records:", removed_records)
    print("Saved to:", output_file)

print("\nSEC cutoff filtering completed!")

print("\nPROCESSED SEC DATA VERIFICATION")

cutoff_date = pd.Timestamp("2026-10-30")

for file in sorted(SEC_OUTPUT.glob("*.json")):

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_records = 0
    future_records = 0
    missing_or_invalid_dates = 0

    for taxonomy, metrics in data.get("facts", {}).items():

        for metric_data in metrics.values():

            for observations in metric_data.get("units", {}).values():

                for record in observations:

                    total_records += 1

                    filed_date = pd.to_datetime(
                        record.get("filed"),
                        errors="coerce"
                    )

                    if pd.isna(filed_date):
                        missing_or_invalid_dates += 1

                    elif filed_date > cutoff_date:
                        future_records += 1

    print(f"\nCompany: {data.get('entityName')}")
    print("Remaining records:", total_records)
    print("Records after cutoff:", future_records)
    print("Missing/invalid filing dates:", missing_or_invalid_dates)

    if future_records == 0 and missing_or_invalid_dates == 0:
        print("Status: PASSED")
    else:
        print("Status: NEEDS REVIEW")

print("\nProcessed SEC verification completed!")