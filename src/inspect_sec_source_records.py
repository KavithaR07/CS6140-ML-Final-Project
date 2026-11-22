import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEC_DIR = ROOT / "data" / "processed" / "sec"

TARGETS = {
    "TSLA": [
        ("ProceedsFromIssuanceOfLongTermDebt", "2012-05-31", "2012-05-10"),
        ("LineOfCreditFacilityInterestRateDuringPeriod", "2012-05-31", "2012-05-10"),
        ("ProceedsFromIssuanceOfWarrants", "2022-03-31", "2022-02-07"),
    ],
    "PEP": [
        ("OtherPostretirementBenefitsPayments", "2011-12-24", "2011-02-18"),
        ("DefinedBenefitPlanHealthCareCostTrendRateAssumedForNextFiscalYear", "2011-12-24", "2011-02-18"),
        ("PensionContributions", "2011-12-24", "2011-02-18"),
        ("PensionContributions", "2011-12-24", "2011-03-31"),
        ("DefinedBenefitPlanHealthCareCostTrendRateAssumedForNextFiscalYear", "2011-12-24", "2011-03-31"),
        ("OtherPostretirementBenefitsPayments", "2011-12-24", "2011-03-31"),
        ("DefinedBenefitPlansEstimatedFutureEmployerContributionsInNextFiscalYear", "2014-12-27", "2014-02-14"),
    ],
}

print("\nSEC SOURCE-LEVEL RECORD INSPECTION")
print("=" * 65)

for ticker, targets in TARGETS.items():

    source_file = SEC_DIR / f"{ticker}.json"

    with open(source_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    facts = data.get("facts", {}).get("us-gaap", {})

    print(f"\n{'=' * 65}")
    print(f"COMPANY: {ticker}")
    print("=" * 65)

    for metric, period_end, filed in targets:

        print(f"\nMetric: {metric}")
        print(f"Period end: {period_end}")
        print(f"Filed: {filed}")

        metric_data = facts.get(metric)

        if metric_data is None:
            print("Metric not found in source JSON.")
            continue

        matches = []

        for unit, observations in metric_data.get("units", {}).items():

            for obs in observations:

                if (
                    obs.get("end") == period_end
                    and obs.get("filed") == filed
                ):
                    matches.append({
                        "unit": unit,
                        "value": obs.get("val"),
                        "start": obs.get("start"),
                        "end": obs.get("end"),
                        "filed": obs.get("filed"),
                        "form": obs.get("form"),
                        "accession": obs.get("accn"),
                        "fiscal_year": obs.get("fy"),
                        "fiscal_period": obs.get("fp"),
                        "frame": obs.get("frame"),
                    })

        if not matches:
            print("No matching source observations found.")
        else:
            for record in matches:
                print(record)

print("\nSOURCE INSPECTION COMPLETED")