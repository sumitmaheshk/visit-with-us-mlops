"""
Data Registration Script
-------------------------
Reads the raw dataset from the repository's `tourism_project/data/` folder,
validates that every column the downstream pipeline depends on is present,
and prints a short data-quality summary. This script is the first job in
the GitHub Actions pipeline (`register-dataset`), so a broken/incomplete
dataset is caught immediately, before any cleaning or training happens.
"""

import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

# Every column the raw dataset must contain (target + all raw features).
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

TARGET = "ProdTaken"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # --- Schema validation -------------------------------------------------
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        # Fail loudly (non-zero exit code) so the GitHub Actions job goes
        # red and the pipeline stops before touching a bad dataset.
        print(f"VALIDATION FAILED - missing columns: {missing_cols}")
        sys.exit(1)

    if df.empty:
        print("VALIDATION FAILED - dataset has 0 rows.")
        sys.exit(1)

    if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        print("VALIDATION FAILED - target column ProdTaken must be binary (0/1).")
        sys.exit(1)

    # --- Data-quality summary ----------------------------------------------
    print("=" * 60)
    print("DATA REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Rows x Columns        : {df.shape[0]} x {df.shape[1]}")
    print(f"Expected columns found: {len(EXPECTED_COLUMNS)}/{len(EXPECTED_COLUMNS)}")
    print(f"Duplicate rows        : {df.duplicated().sum()}")
    print(f"Duplicate CustomerIDs : {df['CustomerID'].duplicated().sum()}")
    print("\nMissing values per column (non-zero only):")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    print(null_counts if not null_counts.empty else "  None")
    print("\nTarget class balance (ProdTaken):")
    print(df[TARGET].value_counts(normalize=True).round(3))
    print("=" * 60)
    print("Result: All expected columns present. VALIDATION PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
