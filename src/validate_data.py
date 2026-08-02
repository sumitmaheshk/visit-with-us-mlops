"""
Data Registration & Validation
--------------------------------
Checks that the raw tourism dataset has the expected schema before it
enters the pipeline, and prints a summary report.
"""

import sys
import pandas as pd

DATA_PATH = "data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def validate(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    if df.empty:
        raise ValueError("Dataset has 0 rows.")

    print("=" * 60)
    print("DATA REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Source file        : {path}")
    print(f"Rows x Columns      : {df.shape[0]} x {df.shape[1]}")
    print(f"Target balance      :\n{df['ProdTaken'].value_counts(normalize=True).round(3)}")
    print(f"Missing values/col  :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Duplicate rows      : {df.duplicated().sum()}")
    print("All expected columns present. Validation PASSED.")
    print("=" * 60)
    return df


if __name__ == "__main__":
    try:
        validate()
    except Exception as e:
        print(f"Validation FAILED: {e}")
        sys.exit(1)