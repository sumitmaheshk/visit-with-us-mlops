import pandas as pd

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

DATA_PATH = "tourism_project/data/tourism.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    if df.empty:
        raise ValueError("Dataset has 0 rows.")

    print("=" * 60)
    print("DATA REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Rows x Columns      : {df.shape[0]} x {df.shape[1]}")
    print(f"Target balance      :\n{df['ProdTaken'].value_counts(normalize=True).round(3)}")
    print(f"Missing values/col  :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Duplicate rows      : {df.duplicated().sum()}")
    print("All expected columns present. Validation PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
