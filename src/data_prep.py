"""
Data Preparation
-----------------
Loads the raw dataset, cleans it, splits it into train/test sets,
and saves both locally.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/tourism.csv"
OUT_DIR = "data/processed"
TARGET = "ProdTaken"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["CustomerID", "Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=col)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def main():
    df = pd.read_csv(RAW_PATH, index_col=0)
    df_clean = clean_data(df)

    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    X_train.to_csv(f"{OUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUT_DIR}/y_test.csv", index=False)

    print(f"Clean shape        : {df_clean.shape}")
    print(f"Train / test rows  : {len(X_train)} / {len(X_test)}")
    print(f"Saved processed splits to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()