"""
Data Preparation Script
------------------------
Loads the raw dataset from the repository's `tourism_project/data/` folder,
cleans it, drops columns that are not useful predictors, and splits it into
train/test sets. The splits are written to the repo-root as CSV files so the
`data-prep` GitHub Actions job can publish them as a workflow artifact
(`data-splits`) for the next job (`model-training`) to consume.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns that carry no predictive signal (row index / unique identifier).
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return a tidy copy of the dataframe."""
    df = df.copy()

    # 1) Drop the unnamed pandas index column (written when the CSV was
    #    originally exported) and the CustomerID identifier column - neither
    #    carries predictive information and including CustomerID would leak
    #    a unique-per-row key into the model.
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # 2) Fix known inconsistent category labels found in this dataset.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # 3) Remove exact duplicate rows, if any.
    df = df.drop_duplicates()

    # 4) Impute missing values defensively (median for numeric columns,
    #    mode for categorical columns) so the pipeline is robust even if a
    #    future data refresh introduces missing values.
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Raw shape           : {df.shape}")

    df_clean = clean_data(df)
    print(f"Clean shape         : {df_clean.shape}")
    print(f"Columns dropped     : {DROP_COLUMNS}")

    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]

    # Stratified split preserves the ~19/81 class balance of ProdTaken in
    # both the train and test sets.
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Training shape:", train.shape)
    print("Training shape:", test.shape)

    # Saved at the repo root - this matches the artifact paths declared in
    # .github/workflows/pipeline.yml for the "data-splits" artifact.
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"Train / test rows   : {len(Xtrain)} / {len(Xtest)}")
    print("Train target balance:")
    print(ytrain.value_counts(normalize=True).round(3))
    print("Test target balance :")
    print(ytest.value_counts(normalize=True).round(3))
    print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
