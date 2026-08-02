import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
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

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df_clean = clean_data(df)

    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]

    # Saved at the repo root (not a subfolder) -- the pipeline.yml artifact
    # paths for this job expect Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv
    # at the top level.
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"Clean shape        : {df_clean.shape}")
    print(f"Train / test rows  : {len(Xtrain)} / {len(Xtest)}")
    print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
