"""
Model Building with Experimentation Tracking
----------------------------------------------
Loads the train/test data, tunes a RandomForestClassifier with
GridSearchCV, logs every trial, evaluates, and saves the best model.
"""

import json
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = "data/processed"
MODEL_DIR = "models"
LOG_PATH = f"{MODEL_DIR}/experiment_log.json"
MODEL_PATH = f"{MODEL_DIR}/best_model.joblib"

CATEGORICAL = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]


def load_splits():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze("columns")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def build_pipeline(numeric_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="passthrough",
    )
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ])
    return pipe


def main():
    X_train, X_test, y_train, y_test = load_splits()
    numeric_cols = [c for c in X_train.columns if c not in CATEGORICAL]

    pipe = build_pipeline(numeric_cols)

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 8, 12],
        "model__min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        pipe, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    print("Best params:", search.best_params_)
    print("Test metrics:", metrics)
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    results = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")

    log = {
        "all_trials": json.loads(results.to_json(orient="records")),
        "best_params": search.best_params_,
        "best_cv_f1": search.best_score_,
        "test_metrics": metrics,
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")
    print(f"Saved experiment log to {LOG_PATH}")


if __name__ == "__main__":
    main()