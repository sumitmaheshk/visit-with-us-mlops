"""
Model Training Script (with MLflow Experimentation Tracking)
--------------------------------------------------------------
Loads the train/test splits produced by `prep.py` (delivered to this job via
the `data-splits` workflow artifact), builds a preprocessing + XGBoost
pipeline, tunes it with GridSearchCV, logs every trial's hyperparameters and
score plus the final chosen model to MLflow, evaluates it on the held-out
test set, and saves the best pipeline to `tourism_project/deployment/` so the
`Commit Trained Model` step of the workflow can commit it back to the repo.

MLflow is pointed at a local, file-based tracking store (`./mlruns`) rather
than a live server. This keeps the GitHub Actions job self-contained (no
server process / networking to manage) while still producing a complete,
inspectable MLflow experiment log, which the workflow uploads as a
`mlflow-tracking` artifact for review.
"""

import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# --- Feature groups ----------------------------------------------------
CATEGORICAL = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]
NUMERIC = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = f"{MODEL_DIR}/best_model.joblib"

# File-based MLflow tracking store, kept local to the job's working
# directory so runs are captured without needing a live tracking server.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"file:{os.path.join(os.getcwd(), 'mlruns')}"
)
EXPERIMENT_NAME = "tourism-wellness-package"


def build_pipeline() -> "make_pipeline":
    """Preprocessing (one-hot + scaling) chained with an XGBoost classifier."""
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        (StandardScaler(), NUMERIC),
    )
    xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    return make_pipeline(preprocessor, xgb_model)


def main() -> None:
    # --- Load the artifact produced by prep.py --------------------------
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    print(f"Loaded Xtrain {Xtrain.shape}, Xtest {Xtest.shape}")

    # --- Define the model and the hyperparameter grid to tune -----------
    pipe = build_pipeline()
    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )

    # --- Set up MLflow experiment tracking -------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="grid_search_best_model") as parent_run:
        print(f"MLflow run id: {parent_run.info.run_id}")

        # Tune the model.
        search.fit(Xtrain, ytrain)

        # Log every hyperparameter combination that was tried as its own
        # nested run, so the full tuning history is captured in MLflow.
        for i, params in enumerate(search.cv_results_["params"]):
            with mlflow.start_run(nested=True, run_name=f"trial_{i}"):
                mlflow.log_params(params)
                mlflow.log_metric(
                    "mean_cv_f1", float(search.cv_results_["mean_test_score"][i])
                )

        best_model = search.best_estimator_
        ypred = best_model.predict(Xtest)

        metrics = {
            "accuracy": accuracy_score(ytest, ypred),
            "precision": precision_score(ytest, ypred),
            "recall": recall_score(ytest, ypred),
            "f1": f1_score(ytest, ypred),
        }

        print("Best params:", search.best_params_)
        print("Best CV f1 :", round(search.best_score_, 4))
        print("Test metrics:", {k: round(v, 4) for k, v in metrics.items()})
        print(classification_report(ytest, ypred))

        # Log the winning hyperparameters, metrics, and the fitted pipeline
        # (preprocessing + model) as a single MLflow model artifact.
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_f1", search.best_score_)
        mlflow.log_metrics(metrics)
        # serialization_format="cloudpickle" avoids MLflow's default
        # skops-based serializer, which does not support XGBoost's
        # custom booster objects out of the box.
        mlflow.sklearn.log_model(
            best_model, name="model", serialization_format="cloudpickle"
        )

    # --- Save the best model so the workflow can commit it to the repo ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
