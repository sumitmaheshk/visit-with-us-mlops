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
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
)

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
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def main():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")

    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        (StandardScaler(), NUMERIC),
    )

    xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    pipe = make_pipeline(preprocessor, xgb_model)

    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
    search.fit(Xtrain, ytrain)

    best_model = search.best_estimator_
    ypred = best_model.predict(Xtest)

    metrics = {
        "accuracy": accuracy_score(ytest, ypred),
        "precision": precision_score(ytest, ypred),
        "recall": recall_score(ytest, ypred),
        "f1": f1_score(ytest, ypred),
    }
    print("Best params:", search.best_params_)
    print("Test metrics:", metrics)
    print(classification_report(ytest, ypred))

    # --- Experiment tracking with MLflow ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("tourism-wellness-package")

    with mlflow.start_run(run_name="best_model"):
        # Log every hyperparameter combination GridSearchCV tried as a nested run
        for i, params in enumerate(search.cv_results_["params"]):
            with mlflow.start_run(nested=True, run_name=f"trial_{i}"):
                mlflow.log_params(params)
                mlflow.log_metric("mean_cv_f1", search.cv_results_["mean_test_score"][i])

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_f1", search.best_score_)
        mlflow.log_metrics(metrics)
        # serialization_format="cloudpickle" avoids mlflow's skops-based
        # serializer, which blocks XGBoost's custom types by default.
        mlflow.sklearn.log_model(best_model, "model", serialization_format="cloudpickle")

    # --- Save the best model so the workflow can commit it to the repo ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.joblib")
    print(f"Saved model to {MODEL_DIR}/best_model.joblib")


if __name__ == "__main__":
    main()
