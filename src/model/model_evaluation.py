import os
import json
import yaml
import numpy as np
import mlflow
import dagshub 
import mlflow.xgboost as mlflow_xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

from src.logger import logging

# ---------------------------------------------------
# Load Params
# ---------------------------------------------------

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully.")
        return params
    except Exception as e:
        logging.error(f"Error loading params.yaml: {e}")
        raise

# ---------------------------------------------------
# Load Numpy
# ---------------------------------------------------

def load_numpy(path):
    return np.load(path)


# ---------------------------------------------------
# Save Metrics
# ---------------------------------------------------

def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
 
# ---------------------------------------------------
# Save Model Info
# ---------------------------------------------------

def save_model_info(run_id, model_path, file_path, logged_model_id):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    model_info = {
        "run_id": run_id,
        "model_path": model_path,
        "logged_model_id": logged_model_id
    }
    with open(file_path, "w") as f:
        json.dump(model_info, f, indent=4)

# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    try:
        params = load_params("params.yaml")

        model_path = params["model_evaluation"]["model_path"]
        embeddings_folder = params["model_evaluation"]["embeddings_folder"]
        metrics_path = params["model_evaluation"]["metrics_path"]
        experiment_info_path = params["model_evaluation"]["experiment_info_path"]
        train_params = params["train"]

        # ------------------------------------------------
        # MLflow Setup
        # ------------------------------------------------

        dagshub.init(
            repo_owner="",
            repo_name="",
            mlflow=True
        )

        # ------------------------------------------------
        # Load Model
        # ------------------------------------------------

        logging.info("Loading trained model...")

        model = XGBClassifier()
        model.load_model(model_path)

        # ------------------------------------------------
        # Load Embeddings
        # ------------------------------------------------

        logging.info("Loading test embeddings...")

        X_test = load_numpy(os.path.join(embeddings_folder, "X_test.npy"))
        y_test = load_numpy(os.path.join(embeddings_folder, "y_test.npy"))

        # ------------------------------------------------
        # Predictions
        # ------------------------------------------------

        logging.info("Making predictions...")

        y_pred = model.predict(X_test)

        # ------------------------------------------------
        # Metrics
        # ------------------------------------------------

        accuracy = float(accuracy_score(y_test, y_pred))

        precision = float(precision_score(
            y_test, y_pred, average="weighted"
        ))

        recall = float(recall_score(
            y_test, y_pred, average="weighted"
        ))

        f1 = float(f1_score(
            y_test, y_pred, average="weighted"
        ))

        report = classification_report(
            y_test, y_pred, output_dict=True
        )

        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report
        }

        save_metrics(metrics, metrics_path)

        logging.info("Metrics saved.")

        # ------------------------------------------------
        # MLflow Logging
        # ------------------------------------------------

        mlflow.set_experiment("training.pipeline")

        with mlflow.start_run(run_name="Final_XGBoost") as run:

            mlflow.log_params({
                "n_estimators": train_params["n_estimators"],
                "max_depth": train_params["max_depth"],
                "learning_rate": train_params["learning_rate"],
                "subsample": train_params["subsample"],
                "colsample_bytree": train_params["colsample_bytree"],
                "reg_alpha": train_params["reg_alpha"],
                "reg_lambda": train_params["reg_lambda"]
            })

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.log_artifact(metrics_path)

            logged_model = mlflow_xgb.log_model(
                model,
                name="xgboost_model"
            )

            logging.info("Model logged successfully.")

            save_model_info(
                run.info.run_id,
                "xgboost_model",
                experiment_info_path,
                logged_model.model_id
            )

        logging.info("Evaluation completed successfully.")

    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    main()