import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import mlflow
import dagshub
import mlflow.xgboost as mlflow_xgb
import mlflow.pytorch as mlflow_pytorch
from typing import Literal
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
# Compute Classification Metrics (shared by both models)
# ---------------------------------------------------

def compute_metrics(
    y_test,
    y_pred,
    average: Literal["micro", "macro", "samples", "weighted", "binary"] = "weighted"
):
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average=average))
    recall = float(recall_score(y_test, y_pred, average=average))
    f1 = float(f1_score(y_test, y_pred, average=average))
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report
    }


# ---------------------------------------------------
# MLP Architecture (must match train_mlp_product.py exactly)
# ---------------------------------------------------

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, num_classes, dropout):
        super(MLPClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------
# Evaluate Urgency Model (XGBoost)
# ---------------------------------------------------

def evaluate_urgency_model(params):

    model_path = params["model_evaluation"]["model_path"]
    embeddings_folder = params["model_evaluation"]["embeddings_folder"]
    metrics_path = params["model_evaluation"]["metrics_path"]
    experiment_info_path = params["model_evaluation"]["experiment_info_path"]
    train_params = params["train"]

    logging.info("Loading trained urgency (XGBoost) model...")

    model = XGBClassifier()
    model.load_model(model_path)

    X_test = load_numpy(os.path.join(embeddings_folder, "X_test_urgency.npy"))
    y_test = load_numpy(os.path.join(embeddings_folder, "y_test_urgency.npy"))

    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    save_metrics(metrics, metrics_path)

    logging.info("Urgency metrics saved.")

    mlflow.set_experiment("training.pipeline")

    with mlflow.start_run(run_name="Final_XGBoost_Urgency2") as run:

        mlflow.log_params({
            "n_estimators": train_params["n_estimators"],
            "max_depth": train_params["max_depth"],
            "learning_rate": train_params["learning_rate"],
            "subsample": train_params["subsample"],
            "colsample_bytree": train_params["colsample_bytree"],
            "reg_alpha": train_params["reg_alpha"],
            "reg_lambda": train_params["reg_lambda"]
        })

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        mlflow.log_artifact(metrics_path)

        logged_model = mlflow_xgb.log_model(model, name="xgboost_model_urgency")

        logging.info("Urgency model logged successfully.")

        save_model_info(
            run.info.run_id,
            "xgboost_model_urgency",
            experiment_info_path,
            logged_model.model_id
        )

    logging.info("Urgency model evaluation completed successfully.")


# ---------------------------------------------------
# Evaluate Product Model (PyTorch MLP)
# ---------------------------------------------------

def evaluate_product_model(params, device):

    model_path = params["train_product"]["output_path"]
    embeddings_folder = params["model_evaluation"]["embeddings_folder"]
    metrics_path = params["model_evaluation_product"]["metrics_path"]
    experiment_info_path = params["model_evaluation_product"]["experiment_info_path"]
    mlp_params = params["train_product"]
    num_classes = params["train_product"]["num_classes"]

    logging.info("Loading trained product (MLP) model...")

    input_dim = mlp_params["embedding_dimension"]
    hidden1, hidden2 = map(int, mlp_params["hidden_layers"].split("-"))
    dropout = mlp_params["dropout"]

    model = MLPClassifier(
        input_dim=input_dim,
        hidden1=hidden1,
        hidden2=hidden2,
        num_classes=num_classes,
        dropout=dropout
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    X_test = load_numpy(os.path.join(embeddings_folder, "X_test_product.npy"))
    y_test = load_numpy(os.path.join(embeddings_folder, "y_test_product.npy"))

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

    metrics = compute_metrics(y_test, y_pred, average="weighted")

    save_metrics(metrics, metrics_path)

    logging.info("Product metrics saved.")

    mlflow.set_experiment("training.pipeline")

    with mlflow.start_run(run_name="Final_MLP_Product2") as run:

        mlflow.log_params({
            "learning_rate": mlp_params["learning_rate"],
            "hidden_layers": mlp_params["hidden_layers"],
            "dropout": mlp_params["dropout"],
            "batch_size": mlp_params["batch_size"],
            "epochs": mlp_params["epochs"],
            "optimizer": "Adam",
            "embedding_dimension": input_dim,
            "num_classes": num_classes
        })

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        mlflow.log_artifact(metrics_path)

        input_example = X_test[:1]  # one sample row, shape (1, 384)

        # ✅ Correct fix
        logged_model = mlflow_pytorch.log_model(
            model,
            name="mlp_model_product",
            input_example=input_example,
            serialization_format="pickle"   # ✅ avoids torch.export/pt2 tracing
            )

        logging.info("Product model logged successfully.")

        save_model_info(
            run.info.run_id,
            "mlp_model_product",
            experiment_info_path,
            logged_model.model_id
        )

    logging.info("Product model evaluation completed successfully.")


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    try:
        params = load_params("params.yaml")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        dagshub.init(
            repo_owner="",
            repo_name="",
            mlflow=True
        )

        evaluate_urgency_model(params)

        evaluate_product_model(params, device)

        logging.info("Evaluation completed successfully for both models.")

    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    main()

