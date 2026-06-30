import os
import yaml
import numpy as np

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

def load_numpy(file_path: str):
    try:
        data = np.load(file_path)
        logging.info(f"Loaded {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise


# ---------------------------------------------------
# Train Model
# ---------------------------------------------------

def train_model(X_train, y_train, params):
    try:
        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        logging.info("Final XGBoost (urgency) model trained successfully.")

        return model

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


# ---------------------------------------------------
# Save Model
# ---------------------------------------------------

def save_model(model, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.save_model(output_path)
        logging.info(f"Model saved to {output_path}")
    except Exception as e:
        logging.error(f"Saving failed: {e}")
        raise


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    try:
        params = load_params("params.yaml")

        input_folder = params["train"]["input_folder"]
        output_path = params["train"]["output_path"]
        train_params = params["train"]

        X_train = load_numpy(
            os.path.join(input_folder, "X_train_urgency.npy")
        )

        y_train = load_numpy(
            os.path.join(input_folder, "y_train_urgency.npy")
        )

        model = train_model(X_train, y_train, train_params)

        save_model(model, output_path)

        logging.info("Urgency training pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
