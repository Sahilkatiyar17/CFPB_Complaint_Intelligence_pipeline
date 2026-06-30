import json
import yaml
import mlflow
import dagshub

from mlflow import MlflowClient

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


def load_model_info(path):

    with open(path, "r") as f:

        return json.load(f)


def register_model(model_name, logged_model_id):

    model_uri = f"models:/{logged_model_id}"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    logging.info(
        f"Model registered successfully. Version: {result.version}"
    )

    return result.version


def assign_alias(model_name, version):

    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version
    )

    logging.info(
        f"'champion' alias assigned to version {version}"
    )


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    try:
        params = load_params("params.yaml")

        experiment_info_path = params["model_evaluation"]["experiment_info_path"]
        model_name = params["register_model"]["model_name"]

        # ------------------------------------------------
        # MLflow Setup
        # ------------------------------------------------

        dagshub.init(
            repo_owner="",
            repo_name="",
            mlflow=True
        )

        logging.info("Loading experiment information...")

        model_info = load_model_info(experiment_info_path)

        logged_model_id = model_info["logged_model_id"]

        version = register_model(
            model_name=model_name,
            logged_model_id=logged_model_id
        )

        assign_alias(
            model_name=model_name,
            version=version
        )

        logging.info("Model registration completed successfully.")

    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    main()