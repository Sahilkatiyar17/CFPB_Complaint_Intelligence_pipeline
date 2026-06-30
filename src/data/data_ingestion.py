# src/data/data_ingestion.py

import os
import yaml
import logging
import pandas as pd

from src.logger import logging
from src.connections import s3_connection


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully.")
        return params
    except Exception as e:
        logging.error(f"Error loading params.yaml: {e}")
        raise


def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Dataset loaded successfully from {data_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

# Optional
# Uncomment later if you decide to load data from AWS S3

# def load_data_from_s3(bucket_name, access_key, secret_key, file_name):
#
#     s3 = s3_connection.s3_operations(
#         bucket_name,
#         access_key,
#         secret_key
#     )
#
#     return s3.fetch_file_from_s3(file_name)


def validate_data(
    df: pd.DataFrame,
    text_column: str,
    target_column: str
) -> None:
    """
    Perform basic validation checks.
    """

    if df.empty:

        raise ValueError("Dataset is empty.")

    if text_column not in df.columns:

        raise ValueError(
            f"Missing text column: {text_column}"
        )

    if target_column not in df.columns:

        raise ValueError(
            f"Missing target column: {target_column}"
        )

    logging.info("Dataset validation successful.")


def save_raw_data(
    df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Save raw dataset.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Raw dataset saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving raw dataset: {e}")
        raise


def main():

    try:

        params = load_params("params.yaml")

        input_path = params["data_ingestion"]["input_path"]

        output_path = params["data_ingestion"]["output_path"]

        text_column = params["data"]["text_column"]

        target_column = params["data"]["target_column"]

        df = load_data(input_path)

        validate_data(
            df,
            text_column,
            target_column
        )

        save_raw_data(
            df,
            output_path
        )
    #     df = load_data(
    #         "./notebooks/complaints_small.csv"
    #     )

    #     validate_data(
    #         df,
    #         "narrative",
    #         "Timely response?"
    #     )

    #     save_raw_data(
    #         df,
    #         "./data_artifacts/raw/complaints_small.csv"
    #     )


        logging.info(
            "Data ingestion completed successfully."
        )

    except Exception as e:

        logging.error(
            f"Data ingestion failed: {e}"
        )

        raise


if __name__ == "__main__":

    main()