import os
import yaml
import pandas as pd

from sklearn.model_selection import train_test_split

from src.logger import logging


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully.")
        return params
    except Exception as e:
        logging.error(f"Error loading params.yaml: {e}")
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info("Preprocessed data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def split_data(df, target_column, test_size, val_size):
    """
    Splits the FULL dataframe (keeping all columns, including product_5)
    so downstream stages can access whichever target they need.
    """

    # Stratify on urgency target column as before
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        stratify=temp_df[target_column],
        random_state=42
    )

    logging.info("Train / Validation / Test split completed.")

    return train_df, val_df, test_df


def save_split(df, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logging.info(f"Saved {file_path}")


def main():
    try:
        params = load_params("params.yaml")

        input_path = params["data_split"]["input_path"]
        train_output = params["data_split"]["train_output"]
        val_output = params["data_split"]["val_output"]
        test_output = params["data_split"]["test_output"]
        test_size = params["data_split"]["test_size"]
        val_size = params["data_split"]["val_size"]
        target_column = params["data"]["target_column"]

        df = load_data(input_path)

        train_df, val_df, test_df = split_data(
            df,
            target_column=target_column,
            test_size=test_size,
            val_size=val_size
        )

        save_split(train_df, train_output)
        save_split(val_df, val_output)
        save_split(test_df, test_output)

        logging.info("Data splitting completed successfully.")

    except Exception as e:
        logging.error(e)
        raise


if __name__ == "__main__":
    main()
