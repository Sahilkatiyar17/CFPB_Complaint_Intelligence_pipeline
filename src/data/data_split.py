import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

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
# Load Data
# ---------------------------------------------------

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info("Preprocessed data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


# ---------------------------------------------------
# Split Data
# ---------------------------------------------------

def split_data(df, text_column, target_column, test_size, val_size):
    """
    Split into:
    (1 - test_size) = 80% Train
    test_size * (1 - val_size) = 10% Test
    test_size * val_size = 10% Validation
    """

    X = df[text_column]
    y = df[target_column]

    # --------------------------------
    # Step 1: Train (80%) + Temp (20%)
    # --------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,      # 0.20 from params.yaml
        stratify=y,
        random_state=42
    )

    # ----------------------------------------
    # Step 2: Temp → Validation (10%) + Test (10%)
    # ----------------------------------------

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_size,       # 0.50 from params.yaml
        stratify=y_temp,
        random_state=42
    )

    logging.info("Train / Validation / Test split completed.")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------
# Save Split
# ---------------------------------------------------

def save_split(X, y, file_path, text_column, target_column):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df = pd.DataFrame({
        text_column: X,
        target_column: y
    })
    df.to_csv(file_path, index=False)
    logging.info(f"Saved {file_path}")


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    try:
        params = load_params("params.yaml")

        input_path = params["data_split"]["input_path"]
        train_output = params["data_split"]["train_output"]
        val_output = params["data_split"]["val_output"]
        test_output = params["data_split"]["test_output"]
        test_size = params["data_split"]["test_size"]    # 0.20
        val_size = params["data_split"]["val_size"]      # 0.50
        text_column = params["data"]["text_column"]
        target_column = params["data"]["target_column"]

        df = load_data(input_path)

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df,
            text_column=text_column,
            target_column=target_column,
            test_size=test_size,
            val_size=val_size
        )

        save_split(X_train, y_train, train_output, text_column, target_column)
        save_split(X_val, y_val, val_output, text_column, target_column)
        save_split(X_test, y_test, test_output, text_column, target_column)

        logging.info("Data splitting completed successfully.")

    except Exception as e:
        logging.error(e)
        raise


if __name__ == "__main__":
    main()