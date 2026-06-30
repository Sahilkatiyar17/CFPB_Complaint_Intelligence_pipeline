import os
import pandas as pd
import yaml 

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
        logging.info(f"Loaded {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise



# ---------------------------------------------------
# Random Undersample
# ---------------------------------------------------

def random_undersample(train_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    try:
        yes_df = train_df[train_df[target_column] == "Yes"]
        no_df = train_df[train_df[target_column] == "No"]

        logging.info(
            f"Before balancing:\n{train_df[target_column].value_counts()}"
        )

        yes_under = yes_df.sample(n=len(no_df), random_state=42)

        balanced_df = pd.concat([yes_under, no_df])

        balanced_df = balanced_df.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

        logging.info(
            f"After balancing:\n{balanced_df[target_column].value_counts()}"
        )

        return balanced_df

    except Exception as e:
        logging.error(f"Balancing failed: {e}")
        raise



# ---------------------------------------------------
# Save Data
# ---------------------------------------------------

def save_data(train_df, val_df, test_df, output_folder):
    try:
        os.makedirs(output_folder, exist_ok=True)

        train_df.to_csv(
            os.path.join(output_folder, "train_balanced.csv"),
            index=False
        )

        val_df.to_csv(
            os.path.join(output_folder, "validation.csv"),
            index=False
        )

        test_df.to_csv(
            os.path.join(output_folder, "test.csv"),
            index=False
        )

        logging.info("Balanced datasets saved.")

    except Exception as e:
        logging.error(f"Saving failed: {e}")
        raise



# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    try:
        params = load_params("params.yaml")

        train_input = params["data_balancing"]["train_input"]
        val_input = params["data_balancing"]["val_input"]
        test_input = params["data_balancing"]["test_input"]
        output_folder = params["data_balancing"]["output_folder"]
        target_column = params["data"]["target_column"]

        train_df = load_data(train_input)
        val_df = load_data(val_input)
        test_df = load_data(test_input)

        balanced_train = random_undersample(train_df, target_column)

        save_data(balanced_train, val_df, test_df, output_folder)

        logging.info("Data balancing completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

