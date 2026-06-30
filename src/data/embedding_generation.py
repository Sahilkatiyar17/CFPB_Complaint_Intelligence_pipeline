import os
import pickle
import yaml
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

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
# Load Embedding Model
# ---------------------------------------------------

def load_embedding_model(model_name: str):
    try:
        logging.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        logging.info("Embedding model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading embedding model: {e}")
        raise


# ---------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------

def generate_embeddings(model, texts, batch_size):
    try:
        embeddings = model.encode(
            texts.tolist(),
            batch_size=batch_size,
            show_progress_bar=True
        )
        return embeddings
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        raise


# ---------------------------------------------------
# Encode Labels
# ---------------------------------------------------

def encode_labels(train_labels, val_labels, test_labels):
    try:
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(train_labels)
        y_validation = encoder.transform(val_labels)
        y_test = encoder.transform(test_labels)
        logging.info("Labels encoded successfully.")
        return y_train, y_validation, y_test, encoder
    except Exception as e:
        logging.error(f"Label encoding failed: {e}")
        raise


# ---------------------------------------------------
# Save Numpy
# ---------------------------------------------------

def save_numpy(array, file_path):
    np.save(file_path, array)


# ---------------------------------------------------
# Process One Branch (urgency OR product)
# ---------------------------------------------------

def process_branch(
    suffix,
    target_column,
    embedding_model,
    text_column,
    balanced_folder,
    output_folder,
    train_batch_size,
    val_batch_size,
    test_batch_size,
    encoder_output_path
):
    logging.info(f"Processing branch: {suffix}")

    train_df = load_data(
        os.path.join(balanced_folder, f"train_balanced_{suffix}.csv")
    )

    val_df = load_data(
        os.path.join(balanced_folder, f"validation_{suffix}.csv")
    )

    test_df = load_data(
        os.path.join(balanced_folder, f"test_{suffix}.csv")
    )

    X_train = generate_embeddings(
        embedding_model, train_df[text_column], batch_size=train_batch_size
    )

    X_validation = generate_embeddings(
        embedding_model, val_df[text_column], batch_size=val_batch_size
    )

    X_test = generate_embeddings(
        embedding_model, test_df[text_column], batch_size=test_batch_size
    )

    y_train, y_validation, y_test, label_encoder = encode_labels(
        train_df[target_column],
        val_df[target_column],
        test_df[target_column]
    )

    os.makedirs(output_folder, exist_ok=True)

    save_numpy(X_train, os.path.join(output_folder, f"X_train_{suffix}.npy"))
    save_numpy(X_validation, os.path.join(output_folder, f"X_validation_{suffix}.npy"))
    save_numpy(X_test, os.path.join(output_folder, f"X_test_{suffix}.npy"))
    save_numpy(y_train, os.path.join(output_folder, f"y_train_{suffix}.npy"))
    save_numpy(y_validation, os.path.join(output_folder, f"y_validation_{suffix}.npy"))
    save_numpy(y_test, os.path.join(output_folder, f"y_test_{suffix}.npy"))

    with open(encoder_output_path, "wb") as file:
        pickle.dump(label_encoder, file)

    logging.info(f"Embeddings and label encoder saved for {suffix}.")


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    try:
        params = load_params("params.yaml")

        text_column = params["data"]["text_column"]
        urgency_target = params["data"]["target_column"]
        product_target = params["data"]["product_target_column"]

        model_name = params["embedding"]["model_name"]
        train_batch_size = params["embedding"]["train_batch_size"]
        val_batch_size = params["embedding"]["val_batch_size"]
        test_batch_size = params["embedding"]["test_batch_size"]
        output_folder = params["embedding"]["output_folder"]

        balanced_folder = params["data_balancing"]["output_folder"]

        embedding_model = load_embedding_model(model_name)

        # ------------------------------------------------
        # Urgency branch
        # ------------------------------------------------

        process_branch(
            suffix="urgency",
            target_column=urgency_target,
            embedding_model=embedding_model,
            text_column=text_column,
            balanced_folder=balanced_folder,
            output_folder=output_folder,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            encoder_output_path="./models/label_encoder_urgency.pkl"
        )

        # ------------------------------------------------
        # Product branch
        # ------------------------------------------------

        process_branch(
            suffix="product",
            target_column=product_target,
            embedding_model=embedding_model,
            text_column=text_column,
            balanced_folder=balanced_folder,
            output_folder=output_folder,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            encoder_output_path="./models/label_encoder_product.pkl"
        )

        logging.info("Embedding generation completed successfully for both branches.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
