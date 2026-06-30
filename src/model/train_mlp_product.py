import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

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
# MLP Architecture
# ---------------------------------------------------

class MLPClassifier(nn.Module):
    """
    384 -> 412 (ReLU, Dropout) -> 256 (ReLU, Dropout) -> num_classes
    """

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
# Build DataLoader
# ---------------------------------------------------

def build_dataloader(X, y, batch_size, shuffle):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------
# Train Loop
# ---------------------------------------------------
def train_model(
    X_train, y_train,
    X_val, y_val,
    mlp_params,
    num_classes,
    device
):
    try:
        input_dim = mlp_params["embedding_dimension"]
        hidden1, hidden2 = map(int, mlp_params["hidden_layers"].split("-"))
        dropout = mlp_params["dropout"]
        lr = mlp_params["learning_rate"]
        batch_size = mlp_params["batch_size"]
        epochs = mlp_params["epochs"]

        model = MLPClassifier(
            input_dim=input_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            num_classes=num_classes,
            dropout=dropout
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_loader = build_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader = build_dataloader(X_val, y_val, batch_size, shuffle=False)

        train_size = len(X_train)   # ✅ fixed
        val_size = len(X_val)       # ✅ fixed

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(epochs):

            # ---------------- Training ----------------
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= train_size   # ✅ fixed

            # ---------------- Validation ----------------
            model.eval()
            val_loss = 0.0
            correct = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item() * X_batch.size(0)

                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == y_batch).sum().item()

            val_loss /= val_size               # ✅ fixed
            val_accuracy = correct / val_size  # ✅ fixed

            # ---------------- Save Best Model ----------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

            if (epoch + 1) % 25 == 0 or epoch == 0:
                logging.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_accuracy:.4f}"
                )

        # ✅ fixed — safety check before load_state_dict
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        else:
            logging.warning("No best model state found; using final epoch weights.")

        logging.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")

        return model

    except Exception as e:
        logging.error(f"MLP training failed: {e}")
        raise

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------

def save_model(model, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(model.state_dict(), output_path)
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

        input_folder = params["train_product"]["input_folder"]
        output_path = params["train_product"]["output_path"]
        mlp_params = params["train_product"]
        num_classes = params["train_product"]["num_classes"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        X_train = load_numpy(os.path.join(input_folder, "X_train_product.npy"))
        y_train = load_numpy(os.path.join(input_folder, "y_train_product.npy"))

        X_val = load_numpy(os.path.join(input_folder, "X_validation_product.npy"))
        y_val = load_numpy(os.path.join(input_folder, "y_validation_product.npy"))

        model = train_model(
            X_train, y_train,
            X_val, y_val,
            mlp_params,
            num_classes,
            device
        )

        save_model(model, output_path)

        logging.info("Product MLP training pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()