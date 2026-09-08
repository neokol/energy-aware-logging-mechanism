# setup_models.py
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ai_models.mlp import MaintenanceMLP

MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODEL_DIR, "mlp_maintenance_v1.pth")

SEED = 42
INPUT_SIZE = 10
HIDDEN_SIZE = 128
NUM_CLASSES = 2
EPOCHS = 200
PATIENCE = 20
LR = 1e-3
BATCH_SIZE = 256


def _load_xy(path):
    df = pd.read_csv(path)
    y = df["label"].astype(int).values
    X = df.drop(columns=["label"]).values.astype(np.float32)
    return X, y


def _report(name, y_true, y_pred):
    print(f"\n[{name}]")
    print(f"  accuracy          : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  balanced accuracy : {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"  f1 (failure class): {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  confusion matrix  : {confusion_matrix(y_true, y_pred).tolist()}")


def train_and_save_mlp():
    print("--- Training Maintenance MLP (AI4I 2020) ---")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X, y = _load_xy("maintenance_train.csv")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # 'balanced' weights (~15x) over-correct and flood the confusion matrix with
    # false alarms; the square root keeps recall high with far better precision.
    class_weights = np.sqrt(
        compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights: {class_weights.round(3).tolist()}")

    model = MaintenanceMLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    X_tr_t = torch.tensor(X_tr)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 20 == 0 or epochs_without_improvement >= PATIENCE:
            print(f"Epoch {epoch:3d}  val_loss={val_loss:.4f}  best={best_val:.4f}")

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    def _predict(X_t):
        with torch.no_grad():
            return model(torch.tensor(X_t)).argmax(dim=1).numpy()

    _report("validation", y_val, _predict(X_val))

    X_test, y_test = _load_xy("maintenance_test.csv")
    _report("held-out test", y_test, _predict(X_test))

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nModel saved successfully at: {SAVE_PATH}")


if __name__ == "__main__":
    train_and_save_mlp()
