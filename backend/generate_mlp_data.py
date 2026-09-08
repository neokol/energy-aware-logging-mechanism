"""
Scenario A – MLP dataset preparation.

Builds a real predictive-maintenance classification dataset (AI4I 2020,
UCI id 601) instead of random noise, so that the FP32 vs INT8 accuracy
comparison in Scenario A is meaningful.

Output (written next to this script, in backend/):
    maintenance_train.csv  -> used by setup_models.py to train the MLP
    maintenance_test.csv   -> uploaded through the UI and used for inference

Both CSVs are already pre-processed (categorical one-hot + numeric scaling),
because the inference path (app/services/mlp_service.py) feeds raw CSV values
straight into the network without any further transformation.
"""

import io
import os
import ssl
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
RAW_PATH = os.path.join("data", "ai4i2020.csv")

TARGET = "Machine failure"
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
# Identifiers + the five per-failure-mode flags (TWF/HDF/PWF/OSF/RNF).
# The flags are components of the target itself -> dropping them avoids label leakage.
DROP_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

OUT_COLUMNS = [
    "label",
    "air_temp",
    "process_temp",
    "rotational_speed",
    "torque",
    "tool_wear",
    "power",       # torque * angular speed  -> drives PWF (power failure)
    "temp_diff",   # process - air temperature -> drives HDF (heat dissipation failure)
    "type_H",
    "type_L",
    "type_M",
]


def _load_raw() -> pd.DataFrame:
    if not os.path.exists(RAW_PATH):
        os.makedirs("data", exist_ok=True)
        print("Downloading AI4I 2020 dataset from UCI ...")
        ssl._create_default_https_context = ssl._create_unverified_context
        with urllib.request.urlopen(RAW_URL) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            with z.open("ai4i2020.csv") as f:
                df = pd.read_csv(f)
        df.columns = [c.encode("ascii", "ignore").decode() for c in df.columns]  # strip BOM
        df.to_csv(RAW_PATH, index=False)
        print(f"Saved raw copy to {RAW_PATH}")
    return pd.read_csv(RAW_PATH)


def create_mlp_dataset():
    df = _load_raw()

    y = df[TARGET].astype(int)
    X = df.drop(columns=DROP_COLUMNS + [TARGET])

    # One-hot the machine 'Type' (L / M / H)
    type_dummies = pd.get_dummies(X["Type"], prefix="type").astype(int)
    for col in ("type_H", "type_L", "type_M"):
        if col not in type_dummies.columns:
            type_dummies[col] = 0
    type_dummies = type_dummies[["type_H", "type_L", "type_M"]]

    X_num = X[NUMERIC_FEATURES].copy()
    # Engineered features (same physical quantities AI4I uses to define failures)
    X_num["power"] = X["Torque [Nm]"] * X["Rotational speed [rpm]"] * (2 * np.pi / 60.0)
    X_num["temp_diff"] = X["Process temperature [K]"] - X["Air temperature [K]"]
    X_num = X_num.astype(np.float32)

    X_train_num, X_test_num, d_train, d_test, y_train, y_test = train_test_split(
        X_num, type_dummies, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_num).astype(np.float32)

    n_numeric = X_num.shape[1]

    def _assemble(x_scaled, dummies, labels) -> pd.DataFrame:
        out = pd.DataFrame(x_scaled, columns=OUT_COLUMNS[1:1 + n_numeric])
        dummies = dummies.reset_index(drop=True)
        for col in ("type_H", "type_L", "type_M"):
            out[col] = dummies[col].values
        out.insert(0, "label", labels.reset_index(drop=True).astype(int))
        return out[OUT_COLUMNS]

    train_df = _assemble(X_train_scaled, d_train, y_train)
    test_df = _assemble(X_test_scaled, d_test, y_test)

    train_df.to_csv("maintenance_train.csv", index=False)
    test_df.to_csv("maintenance_test.csv", index=False)

    print("Created maintenance_train.csv and maintenance_test.csv")
    print(f"  train: {train_df.shape}  positives: {int(train_df['label'].sum())} "
          f"({train_df['label'].mean() * 100:.2f}%)")
    print(f"  test:  {test_df.shape}  positives: {int(test_df['label'].sum())} "
          f"({test_df['label'].mean() * 100:.2f}%)")


if __name__ == "__main__":
    create_mlp_dataset()
