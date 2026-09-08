"""
Scenario A - CNN dataset preparation.

Exports the MNIST test split (10,000 images) to a labelled CSV so the
inference path can compute real accuracy instead of a hard-coded value.

Output (written next to this script, in backend/):
    mnist_test.csv   -> 785 columns: label + pixel0..pixel783, raw 0-255

Pixels are left un-normalised on purpose; app/services/cnn_service.py applies
the same normalisation used during training.
"""

import numpy as np
import pandas as pd
from torchvision import datasets

OUT_PATH = "mnist_test.csv"


def create_cnn_dataset():
    print("Loading MNIST test split ...")
    test = datasets.MNIST("./data", train=False, download=True)

    images = test.data.numpy().reshape(len(test), -1).astype(np.int64)  # (10000, 784), 0-255
    labels = test.targets.numpy().astype(np.int64)

    columns = ["label"] + [f"pixel{i}" for i in range(images.shape[1])]
    df = pd.DataFrame(np.column_stack([labels, images]), columns=columns)
    df.to_csv(OUT_PATH, index=False)

    print(f"Created {OUT_PATH}  shape={df.shape}")
    print(f"  class distribution: {np.bincount(labels).tolist()}")


if __name__ == "__main__":
    create_cnn_dataset()
