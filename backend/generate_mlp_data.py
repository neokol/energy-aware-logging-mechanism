import pandas as pd
import numpy as np

def create_mlp_dataset():
    print("Generating MLP-compatible dataset...")
    rows = 5000  
    cols = 512   
    
    
    data = np.random.randn(rows, cols).astype(np.float32)
    labels = np.random.randint(0, 2, size=(rows,))  # binary labels: 0 or 1

    col_names = ["label"] + [f"feature_{i}" for i in range(cols)]
    df = pd.DataFrame(np.column_stack([labels, data]), columns=col_names)
    df["label"] = df["label"].astype(int)

    filename = "maintenance_data.csv"
    df.to_csv(filename, index=False)
    print(f"Created '{filename}' with shape ({rows}, {cols + 1}) including label column")

if __name__ == "__main__":
    create_mlp_dataset()