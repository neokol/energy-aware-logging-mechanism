import pandas as pd
import numpy as np

def create_mlp_dataset():
    print("Generating MLP-compatible dataset...")
    rows = 5000  
    cols = 512   
    
    
    data = np.random.randn(rows, cols).astype(np.float32)
    
    
    col_names = [f"feature_{i}" for i in range(cols)]
    
    df = pd.DataFrame(data, columns=col_names)
    
    filename = "maintenance_data.csv"
    df.to_csv(filename, index=False)
    print(f"Created '{filename}' with shape ({rows}, {cols})")

if __name__ == "__main__":
    create_mlp_dataset()