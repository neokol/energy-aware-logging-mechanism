import torch
import time
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from backend.app.services.base_model import BaseAIModel
from backend.ai_models.cnn import SimpleCNN
from backend.app.models.enums import PrecisionType


load_dotenv()

CNN_MODEL_PATH = os.getenv("CNN_MODEL_PATH", "trained_models/cnn_mnist_v1.pth.pth")

class CNNModelService(BaseAIModel):
    
    def load_model(self):
        if not os.path.exists(CNN_MODEL_PATH):
            raise FileNotFoundError(f"CNN Model not found at {CNN_MODEL_PATH}. Run setup_cnn.py first.")
        
        model = SimpleCNN()
        model.load_state_dict(torch.load(CNN_MODEL_PATH))
        model.eval()
        return model

    def run_inference(self, df: pd.DataFrame, precision: str) -> tuple[float, float]:
        """
        Expects a DataFrame where columns are pixels (0-783) or (1-784).
        It might have a 'label' column which we should drop if it exists.
        """
        
        # 1. DATA PREPROCESSING (The "Reshape" Trick)
        # Drop non-numeric columns (like 'label' if it exists in your CSV)
        df_numeric = df.select_dtypes(include=[np.number])
        
        # If dataset has 785 columns, the first one is likely the label. Drop it.
        if df_numeric.shape[1] == 785:
            data_values = df_numeric.iloc[:, 1:].values # Keep columns 1 to end
        else:
            data_values = df_numeric.values

        # Convert to Tensor
        # Input shape is (N_samples, 784)
        input_tensor = torch.tensor(data_values, dtype=torch.float32)
        
        # RESHAPE: (N, 784) -> (N, 1, 28, 28)
        # The CNN needs 4 Dimensions: [BatchSize, Channels, Height, Width]
        try:
            input_tensor = input_tensor.view(-1, 1, 28, 28)
        except RuntimeError:
            raise ValueError(f"Shape mismatch! Expected 784 pixels per row, got {df_numeric.shape[1]}")

        # Normalize (0-255 -> 0-1) roughly, or use standard normalization
        input_tensor = input_tensor / 255.0

        # 2. LOAD MODEL
        model = self.load_model()

        # 3. QUANTIZATION (The Thesis Experiment)
        if precision == PrecisionType.INT8.value:
            print("--- Applying INT8 Quantization (CNN) ---")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
        else:
            print("--- Running Standard FP32 (CNN) ---")

        # 4. RUN INFERENCE
        start_time = time.time()
        
        with torch.no_grad():
            # Loop for measurability (CNNs are heavy, so 5 loops is enough)
            for _ in range(5):
                _ = model(input_tensor)

        end_time = time.time()
        latency = end_time - start_time
        
        # Dummy accuracy for Phase 2 (since we don't check labels against predictions yet)
        accuracy = 0.98 if precision == "fp32" else 0.96

        return latency, accuracy