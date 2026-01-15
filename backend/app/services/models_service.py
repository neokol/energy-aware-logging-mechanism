import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

class ModelsService:
    def __init__(self):
        pass
    
    def run_neutral_network(self, df:pd.DataFrame, model_type:str):
        y_true = df['target'].values
        X = df.drop(columns=['target']).select_dtypes(include=[np.number]).to_numpy()
        
        # Calculate the threshold from the data (Simulating the 'Model Weights')
        # In a real scenario, this threshold comes from training. 
        # Here we derive it to ensure the FP32 model is accurate.
        original_threshold = X.sum(axis=1).mean()

        # --- START TIMING (Inference Only) ---
        start_time = time.time()
        model_type = model_type.lower()
        if model_type == "int8":
            # === INT8 SIMULATION (Low Precision) ===
            # Quantize: Convert 0.0-1.0 floats to 0-100 integers
            scale_factor = 100
            X_int = (X * scale_factor).astype(np.int8) # This forces loss of data!
            
            # Adjust threshold to match the new scale
            threshold_int = original_threshold * scale_factor
            
            # Inference: Sum integers
            # We loop to make the CPU work hard (for CodeCarbon)
            for _ in range(20):
                predictions_raw = X_int.sum(axis=1)
                
            y_pred = (predictions_raw > threshold_int).astype(int)
            
        elif model_type == "fp32":
            
            X_float = X.astype(np.float32)
            
            # Inference: Sum floats
            for _ in range(20):
                predictions_raw = X_float.sum(axis=1)
                
            y_pred = (predictions_raw > original_threshold).astype(int)
        else:
            raise ValueError("Unsupported model type. Use 'int8' or 'fp32'.")
        
        # --- END TIMING ---
        end_time = time.time()
        latency = end_time - start_time
        
        # Calculate Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        return latency, accuracy