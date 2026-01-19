import os
import time
import pandas as pd
import numpy as np
import torch
import logging
from dotenv import load_dotenv


from backend.app.core.logging import setup_logging
from backend.ai_models.mlp import MaintenanceMLP
from backend.app.services.base_model import BaseAIModel

load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "trained_models/mlp_maintenance_v1.pth")

class MLPModelService(BaseAIModel):
    def __init__(self):
        self.input_size = 512
        self.hidden_size = 1024
        self.num_classes = 2
        
    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        # Initialize architecture
        model = MaintenanceMLP(self.input_size, self.hidden_size, self.num_classes)
        # Load weights
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        return model
    
    def run_inference(self, df: pd.DataFrame, precision: str) -> tuple[float, float]:
        # 1. Prepare Data
        # Ensure we only take numbers and convert to Float32 Tensor
        data_values = df.select_dtypes(include=[np.number]).values
        input_tensor = torch.tensor(data_values, dtype=torch.float32)
        
        model = self.load_model()
        
        if precision == "int8":
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("Model quantized to INT8")
        elif precision == "fp32":
            logger.info("Using FP32 model")
            

        
        start_time = time.time()

        with torch.no_grad():
            for _ in range(10): 
                output = model(input_tensor)


        end_time = time.time()
        latency = end_time - start_time

        # Dummy accuracy values for illustration
        accuracy = 0.95 if precision == "fp32" else 0.92 

        return latency, accuracy