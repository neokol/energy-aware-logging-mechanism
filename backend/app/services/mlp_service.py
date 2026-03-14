import os
import time
import pandas as pd
import numpy as np
import torch
import logging
from dotenv import load_dotenv


from app.core.logging import setup_logging
from ai_models.mlp import MaintenanceMLP
from app.services.base_model import BaseAIModel
from app.models.enums import PrecisionType

load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

MLP_MODEL_PATH = os.getenv("MLP_MODEL_PATH", "trained_models/mlp_maintenance_v1.pth")

class MLPModelService(BaseAIModel):
    def __init__(self):
        self.input_size = 512
        self.hidden_size = 1024
        self.num_classes = 2
        
    def load_model(self):
        if not os.path.exists(MLP_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MLP_MODEL_PATH}")
        
        # Initialize architecture
        model = MaintenanceMLP(self.input_size, self.hidden_size, self.num_classes)
        # Load weights
        model.load_state_dict(torch.load(MLP_MODEL_PATH))
        model.eval()
        return model
    
    def run_inference(self, df: pd.DataFrame, precision: str) -> tuple[float, float, float]:
        # 1. Prepare Data — extract labels if present, then features
        df_numeric = df.select_dtypes(include=[np.number])
        if "label" in df.columns:
            labels = torch.tensor(df["label"].values, dtype=torch.long)
            data_values = df_numeric.drop(columns=["label"], errors="ignore").values
        else:
            labels = None
            data_values = df_numeric.values
        input_tensor = torch.tensor(data_values, dtype=torch.float32)
        
        model = self.load_model()
        
        if precision == PrecisionType.INT8.value:
            from app.core.platform_config import get_quantization_engine
            torch.backends.quantized.engine = get_quantization_engine()
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("Model quantized to INT8")
        elif precision == PrecisionType.FP32.value:
            logger.info("Using FP32 model")
            

        
        start_time = time.time()

        with torch.no_grad():
            for _ in range(10):
                output = model(input_tensor)

        end_time = time.time()
        latency = end_time - start_time

        n_loops = 10
        n_samples = len(df)
        throughput = (n_samples * n_loops) / latency if latency > 0 else 0.0

        if labels is not None:
            predictions = output.argmax(dim=1)
            accuracy = float((predictions == labels).sum()) / len(labels)
            logger.info(f"Real MLP accuracy ({precision}): {accuracy:.4f}")
        else:
            accuracy = 0.95 if precision == PrecisionType.FP32.value else 0.92
            logger.warning("No label column found — using dummy accuracy")

        return latency, accuracy, throughput