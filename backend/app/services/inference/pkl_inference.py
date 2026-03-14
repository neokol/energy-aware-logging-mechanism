import joblib
import time
import numpy as np
import logging
from codecarbon import EmissionsTracker

from app.core.logging import setup_logging
from app.models.enums import PrecisionType
from app.services.inference.inference_runner import InferenceStrategy

setup_logging()
logger = logging.getLogger(__name__)

class RunPKLInference(InferenceStrategy):
    def run(self, model_record, df, y_true) -> list:
        logger.info(f"🚀 Running Legacy Pickle Mode for {model_record.filename}...")
        
        # Load Model
        model = joblib.load(model_record.filepath)
        
        # Prepare Data (Scikit-learn handles raw DF)
        if 'target' in df.columns:
            X = df.drop(columns=['target'])
        else:
            X = df

        # Setup Tracker
        tracker = EmissionsTracker(output_dir=".", log_level="error")
        tracker.start()
        
        start_time = time.time()
        preds = model.predict(X)
        end_time = time.time()
        
        emissions = tracker.stop()
        
        # Calculate Accuracy
        accuracy = self._calculate_accuracy(y_true, preds)
        
        return [{
            "precision": PrecisionType.FP32,
            "latency": end_time - start_time,
            "emissions": emissions,
            "energy": tracker.final_emissions_data.energy_consumed,
            "cpu_energy": tracker.final_emissions_data.cpu_energy,
            "ram_energy": tracker.final_emissions_data.ram_energy,
            "accuracy": accuracy
        }]

    def _calculate_accuracy(self, y_true, y_pred):
        if y_true is None: return 0.0
        return np.mean(y_pred == y_true)