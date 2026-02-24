import time
import numpy as np
import logging
import psutil
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from codecarbon import EmissionsTracker
from sklearn.metrics import r2_score, accuracy_score

from backend.app.models.enums import PrecisionType
from backend.app.services.inference.inference_runner import InferenceStrategy
from backend.app.core.logging import setup_logging


setup_logging()
logger = logging.getLogger(__name__)

class RunONNXInference(InferenceStrategy):
    def run(self, model_record, df, y_true) -> list:
        logger.info(f"🚀 Running ONNX Mode for {model_record.filename}...")
        
        
        # Prepare Data
        if 'target' in df.columns:
            X = df.drop(columns=['target'])
        else:
            X = df
            
        
        if "int8" in model_record.filename.lower():
            precision = PrecisionType.INT8
        else:
            precision = PrecisionType.FP32
        
        result = self._run_single_pass(
            model_path=model_record.filepath, 
            X=X, 
            y_true=y_true, 
            precision=precision
        )
        return [result]
        

    def _run_single_pass(self, model_path, X, y_true, precision):
        session = ort.InferenceSession(model_path)
        inputs = self._to_onnx_input(session, X)
        output_name = session.get_outputs()[0].name
        
        tracker = EmissionsTracker(output_dir=".", log_level="error")
        tracker.start()
        
        # CPU Load Measure
        psutil.cpu_percent(interval=None) 
        
        start_time = time.time()
        preds = session.run([output_name], inputs)[0]
        end_time = time.time()
        
        cpu_load = psutil.cpu_percent(interval=None)
        
        emissions = tracker.stop()
        duration = end_time - start_time
        energy = tracker.final_emissions_data.energy_consumed
        
        # --- CALCULATIONS ---
        power_watt = (energy * 3_600_000) / duration if duration > 0 else 0
        carbon_intensity = (emissions * 1000) / energy if energy > 0 else 0
        
        return {
            "precision": precision,
            "latency": duration,
            "duration": duration,
            "emissions": emissions,
            "energy": energy,
            "cpu_energy": tracker.final_emissions_data.cpu_energy,
            "ram_energy": tracker.final_emissions_data.ram_energy,
            "accuracy": self._calculate_accuracy(y_true, preds),
            # NEW METRICS
            "cpu_power_watt": power_watt,
            "cpu_load_pct": cpu_load,
            "carbon_intensity": carbon_intensity
        }

    def _to_onnx_input(self, sess, df):
        onnx_inputs = {}
        
        # 1. Create a mapping of {sanitized_name: real_column_name}
        # This lets us find "capital-gain" even if we look for "capital_gain"
        col_map = {}
        for col in df.columns:
            sanitized = col.replace("-", "_") # e.g. "capital-gain" -> "capital_gain"
            col_map[sanitized] = col
            col_map[col] = col # Also map the exact name

        # 2. Iterate through what the MODEL wants
        for inp in sess.get_inputs():
            model_col_name = inp.name # e.g. "capital_gain"
            
            # Check if we have this column (either exact or mapped)
            if model_col_name in col_map:
                real_col_name = col_map[model_col_name]
                data = df[real_col_name].values
                
                # Reshape if necessary (N,) -> (N, 1)
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                
                # Enforce Types based on ONNX expectation
                if 'string' in inp.type:
                    data = data.astype(str)
                elif 'float' in inp.type:
                    data = data.astype(np.float32)
                elif 'int' in inp.type:
                    data = data.astype(np.int64)
                    
                onnx_inputs[model_col_name] = data
            else:
                print(f"⚠️ Warning: Model expects '{model_col_name}' but it is missing from CSV!")
                
        return onnx_inputs

    def _calculate_accuracy(self, y_true, y_pred):
        if y_true is None: 
            return 0.0
            
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Check if the target data is continuous (floats) or categorical (ints/strings)
        if y_true.dtype.kind == 'f':
            # It's Regression (California Housing)
            # Use R-squared. 1.0 is perfect, 0.0 is terrible.
            # We max it with 0.0 so we don't get negative accuracy in the UI.
            score = r2_score(y_true, y_pred)
            return max(0.0, float(score)) 
        else:
            # It's Classification (Adult Income)
            # Use standard accuracy percentage.
            return float(accuracy_score(y_true, y_pred))