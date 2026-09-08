import os
import time
import joblib
import numpy as np
import logging
import psutil
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from codecarbon import EmissionsTracker
from sklearn.metrics import r2_score, accuracy_score

from app.models.enums import PrecisionType
from app.services.inference.inference_runner import InferenceStrategy
from app.core.logging import setup_logging


setup_logging()
logger = logging.getLogger(__name__)

# A single ONNX pass over a test set is sub-second; repeat it until at least this
# many seconds have elapsed so codecarbon has a usable measurement window.
MIN_MEASURE_SECONDS = float(os.getenv("ONNX_MIN_MEASURE_SECONDS", "5"))
MAX_LOOPS = 100_000

ARTIFACTS_DIR = "artifacts_deep_learning"


def _load_matching_preprocessor(df):
    """Find a saved sklearn preprocessor whose input columns are all present in df."""
    if not os.path.isdir(ARTIFACTS_DIR):
        return None
    for fname in sorted(f for f in os.listdir(ARTIFACTS_DIR)
                        if "preprocessor" in f and f.endswith(".joblib")):
        try:
            pre = joblib.load(os.path.join(ARTIFACTS_DIR, fname))
            names = getattr(pre, "feature_names_in_", None)
            if names is not None and set(names).issubset(df.columns):
                logger.info(f"Using preprocessor {fname}")
                return pre
        except Exception as e:
            logger.warning(f"Could not load preprocessor {fname}: {e}")
    return None

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
        
        n_samples = len(y_true) if y_true is not None else next(iter(inputs.values())).shape[0]

        tracker = EmissionsTracker(output_dir=".", log_level="error")
        tracker.start()

        # CPU Load Measure
        psutil.cpu_percent(interval=None)

        loops = 0
        start_time = time.time()
        while True:
            preds = session.run([output_name], inputs)[0]
            loops += 1
            if time.time() - start_time >= MIN_MEASURE_SECONDS or loops >= MAX_LOOPS:
                break
        end_time = time.time()

        cpu_load = psutil.cpu_percent(interval=None)

        emissions = tracker.stop()
        duration = end_time - start_time
        energy = tracker.final_emissions_data.energy_consumed

        # --- CALCULATIONS ---
        power_watt = (energy * 3_600_000) / duration if duration > 0 else 0
        carbon_intensity = (emissions * 1000) / energy if energy > 0 else 0
        throughput = (n_samples * loops) / duration if duration > 0 else 0.0
        logger.info(f"{precision}: {loops} loops over {n_samples} samples in {duration:.2f}s")

        return {
            "precision": precision,
            "latency": duration,
            "duration": duration,
            "emissions": emissions,
            "energy": energy,
            "cpu_energy": tracker.final_emissions_data.cpu_energy,
            "ram_energy": tracker.final_emissions_data.ram_energy,
            "accuracy": self._calculate_accuracy(y_true, preds),
            "throughput": throughput,
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
                
                expected_shape = inp.shape

                # Reshape if necessary (N,) -> (N, 1)
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)

                if len(expected_shape) == 3 and len(data.shape) == 2:
                    # Μετατροπή από (N, Features) σε (N, 1, Features)
                    data = data.reshape(data.shape[0], 1, data.shape[1])
                    logger.info(f"📐 Reshaped input '{model_col_name}' for CNN to {data.shape}")
                
                # Enforce Types based on ONNX expectation
                if 'string' in inp.type:
                    data = data.astype(str)
                elif 'float' in inp.type:
                    data = data.astype(np.float32)
                elif 'int' in inp.type:
                    data = data.astype(np.int64)
                    
                onnx_inputs[model_col_name] = data
            else:
                if inp.name == "input":
                    preprocessor = _load_matching_preprocessor(df)
                    if preprocessor is not None:
                        cols = list(preprocessor.feature_names_in_)
                        data = preprocessor.transform(df[cols]).astype(np.float32)
                    else:
                        logger.warning("No matching preprocessor found — feeding raw values")
                        data = df.values.astype(np.float32)
                    expected_shape = inp.shape
                    
                    if len(expected_shape) == 3 and len(data.shape) == 2:
                        data = data.reshape(data.shape[0], 1, data.shape[1])
                    
                    onnx_inputs[inp.name] = data
                else:
                    print(f"⚠️ Warning: Model expects '{model_col_name}' but it is missing from CSV!")
                
        return onnx_inputs

    # def _calculate_accuracy(self, y_true, y_pred):
    #     if y_true is None: 
    #         return 0.0
            
    #     y_true = np.array(y_true).flatten()
    #     y_pred = np.array(y_pred).flatten()

    #     # Check if the target data is continuous (floats) or categorical (ints/strings)
    #     if y_true.dtype.kind == 'f':
    #         # It's Regression (California Housing)
    #         # Use R-squared. 1.0 is perfect, 0.0 is terrible.
    #         # We max it with 0.0 so we don't get negative accuracy in the UI.
    #         score = r2_score(y_true, y_pred)
    #         return max(0.0, float(score)) 
    #     else:
    #         # It's Classification (Adult Income)
    #         # Use standard accuracy percentage.
    #         return float(accuracy_score(y_true, y_pred))

    def _calculate_accuracy(self, y_true, y_pred):
        if y_true is None: 
            return 0.0
            
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Ελέγχουμε αν είναι Classification βρίσκοντας πόσες μοναδικές τιμές έχει το target
        # Στο Adult (Binary Classification) θα έχει μόνο 2 (το 0 και το 1).
        is_classification = len(np.unique(y_true)) <= 2 or y_true.dtype.kind in ['O', 'U', 'S', 'b']

        if is_classification:
            # Αν το PyTorch έβγαλε πιθανότητες (floats), τις κάνουμε στρογγυλοποίηση στο 0 ή 1
            if y_pred.dtype.kind == 'f':
                y_pred = (y_pred >= 0.5).astype(int)
            
            # Εξασφαλίζουμε ότι και το target είναι ακέραιος
            if y_true.dtype.kind == 'f':
                y_true = y_true.astype(int)
                
            return float(accuracy_score(y_true, y_pred))
        else:
            # Regression (California Housing): return the real R^2 (may be negative
            # if the model is worse than predicting the mean — that is informative).
            return float(r2_score(y_true, y_pred))