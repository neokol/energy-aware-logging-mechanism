from app.models.enums import ModelFormats 
from app.services.inference.inference_runner import InferenceStrategy
from app.services.inference.pkl_inference import RunPKLInference
from app.services.inference.onnx_inference import RunONNXInference

class InferenceFactory:
    @staticmethod
    def get_strategy(filename: str) -> InferenceStrategy:
        ext = filename.split('.')[-1].lower()
        match ext:
            case ModelFormats.PKL.value:
                return RunPKLInference()
            case ModelFormats.ONNX.value:
                return RunONNXInference()
            case _:
                raise ValueError(f"Unknown type {ext}")