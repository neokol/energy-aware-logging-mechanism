import enum

class ModelType(str, enum.Enum):
    MLP = "MLP"
    CNN = "CNN"

class PrecisionType(str, enum.Enum):
    FP32 = "FP32"
    INT8 = "INT8"
    
class AlgorithmType(str, enum.Enum):
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"
    
class ModelFormats(str, enum.Enum):
    PKL = "pkl"
    ONNX = "onnx"