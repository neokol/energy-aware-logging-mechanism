import enum

class ModelType(str, enum.Enum):
    MLP = "MLP"
    CNN = "CNN"

class PrecisionType(str, enum.Enum):
    FP32 = "FP32"
    INT8 = "INT8"