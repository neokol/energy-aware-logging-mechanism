from abc import ABC, abstractmethod
import pandas as pd

class BaseAIModel(ABC):
    """
    The interface that all future models (MLP, CNN, Transformer) must follow.
    """
    
    @abstractmethod
    def load_model(self):
        """Loads the weights from disk."""
        pass

    @abstractmethod
    def run_inference(self, df: pd.DataFrame, precision: str) -> tuple[float, float]:
        """
        Runs the model and returns (latency, accuracy).
        precision: 'fp32' or 'int8'
        """
        pass