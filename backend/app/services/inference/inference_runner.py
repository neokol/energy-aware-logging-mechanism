from abc import ABC, abstractmethod

class InferenceStrategy(ABC):
    @abstractmethod
    def run(self, model_record, df, y_true) -> list:
        """
        Executes the inference and returns a list of result dictionaries.
        Each dictionary contains: precision, latency, emissions, energy, accuracy, etc.
        """
        pass