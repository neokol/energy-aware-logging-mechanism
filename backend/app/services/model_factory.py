from backend.app.services.base_model import BaseAIModel
from backend.app.services.mlp_service import MLPModelService

class ModelFactory:
    @staticmethod
    def get_model_service(model_type: str) -> BaseAIModel:
        """
        Returns the specific service class based on the input string.
        """
        if model_type == "mlp":
            return MLPModelService()
        elif model_type == "cnn":
            return "# Return CNNModelService()"
        else:
            # Default fallback or Error
            raise ValueError(f"Unknown model type: {model_type}")