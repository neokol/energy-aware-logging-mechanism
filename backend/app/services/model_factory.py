from backend.app.models.enums import ModelType
from backend.app.services.base_model import BaseAIModel
from backend.app.services.cnn_service import CNNModelService
from backend.app.services.mlp_service import MLPModelService

class ModelFactory:
    @staticmethod
    def get_model_service(model_type: str | ModelType) -> BaseAIModel:
        """
        Returns the specific service class based on the input string.
        """
        if isinstance(model_type, str):
            try:
                # Convert string "mlp" -> ModelType.MLP
                # We use .upper() because our Enum values are "MLP", "CNN"
                model_type_enum = ModelType(model_type.upper())
            except ValueError:
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            model_type_enum = model_type
        if model_type_enum == ModelType.MLP:
            return MLPModelService()
        elif model_type_enum == ModelType.CNN:
            return CNNModelService()
        else:
            raise ValueError(f"Unknown model type: {model_type}")