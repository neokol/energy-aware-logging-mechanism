from pydantic import BaseModel, ConfigDict
from datetime import datetime
from backend.app.models.enums import ModelType

class DatasetCreate(BaseModel):
    filename: str
    filepath: str
    description: str | None = None
    ai_model: ModelType
    
class DatasetResponse(BaseModel):
    id: str
    filename: str
    filepath: str
    description: str | None = None
    ai_model: ModelType
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)