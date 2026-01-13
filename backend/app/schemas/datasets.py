from pydantic import BaseModel, ConfigDict
from datetime import datetime

class DatasetCreate(BaseModel):
    filename: str
    filepath: str
    description: str | None = None
    ai_model: str
    
class DatasetResponse(BaseModel):
    id: str
    filename: str
    filepath: str
    description: str | None = None
    ai_model: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)