from pydantic import BaseModel, ConfigDict
from datetime import datetime

from backend.app.models.enums import PrecisionType

class ExperimentCreate(BaseModel):
    dataset_id: str
    precision: PrecisionType  

class ExperimentResponse(BaseModel):
    id: str
    dataset_id: str
    precision: PrecisionType
    latency_seconds: float |  None = None
    emissions_kg: float |  None = None
    energy_consumed_kwh: float | None = None
    cpu_energy_kwh: float |  None = None
    ram_energy_kwh: float | None = None
    accuracy: float |  None = None
    duration: float |  None = None
    created_at: datetime |  None = None

    model_config = ConfigDict(from_attributes=True)