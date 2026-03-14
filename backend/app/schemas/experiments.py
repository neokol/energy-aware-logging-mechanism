from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List

from app.models.enums import PrecisionType

class ExperimentCreate(BaseModel):
    dataset_id: str
    precision: PrecisionType  

class ExperimentResponse(BaseModel):
    id: str
    batch_id: str
    dataset_id: str
    model_id: str | None = None
    precision: PrecisionType
    latency_seconds: float |  None = None
    emissions_kg: float |  None = None
    energy_consumed_kwh: float | None = None
    cpu_energy_kwh: float |  None = None
    ram_energy_kwh: float | None = None
    accuracy: float | None = None
    duration: float | None = None
    throughput_samples_per_sec: float | None = None
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)

class ExperimentComparisonResponse(BaseModel):
    dataset_id: str
    fp32: ExperimentResponse
    int8: ExperimentResponse

    model_config = ConfigDict(from_attributes=True)

class ExperimentRequest(BaseModel):
    model_id: str
    dataset_id: str

class BatchExperimentRequest(BaseModel):
    dataset_id: str
    model_ids: List[str]