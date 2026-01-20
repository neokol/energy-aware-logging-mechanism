import uuid
from datetime import datetime
from sqlalchemy import Column, Float, ForeignKey, String, DateTime, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from backend.app.database.db import Base
from backend.app.models.enums import PrecisionType

class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    precision = Column(Enum(PrecisionType), nullable=False)
    
    latency_seconds = Column(Float, nullable=True)
    emissions_kg = Column(Float, nullable=True)
    energy_consumed_kwh = Column(Float, nullable=True)
    cpu_energy_kwh = Column(Float, nullable=True)
    ram_energy_kwh = Column(Float, nullable=True)
    duration = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    dataset = relationship("Dataset", back_populates="experiments")
    