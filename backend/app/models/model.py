import uuid
from datetime import datetime
from sqlalchemy import Column, ForeignKey, String, Text, DateTime, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.db import Base
from app.models.enums import AlgorithmType, PrecisionType, ModelType


class Model(Base):
    __tablename__ = "models"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    filename = Column(String(255), nullable=False)
    filepath = Column(String(1024), nullable=False)
    description = Column(Text, nullable=True)
    algorithm = Column(Enum(AlgorithmType), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    precision = Column(Enum(PrecisionType), nullable=True)
    model_type = Column(Enum(ModelType), nullable=True)
    
    experiments = relationship("Experiment", back_populates="model")