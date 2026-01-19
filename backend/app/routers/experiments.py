import logging
import pandas as pd
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from codecarbon import EmissionsTracker

from backend.app.database.db import get_async_session
from backend.app.models.datasets import Dataset
from backend.app.models.experiments import Experiment
from backend.app.schemas.experiments import ExperimentResponse, ExperimentCreate
from backend.app.services.model_factory import ModelFactory

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/run-experiment", response_model=ExperimentResponse)
async def run_experiment(
    request: ExperimentCreate,
    session: AsyncSession = Depends(get_async_session)
):
    try:
        logger.info(f"Received experiment request for dataset ID: {request.dataset_id}")
        result = await session.execute(select(Dataset).where(Dataset.id == request.dataset_id))
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
            

        df = pd.read_csv(dataset.filepath)
        
        tracker = EmissionsTracker(
            project_name="thesis_mlp_run",
            measure_power_secs=0.1,
            save_to_file=False
        )
        model_service = ModelFactory.get_model_service("mlp")
        tracker.start()
        
        try:
            latency, accuracy = model_service.run_inference(df, request.model_type)
        except Exception as e:
            tracker.stop()
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        tracker.stop()
        data = tracker.final_emissions_data

        
        new_experiment = Experiment(
            dataset_id=dataset.id,
            model_type=request.model_type,
            accuracy=accuracy, 
            latency_seconds=latency,
            emissions_kg=data.emissions,
            energy_consumed_kwh=data.energy_consumed,
            cpu_energy_kwh=data.cpu_energy,
            ram_energy_kwh=data.ram_energy,
            duration=data.duration
        )
        
        session.add(new_experiment)
        await session.commit()
        await session.refresh(new_experiment)

        return new_experiment
    except HTTPException as he:
        logger.error(f"HTTP error during experiment: {he.detail}")
        raise he
    

@router.get("/experiments/", response_model=List[ExperimentResponse])
async def get_experiments(
    session: AsyncSession = Depends(get_async_session)
):
    try:
        logger.info("Fetching all experiments from the database.")
        result = await session.execute(select(Experiment))
        experiments = [row[0] for row in result.all()]

        return experiments
    except Exception as e:
        logger.error(f"Error fetching experiments: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch experiments")