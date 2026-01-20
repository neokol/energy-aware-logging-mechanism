import logging
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from codecarbon import EmissionsTracker

from backend.app.models.datasets import Dataset
from backend.app.models.enums import PrecisionType
from backend.app.models.experiments import Experiment
from backend.app.services.base_model import BaseAIModel


logger = logging.getLogger(__name__)

async def execute_experiment(
    session: AsyncSession, 
    dataset: Dataset, 
    df: pd.DataFrame, 
    model_service: BaseAIModel, 
    precision: PrecisionType
) -> Experiment:
    """
    Orchestrates the full experiment: 
    1. Starts Tracker
    2. Runs Inference (FP32/INT8)
    3. Stops Tracker
    4. Saves to Database
    """
    try:
        logger.info(f"Starting Experiment Run: {precision} for Dataset ID {dataset.id}")
        
        # 1. Start Emissions Tracker
        tracker = EmissionsTracker(
            project_name=f"thesis_{dataset.ai_model}_{precision}",
            measure_power_secs=0.1,
            save_to_file=False
        )
        
        tracker.start()
        
        # 2. Run Inference
        try:
            latency, accuracy = model_service.run_inference(df, precision)
        except Exception as e:
            tracker.stop()
            logger.error(f"Inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"Inference failed for {precision}: {e}")
        
        # 3. Collect Metrics
        tracker.stop()
        data = tracker.final_emissions_data

        # 4. Save to Database
        new_experiment = Experiment(
            dataset_id=dataset.id,
            precision=precision,
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
        
        logger.info(f"Experiment saved. Energy: {data.energy_consumed} kWh")
        return new_experiment
    except Exception as e:
        logger.error(f"Error during experiment execution: {e}")
        raise HTTPException(status_code=500, detail="Experiment execution failed")