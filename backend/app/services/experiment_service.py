import math
import uuid
import logging
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from codecarbon import EmissionsTracker

from app.core.platform_config import get_codecarbon_kwargs
from app.models.datasets import Dataset
from app.models.enums import PrecisionType
from app.models.experiments import Experiment
from app.services.base_model import BaseAIModel


logger = logging.getLogger(__name__)

def _safe_float(value):
    """Return None if value is NaN or None, otherwise return float."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None

async def execute_experiment(
    session: AsyncSession,
    dataset: Dataset,
    df: pd.DataFrame,
    model_service: BaseAIModel,
    precision: PrecisionType,
    batch_id: str = None
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
            **get_codecarbon_kwargs()
        )
        
        tracker.start()
        
        # 2. Run Inference
        try:
            latency, accuracy, throughput = model_service.run_inference(df, precision)
        except Exception as e:
            tracker.stop()
            logger.error(f"Inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"Inference failed for {precision}: {e}")
        
        # 3. Collect Metrics
        tracker.stop()
        data = tracker.final_emissions_data

        cpu_energy = _safe_float(data.cpu_energy) or 0.0
        ram_energy = _safe_float(data.ram_energy) or 0.0
        gpu_energy = _safe_float(getattr(data, 'gpu_energy', None)) or 0.0

        # codecarbon may return NaN for energy_consumed on Apple Silicon —
        # fall back to summing component energies
        energy_consumed = _safe_float(data.energy_consumed) or (cpu_energy + ram_energy + gpu_energy)

        # emissions may also be NaN — estimate from energy × default carbon intensity (Greece ~0.4 kg/kWh)
        emissions = _safe_float(data.emissions) or (energy_consumed * 0.4)

        # 4. Save to Database
        new_experiment = Experiment(
            batch_id=batch_id or str(uuid.uuid4()),
            dataset_id=dataset.id,
            precision=precision,
            accuracy=accuracy,
            latency_seconds=latency,
            throughput_samples_per_sec=throughput,
            emissions_kg=emissions,
            energy_consumed_kwh=energy_consumed,
            cpu_energy_kwh=cpu_energy,
            ram_energy_kwh=ram_energy,
            duration=_safe_float(data.duration)
        )
        
        session.add(new_experiment)
        await session.commit()
        await session.refresh(new_experiment)
        
        logger.info(f"Experiment saved. Energy: {data.energy_consumed} kWh")
        return new_experiment
    except Exception as e:
        logger.error(f"Error during experiment execution: {e}")
        raise HTTPException(status_code=500, detail="Experiment execution failed")