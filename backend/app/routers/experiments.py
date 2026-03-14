import os
import uuid
import logging
import pandas as pd
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import get_async_session
from app.models.datasets import Dataset
from app.models.experiments import Experiment
from app.schemas.experiments import BatchExperimentRequest, ExperimentComparisonResponse, ExperimentRequest, ExperimentResponse
from app.services.experiment_service import execute_experiment
from app.services.inference.experiment import run_batch_experiment, run_experiment_logic
from app.services.model_factory import ModelFactory
from app.models.enums import PrecisionType

logger = logging.getLogger(__name__)

router = APIRouter()


async def _get_dataset_and_model(session: AsyncSession, dataset_id: str):
    """
    Fetches dataset from DB, loads CSV, and instantiates the Model Service.
    """
    # 1. Fetch from DB
    result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        logger.error(f"Dataset with ID {dataset_id} not found")
        raise HTTPException(status_code=404, detail="Dataset not found")

    # 2. Load CSV
    if not os.path.exists(dataset.filepath):
        logger.error(f"File not found at path: {dataset.filepath}")
        raise HTTPException(status_code=404, detail="File not found on disk")
    try:
        df = pd.read_csv(dataset.filepath)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise HTTPException(status_code=500, detail=f"Could not read CSV: {e}")

    # 3. Get Model Service
    service_key = dataset.ai_model.upper()
    try:
        model_service = ModelFactory.get_model_service(service_key)
    except ValueError:
        logger.error(f"Model '{dataset.ai_model}' not supported")
        raise HTTPException(status_code=400, detail=f"Model '{dataset.ai_model}' not supported")

    return dataset, df, model_service


@router.post("/run-experiment", response_model=ExperimentResponse)
async def run_experiment(
    dataset_id: str,
    precision: PrecisionType,
    session: AsyncSession = Depends(get_async_session)
):
    try:
        logger.info(f"Received experiment request for dataset ID: {dataset_id}")
        dataset, df, model_service = await _get_dataset_and_model(session, dataset_id)

        experiment = await execute_experiment(
        session=session,
        dataset=dataset,
        df=df,
        model_service=model_service,
        precision=precision.value
    )
        logger.info(f"Experiment completed for dataset ID: {dataset_id} with model type: {precision.value}")
        return experiment
    except HTTPException as he:
        logger.error(f"HTTP error during experiment: {he.detail}")
        raise he
    
@router.get("/experiments/{dataset_id}", response_model=ExperimentComparisonResponse)
async def get_experiment_by_dataset(
    dataset_id: str, 
    session: AsyncSession = Depends(get_async_session)
):
    try:
        logger.info(f"Fetching experiment for dataset ID: {dataset_id}")
        
        result_fp32 = await session.execute(
        select(Experiment)
        .where(Experiment.dataset_id == dataset_id)
        .where(Experiment.precision == PrecisionType.FP32)
        .order_by(desc(Experiment.created_at)) # Newest first
        .limit(1)
    )
        exp_fp32 = result_fp32.scalar_one_or_none()

        result_int8 = await session.execute(
        select(Experiment)
        .where(Experiment.dataset_id == dataset_id)
        .where(Experiment.precision == PrecisionType.INT8)
        .order_by(desc(Experiment.created_at))
        .limit(1)
    )
        exp_int8 = result_int8.scalar_one_or_none()

        if not exp_fp32 or not exp_int8:
            
            raise HTTPException(status_code=404, detail="Incomplete experiment history. Please run a new comparison.")
        logger.info(f"Fetched experiments for dataset ID: {dataset_id}")
        return ExperimentComparisonResponse(
            dataset_id=dataset_id,
            fp32=exp_fp32,
            int8=exp_int8,
        )

    except Exception as e:
        logger.error(f"Error fetching experiment for dataset ID {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch experiment")

@router.get("/experiments/", response_model=List[ExperimentResponse])
async def get_experiments(
    session: AsyncSession = Depends(get_async_session)
):
    try:
        logger.info("Fetching all experiments from the database.")
        result = await session.execute(select(Experiment))
        experiments = [row[0] for row in result.all()]
        logger.info(f"Fetched {len(experiments)} experiments.")
        return experiments
    except Exception as e:
        logger.error(f"Error fetching experiments: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch experiments")
    
@router.delete("/experiments/{experiment_id}")
async def delete_experiment(
    experiment_id: str, 
    session: AsyncSession = Depends(get_async_session)
):
    try:
        logger.info(f"Received request to delete experiment with ID: {experiment_id}")
        result = await session.execute(select(Experiment).where(Experiment.id == experiment_id))
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            logger.warning(f"Experiment with ID {experiment_id} not found for deletion")
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        await session.delete(experiment)
        await session.commit()
        
        logger.info(f"Experiment with ID {experiment_id} deleted successfully from database")
        
        return {"detail": "Experiment deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting experiment with ID {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not delete experiment")
    
@router.delete("/experiments")
async def delete_all_experiment(
    session: AsyncSession = Depends(get_async_session)
):
    try:
        logger.info("Received request to delete all experiment")
        await session.execute(delete(Experiment)) 

        await session.commit()
        
        logger.info("All experiments deleted successfully from database")
        
        return {"detail": "All experiments deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting experiments: {e}")
        raise HTTPException(status_code=500, detail="Could not delete experiments")
    
def _avg(values: list, key: str) -> float:
    vals = [getattr(e, key) for e in values if getattr(e, key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


@router.get("/compare/{dataset_id}")
async def compare_models(
    dataset_id: str,
    n_runs: int = 1,
    session: AsyncSession = Depends(get_async_session)
):
    try:
        logger.info(f"Starting model comparison for dataset ID: {dataset_id}, n_runs={n_runs}")
        n_runs = max(1, min(n_runs, 10))  # clamp between 1 and 10
        dataset, df, model_service = await _get_dataset_and_model(session, dataset_id)

        batch_id = str(uuid.uuid4())
        runs_fp32, runs_int8 = [], []

        for i in range(n_runs):
            logger.info(f"Run {i + 1}/{n_runs} ...")
            runs_fp32.append(await execute_experiment(session, dataset, df, model_service, PrecisionType.FP32, batch_id=batch_id))
            runs_int8.append(await execute_experiment(session, dataset, df, model_service, PrecisionType.INT8, batch_id=batch_id))

        fp32_energy  = _avg(runs_fp32, "energy_consumed_kwh")
        int8_energy  = _avg(runs_int8, "energy_consumed_kwh")
        fp32_latency = _avg(runs_fp32, "latency_seconds")
        int8_latency = _avg(runs_int8, "latency_seconds")
        fp32_acc     = _avg(runs_fp32, "accuracy")
        int8_acc     = _avg(runs_int8, "accuracy")

        energy_saved_kwh = fp32_energy - int8_energy
        energy_saved_pct = (energy_saved_kwh / fp32_energy * 100) if fp32_energy > 0 else 0
        latency_saved_pct = ((fp32_latency - int8_latency) / fp32_latency * 100) if fp32_latency > 0 else 0

        logger.info(f"Comparison completed ({n_runs} run(s)) for dataset ID: {dataset_id}")
        return {
            "dataset_id": dataset.id,
            "model_type": dataset.ai_model,
            "n_runs": n_runs,
            "fp32_results": runs_fp32[-1],
            "int8_results": runs_int8[-1],
            "averaged": {
                "fp32_energy_kwh": fp32_energy,
                "int8_energy_kwh": int8_energy,
                "fp32_latency_sec": fp32_latency,
                "int8_latency_sec": int8_latency,
                "fp32_accuracy": fp32_acc,
                "int8_accuracy": int8_acc,
            },
            "improvement": {
                "energy_saved_kwh": energy_saved_kwh,
                "energy_saved_percentage": round(energy_saved_pct, 2),
                "latency_reduced_percentage": round(latency_saved_pct, 2),
                "accuracy_loss": round(fp32_acc - int8_acc, 4)
            }
        }
    except HTTPException as he:
        logger.error(f"HTTP error during model comparison: {he.detail}")
        raise he
    

@router.post("/run-inference")
async def run_inference(
    req: ExperimentRequest,
    db: AsyncSession = Depends(get_async_session)
):
    try:
        experiments =await  run_experiment_logic(db, req.model_id, req.dataset_id)
        
        return {
            "message": "Success", 
            "experiments": [
                {"id": e.id, "precision": e.precision, "energy": e.energy_consumed_kwh} 
                for e in experiments
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/run-batch")
async def run_batch(
    req: BatchExperimentRequest,
    db: AsyncSession = Depends(get_async_session)
):
    try:
        # Call the new upgraded service
        batch_id, experiments = await run_batch_experiment(db, req.dataset_id, req.model_ids)
        
        return {
            "message": "Batch completed successfully", 
            "batch_id": batch_id, # Frontend will need this!
            "total_runs": len(experiments)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))