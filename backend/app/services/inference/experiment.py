import uuid
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.datasets import Dataset
from app.models.experiments import Experiment
from app.models.model import Model
from app.services.inference.inference_factory import InferenceFactory


async def run_experiment_logic(session: AsyncSession , model_id: str, dataset_id: str):
    # 1. Fetch Records
    model_record = await session.execute(select(Model).where(Model.id == model_id))
    model_record = model_record.scalar_one_or_none()
    dataset_record = await  session.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset_record = dataset_record.scalar_one_or_none()
    
    # 2. Load Data
    df = pd.read_csv(dataset_record.filepath)
    y_true = df['target'].values if 'target' in df.columns else None

    # 3. Get Strategy from Factory
    strategy = InferenceFactory.get_strategy(model_record.filename)

    # 4. Execute
    results = strategy.run(model_record, df, y_true)

    # 5. Save Results
    saved_experiments = []
    for res in results:
        exp = Experiment(
            dataset_id=dataset_id,
            precision=res["precision"],
            latency_seconds=res["latency"],
            emissions_kg=res["emissions"],
            energy_consumed_kwh=res["energy"],
            cpu_energy_kwh=res["cpu_energy"],
            ram_energy_kwh=res["ram_energy"],
            accuracy=res["accuracy"],
            duration=res["duration"],
            cpu_power_watt=res["cpu_power_watt"],
            cpu_load_pct=res["cpu_load_pct"],
            carbon_intensity=res["carbon_intensity"]
        )
        session.add(exp)
        saved_experiments.append(exp)
    
    await session.commit()
    # Refresh to ensure IDs are available
    for exp in saved_experiments:
        await session.refresh(exp)
    return saved_experiments


async def run_batch_experiment(session: AsyncSession, dataset_id: str, model_ids: list[str]):
    # 1. Generate the shared Group ID for this entire run
    batch_id = str(uuid.uuid4())
    
    # 2. Fetch the Dataset (Only need to do this once!)
    result_dataset = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset_record = result_dataset.scalar_one_or_none()
    
    if not dataset_record:
        raise ValueError("Dataset not found")
        
    df = pd.read_csv(dataset_record.filepath)
    y_true = df['target'].values if 'target' in df.columns else None

    saved_experiments = []

    # 3. Loop through every Model the user selected
    for m_id in model_ids:
        # Fetch the specific model
        result_model = await session.execute(select(Model).where(Model.id == m_id))
        model_record = result_model.scalar_one_or_none()
        
        if not model_record:
            print(f"⚠️ Skipping model {m_id} - Not found in DB.")
            continue

        # Get Strategy and Run (SYNC)
        strategy = InferenceFactory.get_strategy(model_record.filename)
        results = strategy.run(model_record, df, y_true)

        # 4. Save Results with the shared batch_id
        for res in results:
            exp = Experiment(
                batch_id=batch_id,           
                dataset_id=dataset_id,
                model_id=m_id,               
                precision=res["precision"],
                latency_seconds=res["latency"],
                emissions_kg=res["emissions"],
                energy_consumed_kwh=res["energy"],
                cpu_energy_kwh=res["cpu_energy"],
                ram_energy_kwh=res["ram_energy"],
                accuracy=res["accuracy"],
                duration=res.get("duration"),
                cpu_power_watt=res.get("cpu_power_watt"),
                cpu_load_pct=res.get("cpu_load_pct"),
                carbon_intensity=res.get("carbon_intensity")
            )
            session.add(exp)
            saved_experiments.append(exp)
    
    # 5. Commit everything to the database at once
    await session.commit()
    
    for exp in saved_experiments:
        await session.refresh(exp)

    # Return the batch_id so the frontend knows how to look up the results
    return batch_id, saved_experiments