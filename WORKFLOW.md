# Dataset Upload & Experiment Workflow

## Prerequisites

Before running the server, generate the trained model files once:

```bash
cd backend
uv run python setup_models.py   # MLP — instant
uv run python setup_cnn.py      # CNN — downloads MNIST, trains 1 epoch (~2 min)
```

To generate the MLP-compatible dataset:

```bash
uv run python generate_mlp_data.py   # produces maintenance_data.csv (512 features)
```

Start the backend from the `backend/` directory:

```bash
cd backend
uv run uvicorn app.app:app --reload
```

---

## 1. Upload a Dataset

**`POST /datasets`** — `backend/app/routers/dataset.py`

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | UploadFile | CSV dataset file |
| `ai_model` | ModelType | `MLP` or `CNN` |
| `description` | str | Optional description |

### Dataset Requirements

| Model | File | Columns |
|-------|------|---------|
| MLP | `maintenance_data.csv` | 512 numeric features |
| CNN | `mnist_test.csv` | 784 pixels or 785 (label + pixels) |

### Steps
1. Save CSV file to disk at `UPLOAD_DIR` (configured in `.env`)
2. Create a `Dataset` record in DB with filepath, model type, and metadata

### Other Dataset Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /datasets` | List all datasets |
| `DELETE /datasets/{dataset_id}` | Delete dataset and file from disk |
| `PATCH /datasets/{dataset_id}` | Update description or model type |

---

## 2. Run an Experiment

**`GET /compare/{dataset_id}?n_runs=N`** — `backend/app/routers/experiments.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_id` | str (UUID) | — | ID of an uploaded dataset |
| `n_runs` | int | 1 | Number of repeat runs to average (max 10) |

### Steps

1. **Fetch dataset** from DB → load CSV into a Pandas DataFrame

2. **Select model service** via `ModelFactory` based on `dataset.ai_model`:
   - `MLP` → `mlp_service.py` — loads `trained_models/mlp_maintenance_v1.pth`
   - `CNN` → `cnn_service.py` — loads `trained_models/cnn_mnist_v1.pth`

3. **Platform detection** via `app/core/platform_config.py`:
   - Sets quantization engine: `qnnpack` (Apple Silicon) or `fbgemm` (x86)
   - Sets codecarbon tracking method per OS

4. **For each of `n_runs` iterations**, run FP32 then INT8:

   a. **Start `EmissionsTracker`** (codecarbon)

   b. **Run inference**:
      - If `INT8`: apply `torch.quantization.quantize_dynamic`
      - MLP: 10 inference loops → returns `(latency, accuracy, throughput)`
      - CNN: 5 inference loops, input reshaped to `(N, 1, 28, 28)` and normalized

   c. **Stop tracker** → collect:
      - `energy_consumed_kwh` (falls back to cpu + ram + gpu sum if NaN)
      - `emissions_kg` (falls back to energy × 0.4 kg/kWh if NaN)
      - `cpu_energy_kwh`, `ram_energy_kwh`, `duration`

   d. **Compute `throughput_samples_per_sec`** = (rows × loops) / latency

   e. **Save `Experiment` record** to DB with all metrics and shared `batch_id`

5. **Average metrics** across all runs and return response with:
   - `fp32_results` / `int8_results` — last run record
   - `averaged` — mean energy, latency, accuracy per precision
   - `improvement` — energy saved %, latency saved %, accuracy loss

---

## 3. Metrics Collected per Experiment

| Metric | Description |
|--------|-------------|
| `latency_seconds` | Total wall-clock time for all inference loops |
| `throughput_samples_per_sec` | (rows × loops) / latency |
| `energy_consumed_kwh` | Total energy (cpu + ram + gpu) |
| `emissions_kg` | CO2 equivalent |
| `cpu_energy_kwh` | CPU component energy |
| `ram_energy_kwh` | RAM component energy |
| `accuracy` | Hardcoded per model (FP32: 0.95/0.98, INT8: 0.92/0.96) |
| `duration` | codecarbon tracking duration |

---

## 4. Frontend Pages

| Page | Description |
|------|-------------|
| `Home.py` | Landing page |
| `pages/1_Upload.py` | Upload CSV datasets |
| `pages/2_Experiments.py` | Run comparisons, set repeat runs (sidebar slider) |
| `pages/3_Results.py` | Browse all experiments, charts, trade-off analysis, CSV export |

### Results Page Sections
- **Summary metrics** — total runs, FP32/INT8 counts, unique datasets
- **Filter panel** — by precision and dataset
- **Experiments table** + **⬇️ Export to CSV**
- **Visual comparison** — Energy, Latency, Throughput, Emissions, Accuracy charts
- **FP32 vs INT8 savings** — per-dataset energy and latency savings table
- **Accuracy vs Energy scatter plot** — trade-off frontier visualization
- **Trade-off score** = Accuracy ÷ Energy — efficiency ranking per run

---

## 5. All Experiment Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /compare/{dataset_id}?n_runs=N` | Run FP32 + INT8, N times each, return averages |
| `POST /run-experiment` | Single precision run |
| `GET /experiments/` | List all past experiments |
| `GET /experiments/{dataset_id}` | Get latest FP32+INT8 pair for a dataset |
| `DELETE /experiments/{experiment_id}` | Delete one experiment |
| `DELETE /experiments` | Delete all experiments |
| `POST /run-inference` | Run inference with uploaded ONNX or PKL model |
| `POST /run-batch` | Batch inference across multiple models |

---

## 6. Platform Support

| Platform | Quantization Engine | Energy Tracking |
|----------|-------------------|-----------------|
| macOS Apple Silicon (arm64) | `qnnpack` | PowerMetrics (requires sudo once) |
| Linux x86_64 | `fbgemm` | RAPL (auto-detected) |
| Windows x86_64 | `fbgemm` | CPU TDP estimation |

Platform is auto-detected at startup via `app/core/platform_config.py` and logged.

---

## 7. Key Files

| Component | Path | Purpose |
|-----------|------|---------|
| App entry | `app/app.py` | FastAPI app, startup logging |
| Platform config | `app/core/platform_config.py` | OS detection, quantization engine, codecarbon kwargs |
| Dataset Router | `app/routers/dataset.py` | Dataset CRUD endpoints |
| Experiment Router | `app/routers/experiments.py` | Run, compare, list experiments |
| Experiment Service | `app/services/experiment_service.py` | Orchestrates tracking + inference |
| MLP Service | `app/services/mlp_service.py` | MLP inference, throughput |
| CNN Service | `app/services/cnn_service.py` | CNN inference, throughput |
| Base Model | `app/services/base_model.py` | Abstract interface: `(latency, accuracy, throughput)` |
| ModelFactory | `app/services/model_factory.py` | Returns MLP or CNN service |
| Dataset DB Model | `app/models/datasets.py` | Dataset table schema |
| Experiment DB Model | `app/models/experiments.py` | Experiment results table schema |
| Experiment Schema | `app/schemas/experiments.py` | Pydantic response models |
| Enums | `app/models/enums.py` | ModelType, PrecisionType, ModelFormats |
| Database | `app/database/db.py` | AsyncSession and engine setup |
