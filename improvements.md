# Session Summary

## Bug Fixes

| Issue | Fix |
|-------|-----|
| ModuleNotFoundError: No module named 'app' | Must run uvicorn from backend/ directory |
| backend.app.* imports throughout codebase | Replaced with app.* across all files |
| backend.ai_models.* imports | Replaced with ai_models.* |
| cnn_mnist_v1.pth.pth double extension | Fixed to cnn_mnist_v1.pth |
| Windows backslash paths in .env | Fixed to forward slashes |
| BACKEND_URL missing from .env | Added http://127.0.0.1:8000 |
| batch_id NOT NULL constraint failing | Generated UUID in execute_experiment |
| model_id NOT NULL constraint failing | Made nullable in DB model + schema |
| NaN energy values from codecarbon on Apple M4 | Fallback: sum cpu+ram+gpu; estimate emissions from energy × 0.4 |
| None - None TypeError in compare endpoint | Safe or 0.0 fallback on all arithmetic |
| INT8 quantization failing on Apple Silicon | Set torch.backends.quantized.engine = 'qnnpack' |
## Features Added

### Backend

- GET /compare/{dataset_id}?n_runs=N — repeat runs (1–10), averages all metrics, returns averaged block
- throughput_samples_per_sec — computed per model service, stored in DB, exposed in schema
- app/core/platform_config.py — centralized platform detection (macOS ARM / Linux x86 / Windows x86), auto-selects quantization engine and codecarbon kwargs
- Platform info logged at every startup

### Frontend

- pages/3_Results.py — new Results page with:
  - Summary metrics (total runs, FP32/INT8 counts, unique datasets)
  - Filter by precision and dataset
  - Experiments table with all metrics
  - ⬇️ Export to CSV
  - 5 comparison charts: Energy, Latency, Throughput, Emissions, Accuracy
  - FP32 vs INT8 savings table per dataset
  - Accuracy vs Energy scatter plot (trade-off frontier)
  - Trade-off score = Accuracy ÷ Energy with per-dataset breakdown
- pages/2_Experiments.py — repeat runs sidebar slider, averaged metrics panel

### Cross-platform Hardening

- Auto-detects platform at startup
- qnnpack on Apple Silicon, fbgemm on x86
- codecarbon configured per OS
- All paths use forward slashes

### Documentation

- WORKFLOW.md — fully updated with prerequisites, all endpoints, metrics table, frontend pages, platform support table, and key files
## What's Still Worth Doing

- Upload `maintenance_test.csv` via the UI (model type MLP) to run Scenario A
- Scenario B/C (ONNX path): one strategy call per model, but each call now loops the
  inference until `ONNX_MIN_MEASURE_SECONDS` (default 5s) elapse; still no batch-level n_runs
- onnx_inference now auto-selects the matching saved preprocessor by column names
  (was hard-coded to adult only -> housing inference used raw unscaled features -> R^2 ~ 0)
- Regression accuracy is now the real R^2 (can be negative); generate endpoint reports R^2/MAE/RMSE
- codecarbon resolution: even at 2000 loops the MLP runs only ~2–4 s; document this in the thesis limitations
