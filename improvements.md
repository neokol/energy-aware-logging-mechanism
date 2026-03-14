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

- Real accuracy measurement — currently hardcoded (0.95/0.98 FP32, 0.92/0.96 INT8); real label comparison would make the thesis results meaningful
- Upload the MLP dataset — maintenance_data.csv needs to be uploaded via the UI to run MLP experiments
