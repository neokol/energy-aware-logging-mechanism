# 🌿 Energy-Aware Logging Mechanism (Thesis Prototype)
This project is the implementation of an Energy-Aware Logging Mechanism for AI Algorithms. It is designed to measure, record, and analyze the energy consumption (CO2 footprint) of different AI model configurations

Built with FastAPI, CodeCarbon, and SQLite.

## 🚀 Current Features
- Dataset Management: Upload and register CSV datasets for experiments.

- Energy Tracking: Uses CodeCarbon to track CPU/RAM energy usage during algorithm execution.

- AI Simulation: Simulates Neural Network Forward Passes to compare:

    1. FP32 (Standard): High precision, higher energy.

    2. Int8 (Quantized): Lower precision, potential energy savings.

- Metric Logging: Automatically calculates and saves:

    1. Latnecy (Seconds)

    2. Accuracy (vs Ground Truth)

    3. Energy Consumed (kWh)

    2. CO2 Emissions (kg)


## 🛠️ Tech Stack
- Language: Python 3.10+

- Framework: FastAPI

- Database: SQLite (Async via SQLAlchemy)

- Energy Monitoring: CodeCarbon

- Data Processing: Pandas, NumPy, Scikit-learn

- Package Manager: uv

## ⚙️ Installation & Setup

1. Clone & Install Dependencies
    If you are using uv:

    ```
    uv sync
    ```


2. Environment Configuration
Create a .env file in the root directory:


```
HOST=0.0.0.0
PORT=8000
DEBUG=True
DATABASE_URL=sqlite+aiosqlite:///./thesis.db
UPLOAD_DIR=uploaded_datasets
```

## 🏃‍♂️ Running the Application
Start the server using Uvicorn:

```
uv run uvicorn app.main:app --reload
```

The API will be available at: http://127.0.0.1:8000 Interactive Documentation (Swagger UI): http://127.0.0.1:8000/docs

## 🧪 How to Run an Experiment
Step 1: Upload a Dataset

Endpoint: POST /datasets

- Upload your CSV file (e.g., labeled_thesis_data.csv).

- Response: You will get a dataset_id (e.g., "550e8400-..."). Copy this ID.

Step 2: Run the Algorithm

Endpoint: POST /experiments/run-experiment

Body:

```
{
  "dataset_id": "YOUR_DATASET_ID_HERE",
  "model_type": "fp32" 
}
```
(Options for model_type: "fp32" or "int8")

Step 3: Check Results
The API returns a JSON object with the measurement results:
```
{
  "dataset_id": "...",
  "model_type": "fp32",
  "accuracy": 1.0,
  "latency_seconds": 4.23,
  "emissions_kg": 0.000015,
  "energy_consumed_kwh": 0.000042
}
```



## 📂 Project Structure

```
/
├── app/
│   ├── core/           # Logging & Config
│   ├── database/       # Database connection
│   ├── models/         # SQLAlchemy Tables (Dataset, Experiment)
│   ├── routers/        # API Endpoints
│   ├── schemas/        # Pydantic Models (Request/Response)
│   └── services/       # AI Logic & CodeCarbon wrapper
├── logs/               # Application logs (app.log)
├── uploaded_datasets/  # CSV storage
├── thesis.db           # SQLite Database file
└── main.py             # Entry point
```

📝 Logging
Console: Real-time updates.

File (logs/app.log): Persistent history of all uploads and experiments. Useful for auditing thesis data.