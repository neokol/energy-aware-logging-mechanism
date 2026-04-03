import ssl
from backend.app.pytorch_models.helpers import train_pytorch_model
from backend.app.pytorch_models.models import AdultCNN1D, AdultMLP1
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import logging
from fastapi import APIRouter,  HTTPException

from dotenv import load_dotenv
import os

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import fetch_openml
import joblib
import pickle

# ONNX Imports
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


# --- SSL WORKAROUND FOR MACOS (M4) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# -------------------------------------

load_dotenv()

ADULT_DATA_URL = os.getenv("ADULT_DATA_URL")
ARTIFACTS_DIR = "artifacts_deep_learning"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/generate_adult_artifacts")
async def create_adult_artifacts():
    try:
        logger.info("Received request to generate adult artifacts")
        
        data = fetch_openml(name='adult', version=2, as_frame=True)
        X = data.data
        y = (data.target == '>50K').astype(int) # Binary Target

        # 1. Define Preprocessing
        # Numerical and categorical features based on the dataset documentation
        numeric_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
        categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

        X = X[numeric_features + categorical_features].copy()

        logger.info("Convert all categorical columns to string type for ONNX compatibility")
        for col in categorical_features:
            X[col] = X[col].astype(str)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save Test Data to CSV
        logger.info("💾 Saving test_data.csv...")
        test_df = X_test.copy()
        test_df['target'] = y_test 
        test_df.to_csv("adult_test.csv", index=False)

        # 2. Build Pipeline (Preprocessor + Model)
        logger.info("⚙️  Training Legacy Model (Random Forest)...")
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing', missing_values='nan')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False for ONNX compat
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42))
        ])

        model.fit(X_train, y_train)

        # 3. Save Pickle (Legacy)
        logger.info("📦 Saving adult_legacy.pkl...")
        joblib.dump(model, "adult_legacy.pkl")

        # 4. Convert to ONNX (FP32)
        logger.info("🔄 Converting to ONNX (FP32)...")
        
        # Types for ONNX conversion - we need to specify the input types based on the original DataFrame
        initial_types = []
        for name in X_train.columns:
            if name in categorical_features:
                initial_types.append((name, StringTensorType([None, 1])))
            else:
                initial_types.append((name, FloatTensorType([None, 1])))

        onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=12)
        
        with open("adult_fp32.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        # 5. Quantize to INT8 (The Thesis Magic)
        logger.info("⚡ Quantizing to INT8...")
        quantize_dynamic(
            model_input="adult_fp32.onnx",
            model_output="adult_int8.onnx",
            weight_type=QuantType.QUInt8
        )

        logger.info("SUCCESS! Artifacts created:")
        logger.info("   1. adult_test.csv     (Use this as Dataset Upload)")
        logger.info("   2. adult_legacy.pkl   (Upload as Legacy Model)")
        logger.info("   3. adult_fp32.onnx    (Upload as Modern Model)")
        logger.info("   4. adult_int8.onnx    (Auto-generated internally, but good to have)")
        
        
        # Simulate artifact generation logic
        # In a real implementation, this would involve data processing and model training
        
        logger.info("Adult artifacts generated successfully")
        
        return {"message": "Adult artifacts generated successfully"}
    except Exception as e:
        logger.error(f"Error during artifact generation: {e}")
        raise HTTPException(status_code=500, detail="Artifact generation failed")
    
@router.post("/generate_housing_artifacts")
async def generate_california_artifacts():
    try:
        logger.info("Loading California Housing dataset...")
        data = fetch_california_housing(as_frame=True)
        X = data.data
        y = data.target

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save Test Data (CSV)
        logger.info("Saving california_test.csv...")
        test_df = X_test.copy()
        test_df['target'] = y_test
        test_df.to_csv("california_test.csv", index=False)

        # 1. Build Pipeline (Use MLP Regressor instead of Random Forest)
        logger.info("Training Legacy Model (MLP Regressor)...")
        model = Pipeline(steps=[
            ('scaler', StandardScaler()),
            # MLP is a standard Neural Network, perfect for INT8 Quantization
            ('regressor', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
        ])
        model.fit(X_train, y_train)

        # 2. Save Legacy Pickle
        logger.info("Saving california_legacy.pkl...")
        joblib.dump(model, "california_legacy.pkl")

        # 3. Convert to ONNX (FP32)
        logger.info("Converting to ONNX (FP32)...")
        initial_types = [(name, FloatTensorType([None, 1])) for name in X_train.columns]

        # No hacks needed here. MLP naturally creates ai.onnx domain nodes.
        onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=12)
        
        with open("california_fp32.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        # 4. Quantize to INT8
        logger.info("⚡ Quantizing to INT8...")
        quantize_dynamic(
            model_input="california_fp32.onnx",
            model_output="california_int8.onnx",
            weight_type=QuantType.QUInt8
        )

        logger.info("California artifacts created successfully.")
        return {"message": "California artifacts generated successfully"}

    except Exception as e:
        logger.error(f"Error during artifact generation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Artifact generation failed: {str(e)}")
    
@router.post("/generate_adult_deep_learning_artifacts")
async def generate_adult_deep_learning_artifacts():
    try:
        logger.info("🎬 Starting Adult Deep Learning Artifact Generation (PyTorch -> ONNX)...")
        
        # 1. Load Data
        logger.info("⏳ Fetching Adult dataset from OpenML...")
        data = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
        X = data.data
        # Μετατροπή target σε binary (0 ή 1)
        y = (data.target.astype(str).str.strip() == '>50K').astype(int).values
        
        numeric_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
        categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

        # Ensure correct column ordering before preprocessing
        X = X[numeric_features + categorical_features].copy()
        
        # Μετατροπή κατηγορικών σε string για συμβατότητα
        for col in categorical_features:
            X[col] = X[col].astype(str)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Σώζουμε το raw Test Data σε CSV για το upload του Dataset
        test_csv_path = os.path.join(ARTIFACTS_DIR, "adult_dl_test.csv")
        logger.info(f"💾 Saving raw test data to {test_csv_path}...")
        test_df = X_test.copy()
        test_df['target'] = y_test 
        test_df.to_csv(test_csv_path, index=False)

        # 2. Define Preprocessing (Crucial for Tabular data even with PyTorch)
        logger.info("⚙️ Defining and fitting Preprocessor (Sklearn)...")
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Fit on training data
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train).astype(np.float32)
        
        num_features = X_train_processed.shape[1]
        logger.info(f"✅ Preprocessing complete. Feature dimension after One-Hot: {num_features}")

        # Save preprocessor
        preprocessor_path = os.path.join(ARTIFACTS_DIR, "adult_preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"💾 Preprocessor saved to {preprocessor_path}")

        # 3. Prepare PyTorch Tensors & DataLoaders
        X_train_tensor = torch.from_numpy(X_train_processed)
        y_train_tensor = torch.from_numpy(y_train).float()
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Dummies for ONNX export
        dummy_input_mlp = torch.randn(1, num_features)
        dummy_input_cnn = torch.randn(1, 1, num_features) # (Batch, Channels, Features)

        # ======================================================================
        # PART A: MLP (Fully Connected)
        # ======================================================================
        logger.info("🧠 Training Adult MLP (PyTorch)...")
        mlp_model = AdultMLP1(input_dim=num_features)
        train_pytorch_model(mlp_model, train_loader, epochs=5) # 5 epochs is enough for functional artifacts
        mlp_model.eval()

        # Export MLP to ONNX FP32
        mlp_fp32_path = os.path.join(ARTIFACTS_DIR, "adult_mlp_fp32.onnx")
        logger.info(f"🔄 Exporting MLP to ONNX (FP32): {mlp_fp32_path}")
        torch.onnx.export(
            mlp_model, 
            dummy_input_mlp, 
            mlp_fp32_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        # temp_model = onnx.load(mlp_fp32_path)
        m = onnx.load(mlp_fp32_path)
        # Διαγράφουμε τις πληροφορίες σχήματος που προκαλούν το conflict
        for _ in range(len(m.graph.value_info)): m.graph.value_info.pop()
        onnx.save(m, mlp_fp32_path)

        # onnx.save_model(temp_model, mlp_fp32_path, save_as_external_data=False)

        # Quantize MLP to INT8
        mlp_int8_path = os.path.join(ARTIFACTS_DIR, "adult_mlp_int8.onnx")
        logger.info(f"⚡ Quantizing MLP to INT8: {mlp_int8_path}")
        quantize_dynamic(mlp_fp32_path, mlp_int8_path, weight_type=QuantType.QUInt8, extra_options={'EnableSubgraph': True})

        # ======================================================================
        # PART B: 1D CNN
        # ======================================================================
        logger.info("📡 Training Adult 1D CNN (PyTorch)...")
        cnn_model = AdultCNN1D(input_dim=num_features)
        
        # Prepare data for CNN training (requires reshape to include channel dim)
        X_train_cnn = X_train_tensor.unsqueeze(1) # (Batch, Features) -> (Batch, 1, Features)
        train_dataset_cnn = TensorDataset(X_train_cnn, y_train_tensor)
        train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True)
        
        train_pytorch_model(cnn_model, train_loader_cnn, epochs=5)
        cnn_model.eval()

        # Export CNN to ONNX FP32
        cnn_fp32_path = os.path.join(ARTIFACTS_DIR, "adult_cnn_fp32.onnx")
        logger.info(f"🔄 Exporting CNN to ONNX (FP32): {cnn_fp32_path}")
        torch.onnx.export(
            cnn_model, 
            dummy_input_cnn, 
            cnn_fp32_path,
            export_params=True,
            opset_version=12,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        m = onnx.load(cnn_fp32_path)
        # Διαγράφουμε τις πληροφορίες σχήματος που προκαλούν το conflict
        for _ in range(len(m.graph.value_info)): m.graph.value_info.pop()
        onnx.save(m, cnn_fp32_path)
        # temp_model_cnn = onnx.load(cnn_fp32_path)
        # onnx.save_model(temp_model_cnn, cnn_fp32_path, save_as_external_data=False)

        # Quantize CNN to INT8
        cnn_int8_path = os.path.join(ARTIFACTS_DIR, "adult_cnn_int8.onnx")
        logger.info(f"⚡ Quantizing CNN to INT8: {cnn_int8_path}")
        quantize_dynamic(cnn_fp32_path, cnn_int8_path, weight_type=QuantType.QUInt8)

        logger.info("✅ SUCCESS! Adult Deep Learning artifacts created in 'artifacts_deep_learning/'")
        return {
            "message": "Adult Deep Learning artifacts generated successfully",
            "files": [
                test_csv_path,
                preprocessor_path,
                mlp_fp32_path, mlp_int8_path,
                cnn_fp32_path, cnn_int8_path
            ]
        }

    except Exception as e:
        logger.error(f"❌ Error during artifact generation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))