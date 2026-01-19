# setup_models.py
import torch
import os
from backend.ai_models.mlp import MaintenanceMLP


MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_dummy_mlp():
    print("Creating dummy MLP model...")
    
    model = MaintenanceMLP(input_size=512, hidden_size=1024, num_classes=2)
    
    
    save_path = os.path.join(MODEL_DIR, "mlp_maintenance_v1.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully at: {save_path}")

if __name__ == "__main__":
    save_dummy_mlp()