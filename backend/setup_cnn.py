import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

from ai_models.cnn import SimpleCNN


# Define paths
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODEL_DIR, "cnn_mnist_v1.pth")

def train_and_save_cnn():
    print("--- Starting CNN Setup ---")
    
    # 1. Download MNIST Data (The Images) to train
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization
    ])
    
    print("Downloading MNIST dataset... (this might take a moment)")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 2. Initialize Model
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Train for 1 Epoch (Just enough to learn the shapes)
    print("Training model (1 Epoch)...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
            
    # 4. Save the Model
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Model saved successfully at: {SAVE_PATH}")

if __name__ == "__main__":
    train_and_save_cnn()