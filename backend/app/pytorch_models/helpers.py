import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# def train_pytorch_model(model, train_loader, epochs=10):
#     criterion = nn.BCELoss() # Binary Cross Entropy
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels.unsqueeze(1))
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         # Optional: logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")


def train_pytorch_model(model, train_loader, epochs=10, task="classification"):
    """
    Trains a PyTorch model.
    task: 'classification' (uses BCELoss) or 'regression' (uses MSELoss)
    """
    # Select the appropriate loss function based on the task
    if task == "classification":
        criterion = nn.BCELoss() # For Adult Income (0 or 1)
    elif task == "regression":
        criterion = nn.MSELoss() # For California Housing (Continuous values)
    else:
        raise ValueError(f"Unsupported task type: {task}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # labels.unsqueeze(1) transforms shape from [batch] to [batch, 1] to match outputs
            loss = criterion(outputs, labels.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()