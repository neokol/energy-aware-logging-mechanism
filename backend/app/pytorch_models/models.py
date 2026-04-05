import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class AdultMLP1(nn.Module):
    def __init__(self, input_dim):
        super(AdultMLP1, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Binary Classification
        )

    def forward(self, x):
        return self.network(x)

# 2. 1D CNN
class AdultCNN1D(nn.Module):
    def __init__(self, input_dim):
        super(AdultCNN1D, self).__init__()
        # Input shape expected: (Batch, Channels=1, Features)
        self.conv_subsytem = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Halves the feature dimension
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8) # Forces output width to 8 regardless of input
        )
        
        # Flattened size = out_channels (3 adaptive_out (8) = 256
        self.fc_subsystem = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape at start: (Batch, Features)
        # We need to add the channel dimension: (Batch, 1, Features)
        # However, we will handle the reshape in the Inference Strategy,
        # so here we assume x comes in as (Batch, 1, Features)
        x = self.conv_subsytem(x)
        # x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = x.view(x.size(0), -1)
        x = self.fc_subsystem(x)
        return x
    
# ======================================================================
# CALIFORNIA HOUSING MODELS (REGRESSION)
# ======================================================================

class HousingMLP(nn.Module):
    def __init__(self, input_dim=8):
        super(HousingMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # CRITICAL: No Sigmoid here! We are predicting continuous house prices (Regression).
        )

    def forward(self, x):
        return self.network(x)


class HousingCNN1D(nn.Module):
    def __init__(self, input_dim=8):
        super(HousingCNN1D, self).__init__()
        
        self.conv_subsytem = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Halves the spatial dimension
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4) # Forces the output width to exactly 4
        )
        
        # Flattened size: 32 (channels) * 4 (width) = 128 features
        self.fc_subsystem = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
            # CRITICAL: No Sigmoid here!
        )

    def forward(self, x):
        x = self.conv_subsytem(x)
        
        # Use -1 for dynamic batch size handling during ONNX export
        x = x.view(-1, 128) 
        
        x = self.fc_subsystem(x)
        return x