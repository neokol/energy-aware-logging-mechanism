import torch
import torch.nn as nn

class MaintenanceMLP(nn.Module):
    def __init__(self, input_size=512, hidden_size=1024, num_classes=2):
        super(MaintenanceMLP, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        # Activation Function (ReLU)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the network
        """
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out