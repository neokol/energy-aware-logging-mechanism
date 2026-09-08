import torch
import torch.nn as nn

class MaintenanceMLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_classes=2):
        super(MaintenanceMLP, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        """
        Forward pass through the network
        """
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)
        return out
