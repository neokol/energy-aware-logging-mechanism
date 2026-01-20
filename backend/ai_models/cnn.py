import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Layer 1: Convolution
        # Input: 1 channel (grayscale), Output: 16 filters, Kernel: 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2) # Shrinks 28x28 -> 14x14
        
        # Layer 2: Convolution
        # Input: 16 channels, Output: 32 filters
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2) # Shrinks 14x14 -> 7x7
        
        # Fully Connected Layer (Classification)
        # 32 channels * 7 * 7 pixels = 1568 inputs
        self.fc = nn.Linear(32 * 7 * 7, 10) # Output: 10 digits (0-9)

    def forward(self, x):
        # x shape starts as: [Batch, 1, 28, 28]
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten: Turn the 3D cube (32x7x7) into a flat line for the final decision
        x = x.view(x.size(0), -1) 
        
        x = self.fc(x)
        return x