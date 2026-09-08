import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from ai_models.cnn import SimpleCNN

SEED = 42
EPOCHS = 5

MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODEL_DIR, "cnn_mnist_v1.pth")

# Same normalisation must be applied at inference time (app/services/cnn_service.py)
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def _accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += len(target)
    return correct / total


def train_and_save_cnn():
    print("--- Starting CNN Setup ---")
    torch.manual_seed(SEED)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])

    print("Loading MNIST ...")
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Training model ({EPOCHS} epochs) ...")
    for epoch in range(EPOCHS):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{EPOCHS}  train_acc={_accuracy(model, train_loader):.4f}  "
              f"test_acc={_accuracy(model, test_loader):.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nModel saved successfully at: {SAVE_PATH}")
    print(f"Final held-out test accuracy: {_accuracy(model, test_loader):.4f}")


if __name__ == "__main__":
    train_and_save_cnn()
