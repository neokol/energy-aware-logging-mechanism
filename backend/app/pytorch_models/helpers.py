import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def train_pytorch_model(
    model,
    train_loader,
    epochs=10,
    task="classification",
    val_loader=None,
    pos_weight=None,
    seed=42,
    patience=10,
    lr=1e-3,
):
    """
    Trains a PyTorch model.

    task        : 'classification' (BCELoss, models end in Sigmoid) or 'regression' (MSELoss)
    val_loader  : optional; enables early stopping and restores the best-val checkpoint
    pos_weight  : optional float; weight applied to the positive class in classification
    seed        : fixed for reproducibility
    patience    : early-stopping patience (only used when val_loader is given)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if task == "classification":
        base_criterion = nn.BCELoss(reduction="none" if pos_weight else "mean")
    elif task == "regression":
        base_criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported task type: {task}")

    def compute_loss(outputs, labels):
        target = labels.unsqueeze(1)
        loss = base_criterion(outputs, target)
        if task == "classification" and pos_weight:
            weights = torch.where(target > 0.5, float(pos_weight), 1.0)
            loss = (loss * weights).mean()
        return loss

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        if val_loader is None:
            logger.info(f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}")
            continue

        model.eval()
        vrunning = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                vrunning += compute_loss(model(inputs), labels).item()
        val_loss = vrunning / len(val_loader)
        logger.info(
            f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model
