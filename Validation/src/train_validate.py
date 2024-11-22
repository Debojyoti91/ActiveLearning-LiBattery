import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np


def train(model, train_loader, optimizer):
    """Train the GNN model."""
    model.train()
    total_loss = 0
    for data, targets in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(model, test_loader):
    """Evaluate the GNN model."""
    model.eval()
    total_loss = 0
    predictions, actuals = [], []
    with torch.no_grad():
        for data, targets in test_loader:
            out = model(data)
            loss = F.mse_loss(out, targets)
            total_loss += loss.item()
            predictions.append(out.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    return total_loss / len(test_loader), np.concatenate(predictions), np.concatenate(actuals)


def active_learning_loop(model, train_data, train_targets, test_data, test_targets, optimizer, n_steps):
    """Implement the active learning loop."""
    train_losses, test_losses = [], []

    for step in range(n_steps):
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        train_loss = train(model, train_loader, optimizer)
        test_loss, predictions, actuals = test(model, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        uncertainties = np.abs(predictions - actuals)
        mu_selection = np.argmax(uncertainties[:, 0])
        train_data.append(test_data[mu_selection])
        train_targets = np.vstack((train_targets, test_targets[mu_selection]))
        del test_data[mu_selection]
        test_targets = np.delete(test_targets, mu_selection, axis=0)

    return train_losses, test_losses, predictions, actuals

