import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np

def train_model(train_loader, test_loader, target_columns, hidden_channels, epochs, model_path):
    """
    Train the GNN model and perform active learning.
    """
    model = GNNModel(hidden_channels, num_node_features=5, output_features=len(target_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for data, targets in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader))

        # Testing phase
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data, targets in test_loader:
                out = model(data)
                loss = F.mse_loss(out, targets)
                total_test_loss += loss.item()

        test_losses.append(total_test_loss / len(test_loader))
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}")

    torch.save(model.state_dict(), model_path)
    return model, train_losses, test_losses

