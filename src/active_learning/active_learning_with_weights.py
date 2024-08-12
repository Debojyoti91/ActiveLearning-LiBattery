import numpy as np
import pickle
import torch
from torch_geometric.data import DataLoader
from src.gnn_model import train_gnn, evaluate_gnn
from src.utils import weighted_mse_loss

def active_learning_loop_with_weights(model, train_loader, test_loader, optimizer, n_steps=10, device='cpu'):
    all_train_losses, all_test_losses = [], []

    for step in range(n_steps):
        train_loss = train_gnn(model, train_loader, optimizer, device=device)
        test_loss, preds, targets = evaluate_gnn(model, test_loader, device=device)

        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)

        print(f'Step {step}, Train Loss: {train_loss}, Test Loss: {test_loss}')

        # Apply weights based on importance (using SHAP or other methods)
        weights = np.ones(preds.shape)
        weights[:, 0] *= 2.0  # Example: Double weight for first target

        # Maximum Expected Improvement (MEI)
        mei_selection = np.argmax(preds)

        # Maximum Likelihood of Improvement (MLI)
        max_in_train = np.max(targets)
        mli_selection = np.argmax((preds - max_in_train) / (np.std(preds) + 1e-9))

        # Maximum Uncertainty (MU)
        mu_selection = np.argmax(np.std(preds, axis=1))

        # Add the selected points to the training set (simplified)
        if len(train_loader.dataset) < len(test_loader.dataset):
            train_loader.dataset.graphs.append(test_loader.dataset.graphs.pop(mei_selection))
            train_loader.dataset.additional_features = np.vstack(
                [train_loader.dataset.additional_features, test_loader.dataset.additional_features[mei_selection]])
            train_loader.dataset.targets = np.vstack(
                [train_loader.dataset.targets, test_loader.dataset.targets[mei_selection]])

    with open('data/results/active_learning_with_weights_results.pkl', 'wb') as f:
        pickle.dump({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': all_train_losses,
            'test_losses': all_test_losses
        }, f)

    return all_train_losses, all_test_losses

