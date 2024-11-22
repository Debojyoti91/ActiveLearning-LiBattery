import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, test_loader, scaler_y, target_columns, plot_path):
    """
    Evaluate the trained model and generate actual vs. predicted plots.
    """
    model.eval()
    actuals, predictions = [], []

    with torch.no_grad():
        for data, targets in test_loader:
            out = model(data)
            actuals.append(targets.numpy())
            predictions.append(out.numpy())

    actuals = np.concatenate(actuals, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    # Convert to original scale
    actuals_original = scaler_y.inverse_transform(actuals)
    predictions_original = scaler_y.inverse_transform(predictions)

    # Plot actual vs. predicted for each target
    plt.figure(figsize=(14, 12))
    for i, target_name in enumerate(target_columns):
        plt.subplot(2, 3, i + 1)
        plt.scatter(actuals_original[:, i], predictions_original[:, i], alpha=0.6)
        plt.xlabel(f"Actual {target_name}")
        plt.ylabel(f"Predicted {target_name}")
        plt.title(f"{target_name}")
    plt.tight_layout()
    plt.savefig(f"{plot_path}/actual_vs_predicted.png")
    plt.show()

