import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


def evaluate(predictions, actuals, target_columns):
    """Evaluate the model's performance."""
    for i, target in enumerate(target_columns):
        r2 = r2_score(actuals[:, i], predictions[:, i])
        mae = mean_absolute_error(actuals[:, i], predictions[:, i])
        rmse = mean_squared_error(actuals[:, i], predictions[:, i], squared=False)
        pearson_corr = pearsonr(actuals[:, i], predictions[:, i])[0]
        print(f"Metrics for {target}: R²={r2}, MAE={mae}, RMSE={rmse}, Pearson={pearson_corr}")


def plot_results(predictions, actuals, target_columns, output_path):
    """Plot Actual vs Predicted values."""
    plt.figure(figsize=(12, 10))
    for i, target in enumerate(target_columns):
        plt.subplot(2, 3, i + 1)
        plt.scatter(actuals[:, i], predictions[:, i])
        plt.plot([actuals[:, i].min(), actuals[:, i].max()],
                 [actuals[:, i].min(), actuals[:, i].max()], 'k--')
        plt.xlabel(f"Actual {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"{target}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

