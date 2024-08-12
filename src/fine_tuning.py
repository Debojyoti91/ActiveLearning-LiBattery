import torch
from src.gnn_model import train_gnn, evaluate_gnn
from src.active_learning.active_learning_with_weights import active_learning_loop_with_weights
from src.active_learning.active_learning_no_weights import active_learning_loop_no_weights

def fine_tune_model(model, train_loader, test_loader, optimizer, n_steps=50, use_weights=False, device='cpu'):
    if use_weights:
        active_learning_loop_with_weights(model, train_loader, test_loader, optimizer, n_steps=n_steps, device=device)
    else:
        active_learning_loop_no_weights(model, train_loader, test_loader, optimizer, n_steps=n_steps, device=device)

def plot_results(train_losses, test_losses):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Losses over Fine-Tuning Steps')
    plt.grid(True)
    plt.savefig('reports/figures/fine_tuning_losses.png')
    plt.show()

