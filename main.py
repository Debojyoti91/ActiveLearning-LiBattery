import os
import torch
from torch.optim import Adam
from src.data_processing import load_data, preprocess_data, save_data, create_directories, load_processed_data
from src.eda import perform_eda  # Import the EDA module
from src.gnn_model import GNNWithFeatures, prepare_data, create_data_loader, train_gnn, evaluate_gnn
from src.shap_analysis import run_shap_analysis
from src.fine_tuning import fine_tune_model, plot_results

def main(use_weighted_loss=False):
    # Create directories
    create_directories()

    # Step 1: Data Processing
    raw_data_path = 'data/raw/libe.json'
    processed_data_path = 'data/processed/processed_data.csv'
    
    df = load_data(raw_data_path)
    df = preprocess_data(df)
    save_data(df, processed_data_path)

    # Step 2: Perform EDA
    perform_eda(df)

    # Step 3: Prepare Data
    df = load_processed_data(processed_data_path)
    graphs, features, targets = prepare_data(df)
    
    train_loader = create_data_loader(graphs, features, targets, batch_size=32)
    test_loader = create_data_loader(graphs, features, targets, batch_size=32)  # Assuming you're splitting data beforehand

    # Step 4: Initialize and Train GNN Model
    model = GNNWithFeatures(num_node_features=1, num_additional_features=features.shape[1], hidden_channels=64)
    optimizer = Adam(model.parameters(), lr=0.01)
    
    train_gnn(model, train_loader, optimizer, epochs=100)

    # Step 5: Run SHAP Analysis
    run_shap_analysis(model, graphs, features, num_samples=100)

    # Step 6: Active Learning
    fine_tune_model(model, train_loader, test_loader, optimizer, n_steps=50, use_weights=use_weighted_loss)

    # Step 7: Plot Results
    plot_results(train_losses, test_losses)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run the Li Battery Modeling Pipeline')
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use active learning with weighted loss')
    args = parser.parse_args()

    main(use_weighted_loss=args.use_weighted_loss)

