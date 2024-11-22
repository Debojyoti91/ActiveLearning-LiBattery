# Lithium-Ion Battery Property Prediction Using GNN and Active Learning

## Project Overview

This project focuses on predicting lithium-ion battery properties using advanced machine learning techniques, including **Graph Neural Networks (GNNs)**, **SHAP analysis**, and **active learning strategies**. The pipeline is structured to provide robust predictions, interpret model behavior, and enhance performance using minimal data via active learning.

## Features

- **Graph Neural Network (GNN) Modeling**:
  - Predicts key properties:
    - **Electronic Energy**
    - **Total Enthalpy**
    - **Total Entropy**
    - **Free Energy**
    - **Vibration Frequencies**
  - Utilizes molecular graph representations as input.
  
- **Active Learning Framework**:
  - Implements strategies to select the most informative data points:
    - **Maximum Expected Improvement (MEI)**
    - **Maximum Uncertainty (MU)**

- **SHAP Analysis**:
  - Provides interpretability by explaining feature contributions to model predictions.
  
- **Validation Dataset**:
  - Evaluates model performance on independent datasets for better generalization.

- **Visualization**:
  - Plots for actual vs. predicted values.
  - Training and validation learning curves.
  - SHAP-based feature importance visualizations.

## Preprocessed Data (Used for this study)

This project uses the **Lithium-Ion Battery Electrolyte (LIBE) dataset**, containing detailed properties of lithium-ion battery electrolytes. It is the foundation for training and evaluating the models developed in this project.

**Dataset Link**: [LIBE Dataset](https://drive.google.com/drive/folders/1-gLGgO4IJUV73uG8xS2VqtipaGXenuTV?usp=sharing)

## Tools and Technologies

- **Python 3.8+**: Core programming language.
- **PyTorch & PyTorch Geometric**: For GNN modeling.
- **SHAP**: For feature importance analysis and interpretability.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For metrics and preprocessing.
- **Pandas & NumPy**: For data manipulation and numerical operations.

## Project Structure

```plaintext
LIBE/
├── README.md                  # Documentation (this file)
├── notebooks/
│   ├── GNN_Training.ipynb     # Jupyter notebook for step-by-step training
├── src/
│   ├── data_preparation.py    # Data preprocessing and graph creation
│   ├── model_definition.py    # GNN model definition
│   ├── train.py               # Training and active learning
│   ├── evaluate.py            # Evaluation metrics and result visualization
│   ├── shap_analysis.py       # SHAP feature importance analysis
├── results/
│   ├── models/                # Trained models saved here
│   ├── plots/                 # Generated plots saved here
├── LICENSE                    # BSD-2-Clause License
└── .gitignore                 # Git ignore file


# To run the pipeline with the active learning strategy using weighted loss:
python main.py --use_weighted_loss

