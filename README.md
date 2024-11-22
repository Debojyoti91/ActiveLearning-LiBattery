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

## Data

The project utilizes the Lithium-Ion Battery Electrolyte (LIBE) dataset, which contains comprehensive data related to the properties and performance of lithium-ion battery electrolytes. This dataset is critical for training and evaluating the machine learning models developed in this project.

**Citation:**  
1. Spotte-Smith, Evan Walter Clark; Blau, Samuel M.; Xie, Xiaowei; Patel, Hetal; Wood, Brandon; Dwaraknath, Shyam; et al. (2021). Lithium-Ion Battery Electrolyte (LIBE) dataset. figshare. Dataset. [https://doi.org/10.6084/m9.figshare.14226464.v2](https://doi.org/10.6084/m9.figshare.14226464.v2)

## Tools and Technologies

- **Python 3.8**: Core programming language used for the project.
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **scikit-learn**: For machine learning algorithms and data preprocessing.
- **PyTorch & torch-geometric**: For building and training Graph Neural Networks.
- **pymatgen**: For materials science data manipulation.
- **SHAP**: For explainable AI and model interpretability.

## Project Contents

- `src/`: Source code for the project, including data processing, EDA, GNN modeling, SHAP analysis, and active learning.
  - `data_processing.py`: Handles loading, preprocessing, and saving of data.
  - `eda.py`: Contains scripts for exploratory data analysis, including missing value analysis, distribution plots, and correlation heatmaps.
  - `gnn_model.py`: Defines the GNN model, training routines, and data preparation functions.
  - `shap_analysis.py`: Implements SHAP analysis for model interpretation.
  - `active_learning/`: Contains scripts for active learning with and without weighted loss functions.
  - `fine_tuning.py`: Handles fine-tuning of the model using active learning and plotting the results.
  - `utils.py`: Utility functions for parsing and processing data.
  - `main.py`: The central script that orchestrates the entire pipeline, from data processing to model training and evaluation.
- `data/`: Contains the dataset used in the project.
  - `raw/`: Stores the raw LIBE dataset.
  - `processed/`: Stores processed datasets.
- `results/`: Contains results such as model outputs, SHAP values, and active learning results.
- `reports/`: Contains figures, plots, and summaries generated during the project.
  - `figures/`: Stores visualizations such as distribution plots, correlation heatmaps, and SHAP summary plots.
  - `summary.md`: A summary of the project findings and results.
- `environment.yml`: Specifies the Python environment and dependencies for easy setup.
- `README.md`: Provides an overview of the project, installation instructions, and usage guidelines.
- `LICENSE`: The license under which the project is distributed.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Debojyoti91/ActiveLearning-LiBattery.git
   cd ActiveLearning-LiBattery

## Installation and Usage

```bash
# Clone the repository:
git clone https://github.com/Debojyoti91/ActiveLearning-LiBattery.git
cd ActiveLearning-LiBattery

# Create and activate the environment:
conda env create -f environment.yml
conda activate li_battery_modeling

# Download and place the dataset:
# Download the LIBE dataset as described above and place it in the data/raw/ directory.

# To run the entire modeling pipeline:
python main.py

# To run the pipeline with the active learning strategy using weighted loss:
python main.py --use_weighted_loss

