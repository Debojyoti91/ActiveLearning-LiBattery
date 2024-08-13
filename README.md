# ActiveLearning-LiBattery

## Project Overview

This project focuses on modeling lithium battery performance using advanced machine learning techniques, including Graph Neural Networks (GNNs), SHAP analysis, and active learning strategies. The project is structured as a complete machine learning pipeline, starting from data processing and exploratory data analysis (EDA) to model training, interpretation, and fine-tuning. The goal is to leverage these techniques to better understand and predict the behavior of lithium-ion batteries.

## Features

- Implementation of Graph Neural Networks (GNNs) for lithium battery performance modeling.
- SHAP analysis for model interpretability and feature importance.
- Active learning strategies to improve model performance with minimal data.
- Fine-tuning of models based on active learning results.

## Data

The project utilizes the Lithium-Ion Battery Electrolyte (LIBE) dataset, which contains comprehensive data related to the properties and performance of lithium-ion battery electrolytes. This dataset is critical for training and evaluating the machine learning models developed in this project.

**Citation:**  
Spotte-Smith, Evan Walter Clark; Blau, Samuel M.; Xie, Xiaowei; Patel, Hetal; Wood, Brandon; Dwaraknath, Shyam; et al. (2021). Lithium-Ion Battery Electrolyte (LIBE) dataset. figshare. Dataset. [https://doi.org/10.6084/m9.figshare.14226464.v2](https://doi.org/10.6084/m9.figshare.14226464.v2)

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

