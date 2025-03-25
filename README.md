# Lithium-Ion Battery Property Prediction Using GNN and Active Learning

## Project Overview

This project focuses on predicting lithium-ion battery properties using advanced machine learning techniques, including **Graph Neural Networks (GNNs)**, **SHAP analysis**, and **active learning strategies**. The pipeline is structured to provide robust predictions, interpret model behavior, and enhance performance using minimal data via active learning.

## Project Contents

- `src/`: Source code for the project, including data processing, GNN modeling, SHAP analysis, and active learning.
  - `data_preparation.py`: Handles loading, preprocessing, and saving of data.
  - `model_definition.py`: Defines the GNN model architecture and graph representation.
  - `train.py`: Implements the training loop for the GNN and active learning strategies.
  - `evaluate.py`: Evaluates the trained model and generates visualizations.
  - `shap_analysis.py`: Conducts SHAP-based feature importance analysis.
- `notebooks/`: Google Colab notebooks are provided in the **notebook** folder for direct and accurate implementation of these models.

**Cite this work:** 

If you use this repository, codes in your research or work, please cite the following publication:

Das, D. and Chakraborty, D. (2025). *Machine Learning Prediction of Physicochemical Properties in Lithium-Ion Battery Electrolytes With Active Learning Applied to Graph Neural Networks*. Journal of Computational Chemistry, 46: e70009.  
https://doi.org/10.1002/jcc.70009


## Features

- **Graph Neural Network (GNN) Modeling**:
  - Predicts key properties:
    - **Electronic Energy**
    - **Total Enthalpy**
    - **Total Entropy**
    - **Free Energy**
    - **Vibration Frequencies**
  - **molecular graph representations as input**.
  
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

## Processed Data (Utilized for this study)

This project uses the **Lithium-Ion Battery (LIB)** Electrolyte datasets, containing detailed properties of lithium-ion battery electrolytes. These are the final, processed datasets, adapted from the original datasets for training and evaluating the models developed in this project.

**Dataset Link**: [LIBE Dataset](https://drive.google.com/drive/folders/1-gLGgO4IJUV73uG8xS2VqtipaGXenuTV?usp=sharing)

## Tools and Technologies

- **Python 3.8+**: Core programming language.
- **PyTorch & PyTorch Geometric**: For GNN modeling.
- **SHAP**: For feature importance analysis and interpretability.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For metrics and preprocessing.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Google Colab**: For Computation on GPU

## Original Datasets

The project utilizes the Lithium-Ion Battery Electrolyte (LIBE) dataset and MPcules dataset, which contains comprehensive data related to the properties and performance of lithium-ion battery electrolytes. 

**Citation:**  
1. Spotte-Smith, Evan Walter Clark; Blau, Samuel M.; Xie, Xiaowei; Patel, Hetal; Wood, Brandon; Dwaraknath, Shyam; et al. (2021). Lithium-Ion Battery Electrolyte (LIBE) dataset. figshare. Dataset. [https://doi.org/10.6084/m9.figshare.14226464.v2](https://doi.org/10.6084/m9.figshare.14226464.v2)
2. Spotte-Smith, E. W. C.; Cohen, O. A.; Blau, S. M.; Munro, J. M.; Yang, R.; Guha, R. D.; Patel, H. D.; Vijay, S.; Huck, P.; Kingsbury, R.; et al. A database of molecular properties integrated in the Materials Project. Digital Discovery 2023, 2 (6), 1862-1882, 

## Tools and Technologies

- **Python 3.8**: Core programming language used for the project.
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **scikit-learn**: For machine learning algorithms and data preprocessing.
- **PyTorch & torch-geometric**: For building and training Graph Neural Networks.
- **pymatgen**: For materials science data manipulation.
- **SHAP**: For explainable AI and model interpretability.

## Dependencies and Installation 

Before running the code, install the following Python libraries:

pip install torch
pip install shap==0.46.0
pip install scikit-learn==1.5.2
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
pip install pymatgen
pip install matminer==0.9.3
pip install torch-geometric==2.6.1


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









