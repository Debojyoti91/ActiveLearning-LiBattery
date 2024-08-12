import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir='reports/figures'):
    os.makedirs(output_dir, exist_ok=True)

    # Missing Values Analysis
    missing_values = df.isnull().sum()
    plt.figure(figsize=(10, 6))
    plt.bar(df.columns, df.isnull().sum())
    plt.xticks(rotation=90)
    plt.title('Missing Values Count per Column')
    plt.savefig(os.path.join(output_dir, 'missing_values.png'))
    plt.show()

    # Distribution of Numerical Columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(output_dir, f'distribution_{col}.png'))
        plt.show()

    # Correlation Heatmap
    correlation_matrix = df[num_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.show()

