import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os

def load_data(file_path):
    return pd.read_json(file_path)

def preprocess_data(df):
    list_cols = ['bonds', 'composition', 'elements', 'molecule', 'molecule_graph', 'partial_charges',
                 'partial_spins', 'species', 'thermo', 'vibration', 'xyz']

    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    cat_cols = df.select_dtypes(include=[object]).columns.difference(list_cols)
    imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer.fit_transform(df[cat_cols])

    for col in list_cols:
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    return df

def save_data(df, path):
    df.to_csv(path, index=False)

def load_processed_data(path):
    return pd.read_csv(path)

def create_directories():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/results', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

