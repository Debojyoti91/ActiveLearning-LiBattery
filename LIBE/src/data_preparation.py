import pandas as pd
import torch
from torch_geometric.data import Data
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Element
from sklearn.preprocessing import StandardScaler

def parse_molecule_graph(molecule_graph_str):
    """
    Parse molecule graph strings into MoleculeGraph objects.
    """
    try:
        molecule_graph_dict = eval(molecule_graph_str)
        return MoleculeGraph.from_dict(molecule_graph_dict)
    except Exception as e:
        print(f"Error parsing molecule graph: {e}")
        return None

def molgraph_to_pyg(row, numeric_feature_columns):
    """
    Convert MoleculeGraph to PyTorch Geometric Data object.
    """
    try:
        atom_features = []
        bonds = eval(row['bonds'])
        for site in row['molecule_graph'].molecule:
            element = Element(site.specie.symbol)
            node_features = [element.Z] + row[numeric_feature_columns].values.tolist()
            atom_features.append(node_features)

        edge_index = []
        for bond in bonds:
            edge_index.append([bond[0], bond[1]])
            edge_index.append([bond[1], bond[0]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(atom_features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)
    except Exception as e:
        print(f"Error converting MoleculeGraph to PyG Data: {e}")
        return None

def preprocess_data(file_path, target_columns, numeric_feature_columns):
    """
    Preprocess dataset: load data, parse graphs, normalize targets, and create PyTorch Geometric graphs.
    """
    df = pd.read_csv(file_path)
    df['molecule_graph'] = df['molecule_graph'].apply(parse_molecule_graph)
    df = df.dropna(subset=['molecule_graph'])

    y = df[target_columns].values
    scaler_y = StandardScaler()
    y_normalized = scaler_y.fit_transform(y)

    df['pyg_graph'] = df.apply(lambda row: molgraph_to_pyg(row, numeric_feature_columns), axis=1)
    df = df.dropna(subset=['pyg_graph'])

    return df, y_normalized, scaler_y

