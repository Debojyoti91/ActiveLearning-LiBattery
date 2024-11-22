import pandas as pd
import torch
from torch_geometric.data import Data
from pymatgen.core import Element
import re


def parse_molecule_graph(graph_str):
    """Parse molecule graph representation into atom features and bonds."""
    lines = graph_str.splitlines()
    atoms = []
    bonds = []
    for line in lines:
        if line.startswith("Site:"):
            element = re.search(r'Site: (\w+)', line).group(1)
            atoms.append(Element(element).Z)
    if "from    to  to_image      weight" in lines:
        bond_start_idx = lines.index("from    to  to_image      weight") + 1
        for line in lines[bond_start_idx:]:
            bond_parts = line.split()
            if bond_parts[0].isdigit():
                from_idx = int(bond_parts[0])
                to_idx = int(bond_parts[1])
                bonds.append([from_idx, to_idx])
    return atoms, bonds


def graph_to_pyg(row, numeric_features):
    """Convert parsed graph data to PyTorch Geometric Data format."""
    try:
        atoms, bonds = parse_molecule_graph(row['molecule_graph'])
        atom_features = []
        for atomic_number in atoms:
            node_features = [atomic_number]
            node_features.extend([row[feat] for feat in numeric_features])
            atom_features.append(node_features)
        edge_index = []
        for bond in bonds:
            edge_index.append([bond[0], bond[1]])
            edge_index.append([bond[1], bond[0]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(atom_features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)
    except Exception as e:
        print(f"Error converting graph to PyG Data: {e}")
        return None


def preprocess_data(df, target_columns, numeric_features):
    """Preprocess dataset and prepare PyG graphs."""
    df['pyg_graph'] = df.apply(lambda row: graph_to_pyg(row, numeric_features), axis=1)
    df = df.dropna(subset=['pyg_graph'])
    y = df[target_columns].values
    return df, y

