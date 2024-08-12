import torch
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Element

def parse_molecule_graph(molecule_graph_str):
    molecule_graph_dict = eval(molecule_graph_str)
    return MoleculeGraph.from_dict(molecule_graph_dict)

def molgraph_to_pyg(molecule_graph):
    atom_features = []
    for node in molecule_graph.graph.nodes(data=True):
        specie = Element(node[1]['specie'])
        atom_features.append(specie.Z)

    edge_index = []
    for edge in molecule_graph.graph.edges:
        edge_index.append([edge[0], edge[1]])
        edge_index.append([edge[1], edge[0]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    return torch_geometric.data.Data(x=x, edge_index=edge_index)

def flatten_data(graphs, features, num_samples=100):
    max_num_nodes = max([graph.num_nodes for graph in graphs])
    flattened_graphs = []
    for graph in graphs:
        flattened_graph = torch.cat([graph.x.flatten(), torch.zeros(max_num_nodes * graph.x.size(1) - graph.num_nodes * graph.x.size(1))])
        flattened_graphs.append(flattened_graph)
    combined_data = torch.cat([torch.stack(flattened_graphs), features], dim=1)
    return combined_data.numpy()[:num_samples], features[:num_samples], num_samples

def weighted_mse_loss(predictions, targets, weights):
    return torch.mean(weights * (predictions - targets) ** 2)

