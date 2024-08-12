import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from src.utils import parse_molecule_graph, molgraph_to_pyg

class GNNWithFeatures(torch.nn.Module):
    def __init__(self, num_node_features, num_additional_features, hidden_channels):
        super(GNNWithFeatures, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels + num_additional_features, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, 2)

    def forward(self, data, additional_features):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)
        x = torch.cat([x, additional_features], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def prepare_data(df):
    df['parsed_molecule_graph'] = df['molecule_graph'].apply(parse_molecule_graph)
    graphs = [molgraph_to_pyg(mg) for mg in df['parsed_molecule_graph']]

    numeric_columns = df.select_dtypes(include=[float, int]).columns.tolist()
    non_numeric_columns = ['bonds', 'composition', 'elements', 'formula_alphabetical', 'molecule',
                           'molecule_graph', 'partial_charges', 'partial_spins', 'species', 'xyz', 'vibration_frequencies']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[non_numeric_columns])

    features_combined = np.hstack((df[numeric_columns].values, encoded))
    return graphs, features_combined, df[['electronic_energy', 'vibration_frequencies_avg']].values

def create_data_loader(graphs, features, targets, batch_size=32):
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, graphs, additional_features, targets):
            self.graphs = graphs
            self.additional_features = additional_features
            self.targets = targets

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, idx):
            return self.graphs[idx], torch.tensor(self.additional_features[idx], dtype=torch.float), torch.tensor(self.targets[idx], dtype=torch.float)

    dataset = CustomDataset(graphs, features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_gnn(model, train_loader, optimizer, epochs=100, device='cpu'):
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        for data, additional_features, target in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            additional_features = additional_features.to(device)
            target = target.to(device)
            out = model(data, additional_features)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')

def evaluate_gnn(model, loader, device='cpu'):
    model = model.to(device)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, additional_features, target in loader:
            data = data.to(device)
            additional_features = additional_features.to(device)
            target = target.to(device)
            out = model(data, additional_features)
            preds.append(out.cpu().numpy())
            targets.append(target.cpu().numpy())
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    return preds, targets

