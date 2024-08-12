import shap
import pickle
import torch
from src.utils import flatten_data

class ShapGNNWrapper:
    def __init__(self, model, num_node_features, num_additional_features):
        self.model = model
        self.num_node_features = num_node_features
        self.num_additional_features = num_additional_features

    def predict(self, data):
        data = torch.tensor(data, dtype=torch.float32)
        num_graph_features = self.num_node_features
        num_additional_features = self.num_additional_features

        graph_features = data[:, :num_graph_features]
        additional_features = data[:, num_graph_features:num_graph_features + num_additional_features]

        graph_data_list = []
        for i in range(graph_features.shape[0]):
            graph_feature = graph_features[i]
            num_nodes = len(graph_feature) // self.num_node_features
            node_features = graph_feature[:num_nodes * self.num_node_features].view(num_nodes, self.num_node_features)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph_data_list.append(Data(x=node_features, edge_index=edge_index))

        batch = torch_geometric.data.Batch.from_data_list(graph_data_list)
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch, additional_features)
        return output.numpy()

def run_shap_analysis(model, test_graphs, test_features, num_samples=100):
    sampled_test_graphs, sampled_test_features, _ = flatten_data(test_graphs, test_features, num_samples)
    wrapper = ShapGNNWrapper(model, num_node_features=1, num_additional_features=test_features.shape[1])

    explainer = shap.KernelExplainer(wrapper.predict, sampled_test_features[:10])
    shap_values = explainer.shap_values(sampled_test_features)

    with open('data/results/shap_values.pkl', 'wb') as f:
        pickle.dump(shap_values, f)

