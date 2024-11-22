import shap
from torch_geometric.data import Batch

def generate_shap_analysis(model, test_graphs, target_columns, num_node_features, plot_path):
    """
    Perform SHAP analysis on the GNN model.
    """
    explainer = shap.KernelExplainer(model.predict, test_graphs[:10])
    shap_values = explainer.shap_values(test_graphs)

    for i, target_name in enumerate(target_columns):
        shap.summary_plot(shap_values[i], feature_names=[f"Feature {j}" for j in range(num_node_features)])

