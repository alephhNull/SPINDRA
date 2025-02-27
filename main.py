# main.py
import random
import numpy as np
import torch
# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms(True)

# set_seed(42)  #

from data_loader import load_data, prepare_tensors, spatial_to_graph
from models import SpatialEncoder, BulkEncoder, SingleCellEncoder, DrugResponsePredictor, DomainDiscriminator, ImprovedSpatialEncoder, grad_reverse
from trainer import train_model, predict_spatial
from visualize import visualize_and_evaluate

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load and preprocess data
    spatial_data, bulk_data, sc_tumor_data, sc_cellline_data, common_genes = load_data()
    domain_data = prepare_tensors(spatial_data, bulk_data, sc_tumor_data, sc_cellline_data, device)
    edge_index, edge_weights = spatial_to_graph(spatial_data, k=10, device=device)  # Updated to receive edge_weights
    edge_weights = None

    spatial_encoder = SpatialEncoder(input_dim=len(common_genes), edge_index=edge_index).to(device)
    # spatial_encoder = ImprovedSpatialEncoder(input_dim=len(common_genes), edge_index=edge_index, num_layers=3).to(device)
    bulk_encoder = BulkEncoder(input_dim=len(common_genes)).to(device)
    sc_encoder = SingleCellEncoder(input_dim=len(common_genes)).to(device)
    drug_predictor = DrugResponsePredictor(hidden_dim=128).to(device)
    discriminator = DomainDiscriminator(hidden_dim=128).to(device)

    spatial_z = train_model(
        spatial_encoder, bulk_encoder, sc_encoder, drug_predictor, discriminator,
        domain_data, edge_index, device, num_epochs=1000
    )
    spatial_X, _ = domain_data['spatial']
    spatial_z, spatial_pred_labels, spatial_pred_probs = predict_spatial(
        spatial_encoder, drug_predictor, spatial_X, edge_index, edge_weights  # Pass edge_weights
    )
    print(f"Total spatial cells: {spatial_pred_labels.shape[0]}")
    print(f"Sensitive cells: {spatial_pred_labels.sum().item()}")

    visualize_and_evaluate(spatial_data, spatial_z, spatial_pred_probs)

if __name__ == "__main__":
    main()