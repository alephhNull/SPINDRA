# main.py
import random
import numpy as np
import torch
from data_loader import load_data, prepare_tensors, spatial_to_graph
from models import SpatialEncoder, BulkEncoder, SingleCellEncoder, DrugResponsePredictor, DomainDiscriminator, ImprovedSpatialEncoder, TumorEncoder, grad_reverse
from perform_deg import perform_deg
from trainer import train_model, predict_spatial
from visualize import visualize_and_evaluate
from cell_communication import analyze_cell_communication


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load and preprocess data
    spatial_data, bulk_data, sc_tumor_data, sc_cellline_data, common_genes = load_data()
    domain_data = prepare_tensors(spatial_data, bulk_data, sc_tumor_data, sc_cellline_data, device)

    spatial_encoder = SpatialEncoder(input_dim=len(common_genes), use_edge=True).to(device)
    # spatial_encoder = ImprovedSpatialEncoder(input_dim=len(common_genes), num_layers=2, use_gat=True).to(device)
    bulk_encoder = BulkEncoder(input_dim=len(common_genes)).to(device)
    sc_encoder = SingleCellEncoder(input_dim=len(common_genes)).to(device)
    tumor_encoder = TumorEncoder(input_dim=len(common_genes)).to(device)
    drug_predictor = DrugResponsePredictor(hidden_dim=128).to(device)
    discriminator = DomainDiscriminator(hidden_dim=128).to(device)

    spatial_z = train_model(
        spatial_encoder, bulk_encoder, sc_encoder, tumor_encoder, drug_predictor, discriminator,
        domain_data, device, num_epochs=1500, pretrain_epochs=10
    )

    edge_index, edge_weights = spatial_to_graph(spatial_data, k=10, device=device)  # Updated to receive edge_weights
    spatial_X = torch.tensor(spatial_data.X).float().to(device)
    spatial_z, spatial_pred_labels, spatial_pred_probs = predict_spatial(
        spatial_encoder, drug_predictor, spatial_X, edge_index, edge_weights  # Pass edge_weights
    )
    print(f"Total spatial cells: {spatial_pred_labels.shape[0]}")
    print(f"Sensitive cells: {spatial_pred_labels.sum().item()}")

    visualize_and_evaluate(spatial_data, spatial_z, spatial_pred_probs, library_id='GSM6592061_M15')
    perform_deg(spatial_data, 'preprocessed/spatial/GSM6592061_M15_symbol_corrected.h5ad')
    analyze_cell_communication(spatial_data, spatial_pred_probs.cpu().numpy())


if __name__ == "__main__":
    main()