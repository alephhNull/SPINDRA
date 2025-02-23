# main.py
import torch
from data_loader import load_data, prepare_tensors, spatial_to_graph
from models import SpatialEncoder, BulkEncoder, SingleCellEncoder, DrugResponsePredictor, DomainDiscriminator, grad_reverse
from trainer import train_model, predict_spatial
from visualize import visualize_and_evaluate

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    spatial_data, bulk_data, sc_tumor_data, sc_cellline_data, common_genes = load_data()
    domain_data = prepare_tensors(spatial_data, bulk_data, sc_tumor_data, sc_cellline_data, device)
    edge_index = spatial_to_graph(spatial_data, k=5, device=device)

    # Initialize models
    spatial_encoder = SpatialEncoder(input_dim=len(common_genes), edge_index=None).to(device)
    bulk_encoder = BulkEncoder(input_dim=len(common_genes)).to(device)
    sc_encoder = SingleCellEncoder(input_dim=len(common_genes)).to(device)
    drug_predictor = DrugResponsePredictor(hidden_dim=128).to(device)
    discriminator = DomainDiscriminator(hidden_dim=128).to(device)

    # Train the model
    spatial_z = train_model(
        spatial_encoder, bulk_encoder, sc_encoder, drug_predictor, discriminator,
        domain_data, edge_index, device, num_epochs=500
    )

    # Predict on spatial data (redundant here since train_model returns it, but kept for modularity)
    spatial_X, _ = domain_data['spatial']
    spatial_z, spatial_pred_labels = predict_spatial(spatial_encoder, drug_predictor, spatial_X, edge_index)
    print(f"Total spatial cells: {spatial_pred_labels.shape[0]}")
    print(f"Sensitive cells: {spatial_pred_labels.sum().item()}")

    # Visualize and evaluate
    visualize_and_evaluate(spatial_data, spatial_z, spatial_pred_labels)

if __name__ == "__main__":
    main()