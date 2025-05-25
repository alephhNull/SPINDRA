# main.py
import random
import numpy as np
import torch
import os
import argparse
from data_loader import load_data, prepare_tensors, spatial_to_graph
from models import SpatialEncoder, BulkEncoder, SingleCellEncoder, DrugResponsePredictor, DomainDiscriminator, ImprovedSpatialEncoder, TumorEncoder, grad_reverse
from perform_deg import perform_deg
from trainer import train_model, predict_spatial
from visualize import visualize_and_evaluate
from cell_communication import analyze_cell_communication
from preprocess_bulk import preprocess_bulk
from preprocess_visium_spatial import preprocess_visium_spatial
from preprocess_cellline import preprocess_celline
from preprocess_tumor import preprocess_tumor


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load and preprocess data
    spatial_data, bulk_data, sc_tumor_data, sc_cellline_data, common_genes = load_data(
        args.spatial, args.sc_tumor, args.sc_cellline
    )
    domain_data = prepare_tensors(spatial_data, bulk_data, sc_tumor_data, sc_cellline_data, device, k=args.k)

    spatial_encoder = SpatialEncoder(input_dim=len(common_genes), use_edge=True).to(device)
    # spatial_encoder = ImprovedSpatialEncoder(input_dim=len(common_genes), num_layers=2, use_gat=True).to(device)
    bulk_encoder = BulkEncoder(input_dim=len(common_genes)).to(device)
    sc_encoder = SingleCellEncoder(input_dim=len(common_genes)).to(device)
    tumor_encoder = TumorEncoder(input_dim=len(common_genes)).to(device)
    drug_predictor = DrugResponsePredictor(hidden_dim=128).to(device)
    discriminator = DomainDiscriminator(hidden_dim=128).to(device)

    spatial_z = train_model(
        spatial_encoder, bulk_encoder, sc_encoder, tumor_encoder, drug_predictor, discriminator,
        domain_data, device, num_epochs=args.num_epochs, pretrain_epochs=args.pretrain_epochs
    )

    edge_index, edge_weights = spatial_to_graph(spatial_data, k=args.k, device=device)
    spatial_X = torch.tensor(spatial_data.X).float().to(device)
    spatial_z, spatial_pred_labels, spatial_pred_probs = predict_spatial(
        spatial_encoder, drug_predictor, spatial_X, edge_index, edge_weights
    )
    print(f"Total spatial cells: {spatial_pred_labels.shape[0]}")
    print(f"Sensitive cells: {spatial_pred_labels.sum().item()}")

    visualize_and_evaluate(spatial_data, spatial_z, spatial_pred_probs, library_id=args.library_id)
    filename_without_extension = args.spatial.split('/')[-1].split('.')[0]
    perform_deg(spatial_data, f'preprocessed/spatial/{filename_without_extension}_symbol_corrected.h5ad')
    analyze_cell_communication(spatial_data, spatial_pred_probs.cpu().numpy())


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Drug Response Prediction from Spatial Transcriptomics Data')
    
    # Add arguments with default values
    parser.add_argument('--bulk_exp', type=str, default='data/bulk/ALL_expression.csv', 
                       help='Bulk expression data file path')
    parser.add_argument('--bulk_label', type=str, default='data/bulk/ALL_label_binary_wf.csv', 
                       help='Bulk label data file path')
    parser.add_argument('--sc_tumor', type=str, default='data/sc-tumor/gse169246.h5ad', 
                       help='Single-cell tumor data file path')
    parser.add_argument('--sc_cellline', type=str, default='data/sc-cell-line/GSE131984.h5ad', 
                       help='Single-cell cell line data file path')
    parser.add_argument('--spatial', type=str, default='data/spatial/visium-1142243F.h5ad', 
                       help='Visium spatial data file path')
    parser.add_argument('--library_id', type=str, default='1142243F', 
                       help='Library ID for the spatial data')
    parser.add_argument('--num_epochs', type=int, default=2, 
                       help='Number of training epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=2, 
                       help='Number of pretraining epochs')
    parser.add_argument('--k', type=int, default=10, 
                       help='Number of nearest neighbors for graph construction')
    
    args = parser.parse_args()

    # Create directories
    os.makedirs('preprocessed/bulk', exist_ok=True)
    os.makedirs('preprocessed/sc-cell-line', exist_ok=True)
    os.makedirs('preprocessed/sc-tumor', exist_ok=True)
    os.makedirs('preprocessed/spatial', exist_ok=True)
    os.makedirs('output', exist_ok=True)


    # Preprocess data
    # preprocess_bulk(args.bulk_exp, args.bulk_label)
    # preprocess_visium_spatial(args.spatial, args.library_id)
    # preprocess_tumor(args.sc_tumor)
    # preprocess_celline(args.sc_cellline)
    
    # Run main function with args object
    main(args)