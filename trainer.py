# trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from evaluate import evaluate_model
from models import grad_reverse
import scanpy as sc
from umap import UMAP
import seaborn as sns

from visualize import plot_all_embeddings


def train_model(spatial_encoder, bulk_encoder, sc_encoder, tumor_encoder, drug_predictor, discriminator, 
                domain_data, device='cpu', num_epochs=1000, pretrain_epochs=100):
    # Optimizers
    optimizer_G = torch.optim.Adam(
        list(spatial_encoder.parameters()) + list(bulk_encoder.parameters()) + 
        list(sc_encoder.parameters()) + list(tumor_encoder.parameters()) + list(drug_predictor.parameters()),
        lr=0.0001, weight_decay=1e-4
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, weight_decay=1e-4)

    # Domain sizes
    N_spatial = domain_data['spatial_train'][0].size(0)
    N_bulk = domain_data['bulk_train'][0].size(0)
    N_sc_tumor = domain_data['sc_tumor_train'][0].size(0)
    N_sc_cellline = domain_data['sc_cellline_train'][0].size(0)
    total_N = N_spatial + N_bulk + N_sc_tumor + N_sc_cellline

    all_labels = np.concatenate([
                np.zeros(N_spatial),         # Spatial: 0
                np.ones(N_bulk),            # Bulk: 1
                2 * np.ones(N_sc_tumor),    # SC Tumor: 2
                3 * np.ones(N_sc_cellline)  # SC Cell Line: 3
            ])
    
    domain_labels = torch.from_numpy(all_labels).long().to(device)

    print(N_spatial, N_bulk, N_sc_tumor, N_sc_cellline)

    # Balancing weights (inverse of domain sizes)
    class_weights = torch.tensor([
        total_N / (4 * N_spatial), total_N / (4 * N_bulk),
        total_N / (4 * N_sc_tumor), total_N / (4 * N_sc_cellline)
    ]).to(device)

    # Combined class weights
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    bce_loss = nn.BCEWithLogitsLoss()

    # Pretraining phase (unchanged)
    print("Pretraining without adversarial loss...")
    for epoch in range(pretrain_epochs):
        optimizer_G.zero_grad()
        bulk_X, bulk_y = domain_data['bulk_train']
        sc_cellline_X, cell_line_y = domain_data['sc_cellline_train']
        bulk_z = bulk_encoder(bulk_X)
        sc_cellline_z = sc_encoder(sc_cellline_X)
        bulk_pred = drug_predictor(bulk_z).squeeze()
        cellline_pred = drug_predictor(sc_cellline_z).squeeze()
        loss_bulk = bce_loss(bulk_pred, bulk_y)
        loss_cellline = bce_loss(cellline_pred, cell_line_y)
        loss_pred = loss_bulk + loss_cellline
        loss_pred.backward()
        optimizer_G.step()

    # Main training (mostly unchanged, uses updated ce_loss)
    progress_bar = tqdm(range(num_epochs), desc="Training")
    total_losses = []
    adv_losses = []
    pred_losses = []
    embeddings_history = []

    for epoch in progress_bar:
        p = epoch / num_epochs
        lambda_total = 0.5
        lambda_adv = 1.0  # For grad_reverse

        # Feature extraction
        spatial_X, _ = domain_data['spatial_train']
        bulk_X, bulk_y = domain_data['bulk_train']
        sc_tumor_X, _ = domain_data['sc_tumor_train']
        sc_cellline_X, cell_line_y = domain_data['sc_cellline_train']
        edge_index = domain_data['edge_index_train']
        edge_weights = domain_data['edge_weights_train']

        spatial_z = spatial_encoder(spatial_X, edge_index, edge_weights)
        bulk_z = bulk_encoder(bulk_X)
        sc_tumor_z = tumor_encoder(sc_tumor_X)
        sc_cellline_z = sc_encoder(sc_cellline_X)

        # Train Domain Discriminator
        optimizer_D.zero_grad()
        features_detached = torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z]).detach()
        domain_preds = discriminator(features_detached)
        loss_D = ce_loss(domain_preds, domain_labels)  # Uses combined weights
        loss_D.backward()
        optimizer_D.step()

        # Train Generators
        optimizer_G.zero_grad()
        features = torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z])
        features_reversed = grad_reverse(features, lambda_adv)
        domain_preds_adv = discriminator(features_reversed)
        loss_G_adv = ce_loss(domain_preds_adv, domain_labels)  # Uses combined weights
        bulk_pred = drug_predictor(bulk_z).squeeze()
        cellline_pred = drug_predictor(sc_cellline_z).squeeze()
        loss_bulk = bce_loss(bulk_pred, bulk_y)
        loss_cellline = bce_loss(cellline_pred, cell_line_y)
        loss_pred = loss_bulk + loss_cellline
        total_loss = loss_pred + lambda_total * loss_G_adv
        total_loss.backward()
        optimizer_G.step()

        # Calculate accuracies
        with torch.no_grad():
            bulk_pred_labels = (torch.sigmoid(bulk_pred) > 0.5).float()
            bulk_accuracy = (bulk_pred_labels == bulk_y).float().mean().item()
            cellline_pred_labels = (torch.sigmoid(cellline_pred) > 0.5).float()
            cellline_accuracy = (cellline_pred_labels == cell_line_y).float().mean().item()
            domain_accuracy = (torch.argmax(domain_preds, dim=1) == domain_labels).float().mean().item()

        # Log losses
        total_losses.append(total_loss.item())
        adv_losses.append(lambda_total * loss_G_adv.item())
        pred_losses.append(loss_pred.item())

        if (epoch + 1) % 1000 == 0:
            # Concatenate all embeddings
            all_z = torch.cat([
                spatial_z.cpu(),
                bulk_z.cpu(),
                sc_tumor_z.cpu(),
                sc_cellline_z.cpu()
            ], dim=0).detach().numpy()

            # Store the embeddings, labels, and epoch
            embeddings_history.append((epoch + 1, all_z, all_labels))

        progress_bar.set_postfix({
            'Bulk Acc': f"{bulk_accuracy:.4f}",
            'CellLine Acc': f"{cellline_accuracy:.4f}",
            'Domain Acc': f"{domain_accuracy:.4f}",
            'Total Loss': f"{total_loss.item():.4f}",
        })


    evaluate_model(bulk_encoder=bulk_encoder, sc_encoder=sc_encoder, drug_predictor=drug_predictor, spatial_encoder=spatial_encoder,
                    tumor_encoder=tumor_encoder, domain_discriminator=discriminator, domain_data=domain_data, device=device)
    
    plot_all_embeddings(embeddings_history)

    # Plot losses
    plt.figure(figsize=(12, 8))
    plt.plot(range(num_epochs), total_losses, label='Total Loss (Pred + λ * Adv)', color='blue')
    plt.plot(range(num_epochs), adv_losses, label='Adversarial Loss (λ * Adv)', color='orange')
    plt.plot(range(num_epochs), pred_losses, label='Prediction Loss', color='green')
    plt.title('Generator Losses Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return spatial_z

# predict_spatial function remains unchanged
def predict_spatial(spatial_encoder, drug_predictor, spatial_X, edge_index, edge_weights=None):
    spatial_encoder.eval()
    drug_predictor.eval()
    with torch.no_grad():
        spatial_z = spatial_encoder(spatial_X, edge_index, edge_weights) if edge_index is not None else spatial_encoder(spatial_X)
        spatial_pred = drug_predictor(spatial_z).squeeze()
        spatial_pred_labels = (torch.sigmoid(spatial_pred) > 0.5).float()
    return spatial_z, spatial_pred_labels, torch.sigmoid(spatial_pred)