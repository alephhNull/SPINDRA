# trainer.py
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from evaluate import evaluate_model
from models import grad_reverse
from visualize import plot_all_embeddings

def compute_mmd(source, target, sigma=1.0):
    """Compute MMD with a Gaussian kernel."""
    source = source / source.norm(dim=1, keepdim=True)  # Normalize embeddings
    target = target / target.norm(dim=1, keepdim=True)
    xx = torch.exp(-torch.cdist(source, source)**2 / (2 * sigma**2)).mean()
    yy = torch.exp(-torch.cdist(target, target)**2 / (2 * sigma**2)).mean()
    xy = torch.exp(-torch.cdist(source, target)**2 / (2 * sigma**2)).mean()
    return xx + yy - 2 * xy

def train_model(spatial_encoder, bulk_encoder, sc_encoder, tumor_encoder, drug_predictor, discriminator, 
                domain_data, device='cpu', num_epochs=1000, pretrain_epochs=100):
    # Optimizers
    optimizer_G = torch.optim.Adam(
        list(spatial_encoder.parameters()) + list(bulk_encoder.parameters()) + 
        list(sc_encoder.parameters()) + list(tumor_encoder.parameters()) + list(drug_predictor.parameters()),
        lr=0.0001, weight_decay=1e-4
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, weight_decay=1e-4)

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

    print(f"Domain sizes: Spatial={N_spatial}, Bulk={N_bulk}, SC_Tumor={N_sc_tumor}, SC_CellLine={N_sc_cellline}")

    # Balancing weights for domain discriminator
    class_weights = torch.tensor([
        total_N / (4 * N_spatial), total_N / (4 * N_bulk),
        total_N / (4 * N_sc_tumor), total_N / (4 * N_sc_cellline)
    ]).to(device)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    bce_loss = nn.BCEWithLogitsLoss()

    # Pretraining phase with MMD
    print("Pretraining with MMD...")
    for epoch in range(pretrain_epochs):
        optimizer_G.zero_grad()

        # Feature extraction
        spatial_X, _ = domain_data['spatial_train']
        bulk_X, bulk_y = domain_data['bulk_train']
        sc_tumor_X, sc_tumor_y = domain_data['sc_tumor_train']
        sc_cellline_X, cell_line_y = domain_data['sc_cellline_train']
        edge_index = domain_data['edge_index_train']
        edge_weights = domain_data['edge_weights_train']

        spatial_z = spatial_encoder(spatial_X, edge_index, edge_weights)
        bulk_z = bulk_encoder(bulk_X)
        sc_tumor_z = tumor_encoder(sc_tumor_X)
        sc_cellline_z = sc_encoder(sc_cellline_X)

        # Prediction loss
        bulk_pred = drug_predictor(bulk_z).squeeze()
        cellline_pred = drug_predictor(sc_cellline_z).squeeze()
        tumor_pred = drug_predictor(sc_tumor_z).squeeze()
        loss_bulk = bce_loss(bulk_pred, bulk_y)
        loss_cellline = bce_loss(cellline_pred, cell_line_y)
        loss_tumor = bce_loss(tumor_pred, sc_tumor_y)
        loss_pred = loss_bulk + loss_cellline + loss_tumor

        # MMD loss for domain alignment
        lambda_mmd = 1.0  # Hyperparameter for MMD weight
        loss_align = (
            compute_mmd(spatial_z, bulk_z) +
            compute_mmd(spatial_z, sc_tumor_z) +
            compute_mmd(spatial_z, sc_cellline_z) +
            compute_mmd(bulk_z, sc_tumor_z) +
            compute_mmd(bulk_z, sc_cellline_z) +
            compute_mmd(sc_tumor_z, sc_cellline_z)
        )

        # Total pretraining loss
        total_pretrain_loss = loss_pred + lambda_mmd * loss_align
        total_pretrain_loss.backward()
        optimizer_G.step()

    # Main training with adversarial alignment
    progress_bar = tqdm(range(num_epochs), desc="Training")
    total_losses = []
    adv_losses = []
    pred_losses = []
    embeddings_history = []

    for epoch in progress_bar:
        lambda_adv = 0.1  # Hyperparameter for adversarial loss weight

        # Feature extraction
        spatial_z = spatial_encoder(spatial_X, edge_index, edge_weights)
        bulk_z = bulk_encoder(bulk_X)
        sc_tumor_z = tumor_encoder(sc_tumor_X)
        sc_cellline_z = sc_encoder(sc_cellline_X)

        # Train Domain Discriminator
        optimizer_D.zero_grad()
        features_detached = torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z]).detach()
        domain_preds = discriminator(features_detached)
        loss_D = ce_loss(domain_preds, domain_labels)
        loss_D.backward()
        optimizer_D.step()

        # Train Generators
        optimizer_G.zero_grad()
        features = torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z])
        features_reversed = grad_reverse(features)  # Gradient reversal
        domain_preds_adv = discriminator(features_reversed)
        loss_G_adv = ce_loss(domain_preds_adv, domain_labels)

        # Prediction loss
        bulk_pred = drug_predictor(bulk_z).squeeze()
        cellline_pred = drug_predictor(sc_cellline_z).squeeze()
        tumor_pred = drug_predictor(sc_tumor_z).squeeze()
        loss_bulk = bce_loss(bulk_pred, bulk_y)
        loss_cellline = bce_loss(cellline_pred, cell_line_y)
        loss_tumor = bce_loss(tumor_pred, sc_tumor_y)
        loss_pred = loss_bulk + loss_cellline + loss_tumor

        # Total loss for generators
        total_loss = loss_pred + lambda_adv * loss_G_adv
        total_loss.backward()
        optimizer_G.step()

        # Calculate accuracies
        with torch.no_grad():
            bulk_pred_labels = (torch.sigmoid(bulk_pred) > 0.5).float()
            bulk_accuracy = (bulk_pred_labels == bulk_y).float().mean().item()
            cellline_pred_labels = (torch.sigmoid(cellline_pred) > 0.5).float()
            cellline_accuracy = (cellline_pred_labels == cell_line_y).float().mean().item()
            tumor_pred_labels = (torch.sigmoid(tumor_pred) > 0.5).float()
            tumor_accuracy = (tumor_pred_labels == sc_tumor_y).float().mean().item()
            domain_accuracy = (torch.argmax(domain_preds, dim=1) == domain_labels).float().mean().item()

        # Log losses
        total_losses.append(total_loss.item())
        adv_losses.append(lambda_adv * loss_G_adv.item())
        pred_losses.append(loss_pred.item())

        epoch_list = [10, 50, 100, 500, 1500]
        if (epoch + 1) in epoch_list:
            all_z = torch.cat([spatial_z.cpu(), bulk_z.cpu(), sc_tumor_z.cpu(), sc_cellline_z.cpu()], dim=0).detach().numpy()
            all_y = np.concatenate([2 * np.ones(N_spatial), bulk_y.cpu(), sc_tumor_y.cpu(), cell_line_y.cpu()])
            embeddings_history.append((epoch + 1, all_z, all_labels, all_y))

        progress_bar.set_postfix({
            'Bulk Acc': f"{bulk_accuracy:.4f}",
            'CellLine Acc': f"{cellline_accuracy:.4f}",
            'Tumor Acc': f"{tumor_accuracy:.4f}",
            'Domain Acc': f"{domain_accuracy:.4f}",
            'Total Loss': f"{total_loss.item():.4f}",
        })

    # Evaluation and visualization
    evaluate_model(bulk_encoder=bulk_encoder, sc_encoder=sc_encoder, drug_predictor=drug_predictor, 
                   spatial_encoder=spatial_encoder, tumor_encoder=tumor_encoder, 
                   domain_discriminator=discriminator, domain_data=domain_data, device=device)
    # plot_all_embeddings(embeddings_history)

    # Plot losses
    plt.figure(figsize=(12, 8))
    plt.plot(range(num_epochs), total_losses, label='Total Loss (Pred + Adv)', color='blue')
    plt.plot(range(num_epochs), adv_losses, label='Adversarial Loss (λ * Adv)', color='yellow')
    plt.plot(range(num_epochs), pred_losses, label='Prediction Loss', color='green')
    plt.title('Generator Losses Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/loss_over_time.pdf', format='pdf')
    plt.show()

    return spatial_z

def predict_spatial(spatial_encoder, drug_predictor, spatial_X, edge_index, edge_weights=None):
    spatial_encoder.eval()
    drug_predictor.eval()
    with torch.no_grad():
        spatial_z = spatial_encoder(spatial_X, edge_index, edge_weights) if edge_index is not None else spatial_encoder(spatial_X)
        spatial_pred = drug_predictor(spatial_z).squeeze()
        spatial_pred_labels = (torch.sigmoid(spatial_pred) > 0.5).float()
    return spatial_z, spatial_pred_labels, torch.sigmoid(spatial_pred)