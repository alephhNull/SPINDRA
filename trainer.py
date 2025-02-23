# trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random

from models import grad_reverse

def train_model(spatial_encoder, bulk_encoder, sc_encoder, drug_predictor, discriminator, 
                domain_data, edge_index, device, num_epochs=1000):
    # Optimizers
    optimizer_G = torch.optim.Adam(
        list(spatial_encoder.parameters()) +
        list(bulk_encoder.parameters()) +
        list(sc_encoder.parameters()) +
        list(drug_predictor.parameters()),
        lr=0.001
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # Domain weights for balanced loss
    N_spatial = domain_data['spatial'][0].size(0)
    N_bulk = domain_data['bulk'][0].size(0)
    N_sc_tumor = domain_data['sc_tumor'][0].size(0)
    N_sc_cellline = domain_data['sc_cellline'][0].size(0)
    total_N = N_spatial + N_bulk + N_sc_tumor + N_sc_cellline
    weights = [total_N / (4 * N_spatial), total_N / (4 * N_bulk), 
               total_N / (4 * N_sc_tumor), total_N / (4 * N_sc_cellline)]
    weights = torch.tensor(weights).to(device)

    ce_loss = nn.CrossEntropyLoss(weight=weights)
    bce_loss = nn.BCEWithLogitsLoss()

    progress_bar = tqdm(range(num_epochs), desc="Training")

    for epoch in progress_bar:
        # Schedule lambda_adv
        p = float(epoch) / num_epochs
        lambda_adv = 2. / (1. + np.exp(-10. * p)) - 1

        # Feature extraction
        spatial_X, _ = domain_data['spatial']
        bulk_X, bulk_y = domain_data['bulk']
        sc_tumor_X, _ = domain_data['sc_tumor']
        sc_cellline_X, cell_line_y = domain_data['sc_cellline']

        spatial_z = spatial_encoder(spatial_X, edge_index) if edge_index is not None else spatial_encoder(spatial_X)
        bulk_z = bulk_encoder(bulk_X)
        sc_tumor_z = sc_encoder(sc_tumor_X)
        sc_cellline_z = sc_encoder(sc_cellline_X)

        # Domain labels: 0: spatial, 1: bulk, 2: sc_tumor, 3: sc_cellline
        domain_labels = torch.cat([
            torch.zeros(N_spatial),
            torch.ones(N_bulk),
            2 * torch.ones(N_sc_tumor),
            3 * torch.ones(N_sc_cellline)
        ]).long().to(device)
        

        # Train Domain Discriminator
        optimizer_D.zero_grad()
        features_detached = torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z]).detach()
        domain_preds = discriminator(features_detached)
        loss_D = ce_loss(domain_preds, domain_labels)
        loss_D.backward()
        optimizer_D.step()

        # Train Generators with Adversarial Loss using GRL
        optimizer_G.zero_grad()
        features = torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z])
        features_reversed = grad_reverse(features, lambda_adv)
        domain_preds_adv = discriminator(features_reversed)
        loss_G_adv = ce_loss(domain_preds_adv, domain_labels)

        # Shared predictor
        bulk_pred = drug_predictor(bulk_z).squeeze()
        cellline_pred = drug_predictor(sc_cellline_z).squeeze()
        loss_bulk = bce_loss(bulk_pred, bulk_y)
        loss_cellline = bce_loss(cellline_pred, cell_line_y)
        loss_pred = loss_bulk + loss_cellline

        total_loss = loss_G_adv + loss_pred
        total_loss.backward()
        optimizer_G.step()

        # Calculate accuracies
        with torch.no_grad():
            bulk_pred_labels = (torch.sigmoid(bulk_pred) > 0.5).float()
            bulk_accuracy = (bulk_pred_labels == bulk_y).float().mean().item()
            cellline_pred_labels = (torch.sigmoid(cellline_pred) > 0.5).float()
            cellline_accuracy = (cellline_pred_labels == cell_line_y).float().mean().item()
            domain_accuracy = (torch.argmax(domain_preds, dim=1) == domain_labels).float().mean().item()

        progress_bar.set_postfix({
            'Bulk Acc': f"{bulk_accuracy:.4f}",
            'CellLine Acc': f"{cellline_accuracy:.4f}",
            'Domain Acc': f"{domain_accuracy:.4f}",
            'G Loss': f"{total_loss.item():.4f}",
            'D Loss': f"{loss_D.item():.4f}"
        })

    return spatial_z

def predict_spatial(spatial_encoder, drug_predictor, spatial_X, edge_index):
    with torch.no_grad():
        spatial_z = spatial_encoder(spatial_X, edge_index) if edge_index is not None else spatial_encoder(spatial_X)
        spatial_pred = drug_predictor(spatial_z).squeeze()
        spatial_pred_labels = (torch.sigmoid(spatial_pred) > 0.5).float()
    return spatial_z, spatial_pred_labels