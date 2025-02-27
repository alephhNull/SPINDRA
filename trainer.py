# trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from models import grad_reverse
import scanpy as sc

def train_model(spatial_encoder, bulk_encoder, sc_encoder, drug_predictor, discriminator, 
                domain_data, edge_index, device, num_epochs=1000, pretrain_epochs=100, edge_weights=None):
    # Optimizers with reduced learning rates
    optimizer_G = torch.optim.Adam(
        list(spatial_encoder.parameters()) + 
        list(bulk_encoder.parameters()) + 
        list(sc_encoder.parameters()) + 
        list(drug_predictor.parameters()),
        lr=0.0001,
        weight_decay=1e-4
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), 
        lr=0.0001,
        weight_decay=1e-4
    )

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

    # Pretraining phase
    print("Pretraining without adversarial loss...")
    for epoch in range(pretrain_epochs):
        optimizer_G.zero_grad()
        bulk_X, bulk_y = domain_data['bulk']
        sc_cellline_X, cell_line_y = domain_data['sc_cellline']
        bulk_z = bulk_encoder(bulk_X)
        sc_cellline_z = sc_encoder(sc_cellline_X)
        bulk_pred = drug_predictor(bulk_z).squeeze()
        cellline_pred = drug_predictor(sc_cellline_z).squeeze()
        loss_bulk = bce_loss(bulk_pred, bulk_y)
        loss_cellline = bce_loss(cellline_pred, cell_line_y)
        loss_pred = loss_bulk + loss_cellline
        loss_pred.backward()
        optimizer_G.step()

    # Main training
    progress_bar = tqdm(range(num_epochs), desc="Training")
    total_losses = []
    adv_losses = []
    pred_losses = []
    spatial_pred_history = []  # List to store spatial predictions

    for epoch in progress_bar:
        # Schedule the adversarial weight
        p = epoch / num_epochs
        lambda_total = 0.1
        lambda_adv = 1.0  # Fixed for grad_reverse, standard in DANN

        # Feature extraction
        spatial_X, _ = domain_data['spatial']
        bulk_X, bulk_y = domain_data['bulk']
        sc_tumor_X, _ = domain_data['sc_tumor']
        sc_cellline_X, cell_line_y = domain_data['sc_cellline']

        spatial_z = spatial_encoder(spatial_X, edge_index, edge_weights) if edge_index is not None else spatial_encoder(spatial_X)
        bulk_z = bulk_encoder(bulk_X)
        sc_tumor_z = sc_encoder(sc_tumor_X)
        sc_cellline_z = sc_encoder(sc_cellline_X)

        # Domain labels
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

        # Train Generators
        optimizer_G.zero_grad()
        features = torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z])
        features_reversed = grad_reverse(features, lambda_adv)
        domain_preds_adv = discriminator(features_reversed)
        loss_G_adv = ce_loss(domain_preds_adv, domain_labels)
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

        progress_bar.set_postfix({
            'Bulk Acc': f"{bulk_accuracy:.4f}",
            'CellLine Acc': f"{cellline_accuracy:.4f}",
            'Domain Acc': f"{domain_accuracy:.4f}",
            'Total Loss': f"{total_loss.item():.4f}",
        })

        # Collect spatial predictions for the last 100 epochs
    #     if epoch % 100 == 0:
    #         with torch.no_grad():
    #             # Set models to eval mode for consistent predictions
    #             spatial_encoder.eval()
    #             drug_predictor.eval()
    #             spatial_z = spatial_encoder(spatial_X, edge_index) if edge_index is not None else spatial_encoder(spatial_X)
    #             spatial_pred = drug_predictor(spatial_z).squeeze()
    #             spatial_probs = torch.sigmoid(spatial_pred).cpu().numpy()
    #             spatial_pred_history.append((epoch, spatial_probs))
    #             # Switch back to train mode
    #             spatial_encoder.train()
    #             drug_predictor.train()

    # Plot losses over time
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

    # num_epochs_to_plot = min(25, len(spatial_pred_history))  # Use all 100 if available
    # # Optionally reduce to 25 for a 5x5 grid if 100 is too much
    # # Uncomment the following lines to plot 25 instead:
    # # selected_indices = np.linspace(0, len(spatial_pred_history) - 1, 25).astype(int)
    # # spatial_pred_history = [spatial_pred_history[i] for i in selected_indices]
    # # num_epochs_to_plot = 25

    # num_cols = 5  # For a 10x10 grid
    # num_rows = (num_epochs_to_plot + num_cols - 1) // num_cols
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))  # Adjust figsize as needed
    # axes = axes.flatten()

    # print("Generating spatial heatmaps of predicted drug response probabilities...")
    # spatial_data = sc.read("preprocessed/spatial/visium_breast_cancer.h5ad")  # Load once

    # for idx, (epoch, probs) in enumerate(spatial_pred_history[:num_epochs_to_plot]):
    #     spatial_data.obs['predicted_response_prob'] = probs
    #     sc.pl.spatial(
    #         spatial_data,
    #         color=['predicted_response_prob'],
    #         title=f'Epoch {epoch}',
    #         cmap='coolwarm',
    #         library_id='1142243F',
    #         show=False,  # Don’t display individually
    #         ax=axes[idx]  # Plot on the specific subplot
    #     )
    #     axes[idx].set_title(f'Epoch {epoch}')

    # # Remove unused subplots
    # for ax in axes[num_epochs_to_plot:]:
    #     ax.remove()

    # plt.tight_layout()
    # plt.show()

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