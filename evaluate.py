import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score



def evaluate_model(bulk_encoder, sc_encoder, drug_predictor, spatial_encoder, tumor_encoder, domain_discriminator,
                    domain_data, device='cpu'):
    """
    Evaluate the model on the validation sets for bulk and single-cell cell line data,
    calculating accuracy, AUC, and F1 score.

    Args:
        bulk_encoder (nn.Module): Encoder for bulk data.
        sc_encoder (nn.Module): Encoder for single-cell data.
        drug_predictor (nn.Module): Predictor for drug response.
        domain_data (dict): Dictionary containing validation data with keys 'bulk_val' and 'sc_cellline_val',
                            where each value is a tuple of (features, labels).
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').

    Returns:
        dict: A dictionary containing accuracy, AUC, and F1 score for both validation sets.
    """
    # Set models to evaluation mode
    bulk_encoder.eval()
    sc_encoder.eval()
    spatial_encoder.eval()
    tumor_encoder.eval()
    domain_discriminator.eval()
    drug_predictor.eval()

    # Extract validation data
    bulk_val_X, bulk_val_y = domain_data['bulk_val']
    sc_cellline_val_X, sc_cellline_val_y = domain_data['sc_cellline_val']
    sc_tumor_val, _ = domain_data['sc_tumor_val']
    spatial_val, _ = domain_data['spatial_val']
    edge_index_val = domain_data['edge_index_val']
    edge_weights_val = domain_data['edge_weights_val']

    with torch.no_grad():
        spatial_z = spatial_encoder(spatial_val, edge_index_val, edge_weights_val)
        bulk_z = bulk_encoder(bulk_val_X)
        sc_tumor_z = sc_encoder(sc_tumor_val)
        sc_cellline_z = sc_encoder(sc_cellline_val_X)

    # Evaluate on bulk validation set
    with torch.no_grad():
        bulk_val_pred = drug_predictor(bulk_z).squeeze()
        bulk_val_probs = torch.sigmoid(bulk_val_pred).cpu().numpy()  # Probabilities for AUC
        bulk_val_pred_labels = (bulk_val_probs > 0.5).astype(int)    # Binary predictions
        bulk_val_y_np = bulk_val_y.cpu().numpy()                     # True labels as NumPy array

        # Calculate metrics for bulk
        bulk_accuracy = accuracy_score(bulk_val_y_np, bulk_val_pred_labels)
        bulk_auc = roc_auc_score(bulk_val_y_np, bulk_val_probs)
        bulk_f1 = f1_score(bulk_val_y_np, bulk_val_pred_labels)

    # Evaluate on single-cell cell line validation set
    with torch.no_grad():
        sc_cellline_val_pred = drug_predictor(sc_cellline_z).squeeze()
        sc_cellline_val_probs = torch.sigmoid(sc_cellline_val_pred).cpu().numpy()  # Probabilities for AUC
        sc_cellline_val_pred_labels = (sc_cellline_val_probs > 0.5).astype(int)    # Binary predictions
        sc_cellline_val_y_np = sc_cellline_val_y.cpu().numpy()                     # True labels as NumPy array

        # Calculate metrics for single-cell
        sc_cellline_accuracy = accuracy_score(sc_cellline_val_y_np, sc_cellline_val_pred_labels)
        sc_cellline_auc = roc_auc_score(sc_cellline_val_y_np, sc_cellline_val_probs)
        sc_cellline_f1 = f1_score(sc_cellline_val_y_np, sc_cellline_val_pred_labels)
    

    with torch.no_grad():

        # Domain labels
        domain_labels_val = torch.cat([
            torch.zeros(spatial_z.size(0)),
            torch.ones(bulk_z.size(0)),
            2 * torch.ones(sc_tumor_z.size(0)),
            3 * torch.ones(sc_cellline_z.size(0))
        ]).long().to(device)

        features_val = torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z])
        domain_preds_val = domain_discriminator(features_val).argmax(dim=1)

    domain_labels_val = domain_labels_val.cpu().numpy()
    domain_preds_val = domain_preds_val.cpu().numpy()

    domains = np.unique(domain_labels_val)

    # Compute accuracy for each domain
    per_domain_acc = []
    for domain in domains:
        # Mask for samples in this domain
        mask = domain_labels_val == domain
        # Accuracy for this domain
        domain_acc = accuracy_score(domain_labels_val[mask], domain_preds_val[mask])
        per_domain_acc.append(domain_acc)

    # Balanced accuracy is the average of per-domain accuracies
    balanced_acc = np.mean(per_domain_acc)
    print(f"Balanced Domain Discriminator Accuracy: {balanced_acc:.4f}")


    # Print the metrics
    print("Bulk Validation Metrics:")
    print(f"  Accuracy: {bulk_accuracy:.4f}")
    print(f"  AUC: {bulk_auc:.4f}")
    print(f"  F1 Score: {bulk_f1:.4f}")

    print("Single-Cell Cell Line Validation Metrics:")
    print(f"  Accuracy: {sc_cellline_accuracy:.4f}")
    print(f"  AUC: {sc_cellline_auc:.4f}")
    print(f"  F1 Score: {sc_cellline_f1:.4f}")

    # Return the metrics as a dictionary
    return {
        'bulk': {
            'accuracy': bulk_accuracy,
            'auc': bulk_auc,
            'f1': bulk_f1
        },
        'sc_cellline': {
            'accuracy': sc_cellline_accuracy,
            'auc': sc_cellline_auc,
            'f1': sc_cellline_f1
        },
        'balanced_accuracy': balanced_acc
    }