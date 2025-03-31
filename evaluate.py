import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, silhouette_score

def compute_moran_I(values, edge_index, edge_weights):
    """
    Compute Moran's I for a set of values given a spatial weight matrix.
    
    Args:
        values (np.ndarray): Array of shape (N,) with the variable of interest (e.g., predicted probabilities).
        edge_index (torch.Tensor or np.ndarray): Tensor of shape (2, E) indicating edge connections.
        edge_weights (torch.Tensor or np.ndarray): Tensor of shape (E,) with weights for each edge.
        
    Returns:
        float: Moran's I statistic.
    """
    # Convert to numpy arrays if needed
    edge_index = edge_index.cpu().numpy()
    edge_weights = edge_weights.cpu().numpy()
    
    N = len(values)
    mean_val = np.mean(values)
    numerator = 0.0
    # Sum over all edges
    for k in range(len(edge_weights)):
        i = edge_index[0, k]
        j = edge_index[1, k]
        numerator += edge_weights[k] * (values[i] - mean_val) * (values[j] - mean_val)
    denominator = np.sum((values - mean_val) ** 2)
    S0 = np.sum(edge_weights)
    
    moran_I = (N / S0) * (numerator / denominator)
    return moran_I

def evaluate_model(bulk_encoder, sc_encoder, drug_predictor, spatial_encoder, tumor_encoder, domain_discriminator,
                   domain_data, device='cpu'):
    """
    Evaluate the model on the validation sets for bulk and single-cell cell line data,
    calculating accuracy, AUC, and F1 score, and also compute a spatial autocorrelation metric (Moran's I)
    for the spatial domain predictions.

    Args:
        bulk_encoder (nn.Module): Encoder for bulk data.
        sc_encoder (nn.Module): Encoder for single-cell data.
        drug_predictor (nn.Module): Predictor for drug response.
        domain_data (dict): Dictionary containing validation data with keys 'bulk_val', 'sc_cellline_val',
                            'sc_tumor_val', and 'spatial_val'. 'spatial_val' should be a tuple (features, _).
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').

    Returns:
        dict: A dictionary containing accuracy, AUC, and F1 score for each validation set, plus the balanced
              domain discriminator accuracy and Moran's I score for the spatial domain.
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
    sc_tumor_val_X, sc_tumor_val_y = domain_data['sc_tumor_val']
    spatial_val, _ = domain_data['spatial_val']
    edge_index_val = domain_data['edge_index_val']
    edge_weights_val = domain_data['edge_weights_val']

    with torch.no_grad():
        spatial_z = spatial_encoder(spatial_val, edge_index_val, edge_weights_val)
        bulk_z = bulk_encoder(bulk_val_X)
        sc_tumor_z = tumor_encoder(sc_tumor_val_X)
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

        # Calculate metrics for single-cell cell line
        sc_cellline_accuracy = accuracy_score(sc_cellline_val_y_np, sc_cellline_val_pred_labels)
        sc_cellline_auc = roc_auc_score(sc_cellline_val_y_np, sc_cellline_val_probs)
        sc_cellline_f1 = f1_score(sc_cellline_val_y_np, sc_cellline_val_pred_labels)

    # Evaluate on single-cell tumor validation set
    with torch.no_grad():
        sc_tumor_val_pred = drug_predictor(sc_tumor_z).squeeze()
        sc_tumor_val_probs = torch.sigmoid(sc_tumor_val_pred).cpu().numpy()  # Probabilities for AUC
        sc_tumor_val_pred_labels = (sc_tumor_val_probs > 0.5).astype(int)    # Binary predictions
        sc_tumor_val_y_np = sc_tumor_val_y.cpu().numpy()                     # True labels as NumPy array

        # Calculate metrics for single-cell tumor
        sc_tumor_accuracy = accuracy_score(sc_tumor_val_y_np, sc_tumor_val_pred_labels)
        sc_tumor_auc = roc_auc_score(sc_tumor_val_y_np, sc_tumor_val_probs)
        sc_tumor_f1 = f1_score(sc_tumor_val_y_np, sc_tumor_val_pred_labels)

    # Compute Moran's I for spatial domain predictions
    with torch.no_grad():
        spatial_val_pred = drug_predictor(spatial_z).squeeze()
        spatial_val_probs = torch.sigmoid(spatial_val_pred).cpu().numpy()
        moran_I = compute_moran_I(spatial_val_probs, edge_index_val, edge_weights_val)
        print(f"Moran's I for spatial domain predictions: {moran_I:.3f}")

    # Evaluate domain discriminator on all features
    with torch.no_grad():
        # Domain labels: 0 for spatial, 1 for bulk, 2 for sc tumor, 3 for sc cell line
        domain_labels_val = torch.cat([
            torch.zeros(spatial_z.size(0)),
            torch.ones(bulk_z.size(0)),
            2 * torch.ones(sc_tumor_z.size(0)),
            3 * torch.ones(sc_cellline_z.size(0))
        ]).long().to(device)

        features_val = torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z])
        domain_preds_val = domain_discriminator(features_val).argmax(dim=1)

    domain_labels_val_np = domain_labels_val.cpu().numpy()
    domain_preds_val_np = domain_preds_val.cpu().numpy()

    domains = np.unique(domain_labels_val_np)

    # Compute accuracy for each domain
    per_domain_acc = []
    for domain in domains:
        mask = domain_labels_val_np == domain
        domain_acc = accuracy_score(domain_labels_val_np[mask], domain_preds_val_np[mask])
        per_domain_acc.append(domain_acc)

    # Balanced accuracy is the average of per-domain accuracies
    balanced_acc = np.mean(per_domain_acc)
    print(f"Balanced Domain Discriminator Accuracy: {balanced_acc:.4f}")

    sil_score = silhouette_score(features_val.cpu().numpy(), domain_labels_val_np)
    print(f"Silhouette Score = {sil_score:.3f}")

    # Print the metrics
    print("Bulk Validation Metrics:")
    print(f"  Accuracy: {bulk_accuracy:.4f}")
    print(f"  AUC: {bulk_auc:.4f}")
    print(f"  F1 Score: {bulk_f1:.4f}")

    print("Single-Cell Cell Line Validation Metrics:")
    print(f"  Accuracy: {sc_cellline_accuracy:.4f}")
    print(f"  AUC: {sc_cellline_auc:.4f}")
    print(f"  F1 Score: {sc_cellline_f1:.4f}")

    print("Single-Cell Tumor Validation Metrics:")
    print(f"  Accuracy: {sc_tumor_accuracy:.4f}")
    print(f"  AUC: {sc_tumor_auc:.4f}")
    print(f"  F1 Score: {sc_tumor_f1:.4f}")

    # Return the metrics as a dictionary, including Moran's I for the spatial domain
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
        'sc_tumor': {
            'accuracy': sc_tumor_accuracy,
            'auc': sc_tumor_auc,
            'f1': sc_tumor_f1
        },
        'balanced_accuracy': balanced_acc,
        'silhouette_score': sil_score,
        'moran_I': moran_I
    }
