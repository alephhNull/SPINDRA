import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, silhouette_score , precision_score
import matplotlib.pyplot as plt
import umap
import pandas as pd
import os 
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, recall_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def plot_validation_summary(
    y_true, y_pred, y_probs, X_umap, save_dir, data_name="bulk", label_map=None):
    # --------- ساخت لیبل‌ها ---------
    if label_map is None:
        label_map = {0: 'Resistant', 1: 'Sensitive'}
    color_dict = {'Sensitive': 'red', 'Resistant': 'blue'}

    val_label_gt = pd.Series(y_true).map(label_map)
    val_label_pred = pd.Series(y_pred).map(label_map)

    df_umap_val = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
    df_umap_val['GroundTruth'] = val_label_gt.values
    df_umap_val['Predicted'] = val_label_pred.values

    # --------- محاسبه متریک‌ها ---------
    accuracy = accuracy_score(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        specificity = np.nan

    # --------- ذخیره متریک‌ها در فایل متنی ---------
    metrics_text = (
        f"Accuracy:           {accuracy:.4f}\n"
        f"Balanced Accuracy:  {balanced_acc:.4f}\n"
        f"AUC:                {auc_val:.4f}\n"
        f"Precision:          {precision:.4f}\n"
        f"Recall:             {recall:.4f}\n"
        f"Specificity:        {specificity:.4f}\n"
        f"F1-score:           {f1:.4f}\n"
        f"MCC:                {mcc:.4f}\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{data_name}_val_metrics.txt"), "w") as f:
        f.write(metrics_text)

    # --------- جدول متریک‌ها برای جدول و بارپلات ---------
    metrics_dict = {
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_acc,
        'AUC': auc_val,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1-score': f1,
        'MCC': mcc,
    }
    table_data = [[k, f"{v:.4f}"] for k, v in metrics_dict.items()]
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    # --------- پلات UMAP ساده ---------
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for label in color_dict:
        sub = df_umap_val[df_umap_val['GroundTruth'] == label]
        axs[0].scatter(sub['UMAP1'], sub['UMAP2'], c=color_dict[label], label=label, s=60, alpha=0.7, edgecolor='k')
    axs[0].set_title(f'{data_name.capitalize()} UMAP by Ground Truth')
    axs[0].set_xlabel('UMAP1')
    axs[0].set_ylabel('UMAP2')
    axs[0].legend(title='True Label')
    for label in color_dict:
        sub = df_umap_val[df_umap_val['Predicted'] == label]
        axs[1].scatter(sub['UMAP1'], sub['UMAP2'], c=color_dict[label], label=label, s=60, alpha=0.7, edgecolor='k')
    axs[1].set_title(f'{data_name.capitalize()} UMAP by Predicted')
    axs[1].set_xlabel('UMAP1')
    axs[1].set_ylabel('UMAP2')
    axs[1].legend(title='Predicted Label')
    plt.suptitle(f'UMAP of {data_name.capitalize()} Validation Set: Ground Truth vs. Predicted', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, f"umap_{data_name}_val_side_by_side_gt_vs_pred.png"), dpi=300)
    plt.close()

    # --------- پلات جدول متریک‌ها ---------
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 0.8]})
    # UMAP پلات‌ها مثل بالا ...
    for label in color_dict:
        sub = df_umap_val[df_umap_val['GroundTruth'] == label]
        axs[0].scatter(sub['UMAP1'], sub['UMAP2'], c=color_dict[label], label=label, s=60, alpha=0.7, edgecolor='k')
    axs[0].set_title(f'{data_name.capitalize()} UMAP by Ground Truth')
    axs[0].set_xlabel('UMAP1')
    axs[0].set_ylabel('UMAP2')
    axs[0].legend(title='True Label')
    for label in color_dict:
        sub = df_umap_val[df_umap_val['Predicted'] == label]
        axs[1].scatter(sub['UMAP1'], sub['UMAP2'], c=color_dict[label], label=label, s=60, alpha=0.7, edgecolor='k')
    axs[1].set_title(f'{data_name.capitalize()} UMAP by Predicted')
    axs[1].set_xlabel('UMAP1')
    axs[1].set_ylabel('UMAP2')
    axs[1].legend(title='Predicted Label')
    axs[2].axis('off')
    table = axs[2].table(cellText=table_data, colLabels=["Metric", "Value"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2)
    axs[2].set_title('Validation Metrics', fontsize=16, pad=25)
    plt.suptitle(f'UMAP of {data_name.capitalize()} Validation Set: Ground Truth vs. Predicted', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, f"umap_{data_name}_val_side_by_side_gt_vs_pred_with_metrics_table.png"), dpi=300)
    plt.close()

    # --------- بارپلات متریک‌ها ---------
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, values, color="steelblue")
    ax.set_xlim(0, 1.10)
    ax.set_title("Validation Metrics")
    ax.set_xlabel("Value")
    for i, (v, n) in enumerate(zip(values, names)):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=13, color='black')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{data_name}_validation_metrics_barplot.png"), dpi=300)
    plt.close()

    # --------- ROC Curve ---------
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{data_name.capitalize()} Validation ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"roc_curve_{data_name}_val.png"), dpi=300)
    plt.close()

    # --------- Confusion Matrix ---------
    labels = [label_map[0], label_map[1]]

    plt.figure(figsize=(4,4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[f'Pred {lbl}' for lbl in labels],
        yticklabels=[f'True {lbl}' for lbl in labels]
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix ({data_name.capitalize()} Validation)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{data_name}_val.png"), dpi=300)
    plt.close()




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

        X_val = bulk_val_X.cpu().numpy()

        # اجرای UMAP روی validation
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap_val = reducer.fit_transform(X_val)

        plot_validation_summary(
        y_true=bulk_val_y_np,
        y_pred=bulk_val_pred_labels,
        y_probs=bulk_val_probs,
        X_umap=X_umap_val,
        save_dir="figures_prediction/bulk",
        data_name="bulk"
            )



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

        reducer = umap.UMAP(n_components=2, random_state=42)
        X_val_sc = sc_cellline_val_X.cpu().numpy()
        X_umap_val_sc = reducer.fit_transform(X_val_sc)

    plot_validation_summary(
        y_true=sc_cellline_val_y_np,
        y_pred=sc_cellline_val_pred_labels,
        y_probs=sc_cellline_val_probs,
        X_umap=X_umap_val_sc,
        save_dir="figures_prediction/sc_cellline",
        data_name="sc_cellline"
        )

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

        reducer = umap.UMAP(n_components=2, random_state=42)
        X_val_tumor = sc_tumor_val_X.cpu().numpy()
        X_umap_val_tumor = reducer.fit_transform(X_val_tumor)


        plot_validation_summary(
        y_true=sc_tumor_val_y_np,
        y_pred=sc_tumor_val_pred_labels,
        y_probs=sc_tumor_val_probs,
        X_umap=X_umap_val_tumor,
        save_dir="figures_prediction/sc_tumor",
        data_name="sc_tumor"
            )



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
