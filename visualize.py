# visualize.py
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from umap import UMAP
import seaborn as sns
import pandas as pd


def visualize_and_evaluate(spatial_data, spatial_z, spatial_pred_probs, filter_malignant=False, library_id=None):
    """
    Parameters:
        spatial_data: an AnnData object containing spatial information.
        spatial_z: encoded features (e.g., from your spatial encoder).
        spatial_pred_probs: predicted probabilities for the positive class 
                            (values between 0 and 1), as a torch tensor.
    """

    if filter_malignant:
        # 1. Filter to malignant cells only
        malignant_mask = spatial_data.obs['cell_type'] == 'malignant cell'

        # Check if there are any malignant cells; if not, exit early
        if malignant_mask.sum() == 0:
            print("No malignant cells found in the dataset.")
            return

        # Subset the AnnData object and predicted probabilities
        spatial_data = spatial_data[malignant_mask, :]
        spatial_pred_probs = spatial_pred_probs[malignant_mask]
        spatial_z = spatial_z[malignant_mask]

    # Add probabilities to the subset for spatial visualization
    spatial_data.obs['predicted_response_prob'] = (spatial_pred_probs.cpu().numpy() > 0.5).astype(int)

    # Generate the spatial heatmap for malignant cells only
    print("Generating spatial heatmap of predicted drug response probabilities for malignant cells...")
    sc.pl.spatial(
        spatial_data,
        color=['predicted_response_prob'],
        title='Predicted Drug Response Probability (Malignant Cells Only)',
        cmap='coolwarm',
        library_id=library_id,
        show=True
    )
    plt.savefig('output/prediction_heatmap.pdf', format='pdf')

    # UMAP Visualization
    print("Computing UMAP embedding of encoded features...")
    umap_model = UMAP(n_components=2, random_state=42)
    spatial_z_umap = umap_model.fit_transform(spatial_z.cpu().numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        spatial_z_umap[:, 0],
        spatial_z_umap[:, 1],
        c=spatial_pred_probs.cpu().numpy(),  # Color by probability
        cmap='coolwarm',
        s=10,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Predicted Response Probability\n(0: Resistant, 1: Sensitive)')
    plt.title('UMAP of Spatial Encoded Features (Colored by Response Probability)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.savefig('output/umap_embedding.pdf', format='pdf')
    plt.show()

    # 3. Cluster Evaluation: Convert probabilities to binary labels (threshold = 0.5) (unchanged)
    print("Evaluating clustering performance of predictions (after thresholding probabilities)...")
    spatial_pred_labels = (spatial_pred_probs > 0.5).long()

    # Only compute clustering metrics if we have at least two distinct clusters
    if spatial_pred_labels.unique().numel() > 1:
        silhouette = silhouette_score(spatial_z.cpu().numpy(), spatial_pred_labels.cpu().numpy())
        davies_bouldin = davies_bouldin_score(spatial_z.cpu().numpy(), spatial_pred_labels.cpu().numpy())
        print(f"Silhouette Score: {silhouette:.4f} (closer to 1 is better)")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    else:
        print("Only one cluster detected; cannot compute clustering metrics.")


def plot_all_embeddings(embeddings_history):
    """
    Plot UMAP embeddings for all collected epochs in two separate figures:
    one using domain labels and one using response labels.
    Computes UMAP only once per epoch.

    Args:
        embeddings_history (list): List of tuples (epoch, all_z, all_labels, all_y) for each collected epoch.
    """
    num_epochs = len(embeddings_history)
    if num_epochs == 0:
        print("No embeddings collected for plotting.")
        return

    # Set up subplot grids for the two figures.
    cols = 5
    rows = (num_epochs + cols - 1) // cols

    # Figure for Domain labels with silhouette scores.
    fig_domain, axes_domain = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes_domain = axes_domain.flatten()

    # Figure for Response labels.
    fig_response, axes_response = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes_response = axes_response.flatten()

    # Mapping from numerical labels to names.
    domain_names = ['Spatial', 'Bulk', 'SC Tumor', 'SC Cell Line']
    response_names = ['resistant', 'sensitive', 'unknown']

    for idx, (epoch, all_z, all_labels, all_y) in enumerate(embeddings_history):
        # Compute silhouette score for the domains.
        sil_score = silhouette_score(all_z, all_labels)
        print(f"Epoch {epoch}: Silhouette Score = {sil_score:.3f}")

        # Convert numerical labels to string labels.
        all_labels_str = [domain_names[int(label)] for label in all_labels]
        all_responses_str = [response_names[int(response)] for response in all_y]

        # Compute UMAP embeddings (2D) only once.
        umap_model = UMAP(n_components=2)
        embedding_2d = umap_model.fit_transform(all_z)

        # Plot the domain-based scatter plot.
        sns.scatterplot(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            hue=all_labels_str,
            palette='Set1',
            s=10,
            alpha=0.5,
            ax=axes_domain[idx],
            legend=(idx == 0)  # show legend only on the first subplot
        )
        axes_domain[idx].set_title(f'Epoch {epoch}, Silhouette: {sil_score:.3f}')
        axes_domain[idx].set_xlabel('UMAP1')
        axes_domain[idx].set_ylabel('UMAP2')

        # Plot the response-based scatter plot.
        sns.scatterplot(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            hue=all_responses_str,
            palette='Set2',
            s=10,
            alpha=0.5,
            ax=axes_response[idx],
            legend=(idx == 0)  # show legend only on the first subplot
        )
        axes_response[idx].set_title(f'Epoch {epoch}')
        axes_response[idx].set_xlabel('UMAP1')
        axes_response[idx].set_ylabel('UMAP2')

    # Remove any unused subplots.
    for ax in axes_domain[num_epochs:]:
        ax.remove()
    for ax in axes_response[num_epochs:]:
        ax.remove()

    # Add shared legends to each figure.
    if num_epochs > 0:
        handles_domain, labels_domain = axes_domain[0].get_legend_handles_labels()
        fig_domain.legend(handles_domain, labels_domain, title='Domain', loc='upper right')

        handles_response, labels_response = axes_response[0].get_legend_handles_labels()
        fig_response.legend(handles_response, labels_response, title='Response', loc='upper right')

    # Add a suptitle for additional context.
    fig_domain.suptitle('UMAP Embeddings with Silhouette Scores (Domain Labels)\n(Lower silhouette score indicates better domain alignment)', fontsize=16)
    fig_response.suptitle('UMAP Embeddings (Response Labels)', fontsize=16)

    plt.tight_layout()
    plt.show()
