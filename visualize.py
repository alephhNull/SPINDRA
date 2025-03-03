# visualize.py
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from umap import UMAP
import seaborn as sns


def visualize_and_evaluate(spatial_data, spatial_z, spatial_pred_probs):
    """
    Parameters:
        spatial_data: an AnnData object containing spatial information.
        spatial_z: encoded features (e.g., from your spatial encoder).
        spatial_pred_probs: predicted probabilities for the positive class 
                            (values between 0 and 1), as a torch tensor.
    """
    # 1. Add probabilities to AnnData for spatial visualization.
    spatial_data.obs['predicted_response_prob'] = spatial_pred_probs.cpu().numpy()

    print("Generating spatial heatmap of predicted drug response probabilities...")
    sc.pl.spatial(
        spatial_data,
        color=['predicted_response_prob'],
        title='Predicted Drug Response Probability',
        cmap='coolwarm',
        library_id='1142243F',
        show=True
    )

    # 2. UMAP Visualization using the continuous probability values for color.
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
    plt.show()

    # 3. Cluster Evaluation: Convert probabilities to binary labels (threshold = 0.5)
    print("Evaluating clustering performance of predictions (after thresholding probabilities)...")
    spatial_pred_labels = (spatial_pred_probs > 0.5).long()

    # Only compute clustering metrics if we have at least two distinct clusters.
    if spatial_pred_labels.unique().numel() > 1:
        silhouette = silhouette_score(spatial_z.cpu().numpy(), spatial_pred_labels.cpu().numpy())
        davies_bouldin = davies_bouldin_score(spatial_z.cpu().numpy(), spatial_pred_labels.cpu().numpy())
        print(f"Silhouette Score: {silhouette:.4f} (closer to 1 is better)")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    else:
        print("Only one cluster detected; cannot compute clustering metrics.")


def plot_all_embeddings(embeddings_history):
    """
    Plot UMAP embeddings for all collected epochs in a single figure and compute silhouette scores.

    Args:
        embeddings_history (list): List of tuples (epoch, all_z, all_labels) for each collected epoch.
    """
    num_plots = len(embeddings_history)
    if num_plots == 0:
        print("No embeddings collected for plotting.")
        return

    # Set up subplot grid: 5 columns, calculate rows based on number of plots
    cols = 5
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()  # Flatten to easily index subplots

    # Domain names for legend
    domain_names = ['Spatial', 'Bulk', 'SC Tumor', 'SC Cell Line']

    for idx, (epoch, all_z, all_labels) in enumerate(embeddings_history):
        # Compute silhouette score; lower values indicate better domain alignment
        sil_score = silhouette_score(all_z, all_labels)
        print(f"Epoch {epoch}: Silhouette Score = {sil_score:.3f}")

        # Convert numerical labels to domain names for better legend
        all_labels_str = [domain_names[int(label)] for label in all_labels]

        # Apply UMAP to reduce embeddings to 2D
        umap_model = UMAP(n_components=2)
        embedding_2d = umap_model.fit_transform(all_z)

        # Create scatter plot
        sns.scatterplot(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            hue=all_labels_str,
            palette='Set1',
            s=10,
            alpha=0.5,
            ax=axes[idx],
            legend=(idx == 0)  # Show legend only on the first subplot
        )
        # Update title with silhouette score
        axes[idx].set_title(f'Epoch {epoch}, Silhouette: {sil_score:.3f}')
        axes[idx].set_xlabel('UMAP1')
        axes[idx].set_ylabel('UMAP2')

    # Remove unused subplots
    for ax in axes[num_plots:]:
        ax.remove()

    # Add a shared legend to the figure
    if num_plots > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Domain', loc='upper right')

    # Add a suptitle to explain the silhouette score
    fig.suptitle('UMAP Embeddings with Silhouette Scores\n(Lower silhouette score indicates better domain alignment)', fontsize=16)

    plt.tight_layout()
    plt.show()