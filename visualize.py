# visualize.py
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from umap import UMAP

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
