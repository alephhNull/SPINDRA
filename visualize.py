# visualize.py
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from umap import UMAP

def visualize_and_evaluate(spatial_data, spatial_z, spatial_pred_labels):
    # Add predictions to AnnData
    spatial_data.obs['predicted_response'] = spatial_pred_labels.cpu().numpy()
    spatial_data.obs['predicted_response'] = spatial_data.obs['predicted_response'].astype('category')
    spatial_data.obs['predicted_response'] = spatial_data.obs['predicted_response'].map({0: 'Resistant', 1: 'Sensitive'})

    # 1. Spatial Plot
    print("Generating spatial plot of predicted drug response...")
    sc.pl.spatial(
        spatial_data,
        color=['predicted_response'],
        title='Predicted Drug Response (Sensitive vs Resistant)',
        cmap='coolwarm',
        library_id='1142243F',
        show=True
    )

    # 2. UMAP Visualization
    print("Computing UMAP embedding of encoded features...")
    umap = UMAP(n_components=2, random_state=42)
    spatial_z_umap = umap.fit_transform(spatial_z.cpu().numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        spatial_z_umap[:, 0],
        spatial_z_umap[:, 1],
        c=spatial_pred_labels.cpu().numpy(),
        cmap='coolwarm',
        s=10,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Predicted Response (0: Resistant, 1: Sensitive)')
    plt.title('UMAP of Spatial Encoded Features (Colored by Predictions)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.show()

    # 3. Cluster Evaluation
    print("Evaluating clustering performance of predictions...")
    if spatial_pred_labels.unique().numel() > 1:  # Ensure there are at least two clusters
        silhouette = silhouette_score(spatial_z.cpu().numpy(), spatial_pred_labels.cpu().numpy())
        davies_bouldin = davies_bouldin_score(spatial_z.cpu().numpy(), spatial_pred_labels.cpu().numpy())
        print(f"Silhouette Score: {silhouette:.4f} (closer to 1 is better)")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    else:
        print("Only one cluster detected; cannot compute clustering metrics.")