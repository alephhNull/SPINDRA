# data_loader.py
import scanpy as sc
import torch
import pandas as pd

def load_data(folder='preprocessed'):
    # Load datasets
    spatial_data = sc.read(f"{folder}/spatial/visium_breast_cancer.h5ad")
    bulk_data = pd.read_csv(f"{folder}/bulk/bulk_data.csv")
    sc_tumor_data = sc.read(f"{folder}/sc-tumor/GSE169246.h5ad")
    sc_cellline_data = sc.read(f"{folder}/sc-cell-line/GSE117872_HN120.h5ad")

    # Extract common genes
    common_genes = list(
        set(spatial_data.var_names) &
        set(bulk_data.columns) &
        set(sc_tumor_data.var_names) &
        set(sc_cellline_data.var_names)
    )

    # Subset data to common genes
    spatial_data = spatial_data[:, common_genes]
    bulk_data = bulk_data.loc[:, common_genes + ['PACLITAXEL']]
    sc_tumor_data = sc_tumor_data[:, common_genes]
    sc_cellline_data = sc_cellline_data[:, common_genes]

    return spatial_data, bulk_data, sc_tumor_data, sc_cellline_data, common_genes

def prepare_tensors(spatial_data, bulk_data, sc_tumor_data, sc_cellline_data, device):
    # Prepare tensors
    bulk_data_X = torch.tensor(bulk_data.drop(columns='PACLITAXEL').values).float()
    bulk_data_y = torch.tensor((bulk_data['PACLITAXEL'] == 'sensitive').values).float()
    cell_line_X = torch.tensor(sc_cellline_data.X).float()
    cell_line_y = torch.tensor((sc_cellline_data.obs['sensitive'] == 1).values).float()
    tumor_X = torch.tensor(sc_tumor_data.X).float()
    spatial_X = torch.tensor(spatial_data.X).float()

    # Move tensors to device
    bulk_data_X, bulk_data_y = bulk_data_X.to(device), bulk_data_y.to(device)
    cell_line_X, cell_line_y = cell_line_X.to(device), cell_line_y.to(device)
    tumor_X = tumor_X.to(device)
    spatial_X = spatial_X.to(device)

    return {
        'spatial': (spatial_X, None),
        'bulk': (bulk_data_X, bulk_data_y),
        'sc_tumor': (tumor_X, None),
        'sc_cellline': (cell_line_X, cell_line_y)
    }


def spatial_to_graph(adata, k=5, device='cpu'):
    # Spatial coordinates (normalized)
    coords = torch.tensor(adata.obsm['spatial_scaled'], dtype=torch.float32)

    # Compute pairwise distances
    distances = torch.cdist(coords, coords)  # Shape: (N, N)

    # Get k-nearest neighbors (indices and distances)
    topk_distances, topk_indices = torch.topk(distances, k=k, dim=1, largest=False)

    # Create edge_index: source nodes (repeated) and target nodes (k-NN indices)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(len(coords)), k),  # Source nodes
        topk_indices.flatten()                                 # Target nodes
    ])

    # Compute edge weights (e.g., inverse distance or normalized distance)
    edge_weights = 1 / (topk_distances.flatten() + 1e-6)  # Inverse distance, avoid division by zero

    # Normalize weights (optional, to ensure consistent scale)
    edge_weights = edge_weights / edge_weights.max()  # Normalize to [0, 1]

    # Move to device
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)

    return edge_index, edge_weights