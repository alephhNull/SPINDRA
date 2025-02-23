# data_loader.py
import scanpy as sc
import torch
import pandas as pd

def load_data(folder='preprocessed'):
    # Load datasets
    spatial_data = sc.read(f"{folder}/spatial/visium_breast_cancer.h5ad")
    bulk_data = pd.read_csv(f"{folder}/bulk/bulk_data.csv")
    sc_tumor_data = sc.read(f"{folder}/sc-tumor/GSE169246.h5ad")
    sc_cellline_data = sc.read(f"{folder}/sc-cell-line/GSE131984.h5ad")

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
    cell_line_y = torch.tensor((sc_cellline_data.obs['response'] == 'R').values).float()
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
    coords = adata.obsm['spatial_scaled']

    # k-NN graph for edges
    distances = torch.cdist(torch.tensor(coords), torch.tensor(coords))
    _, topk_indices = torch.topk(distances, k=k, dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(len(coords)), k),
        topk_indices.flatten()
    ])

    return edge_index.to(device)