import torch
import scanpy as sc
from torch_geometric.data import Data

# adata_spatial = sc.read("preprocessed/spatial/visium_breast_cancer.h5ad")


def spatial_to_graph(adata, k=5):
    # Spatial coordinates (normalized)
    coords = adata.obsm['spatial_scaled']

    # k-NN graph for edges
    distances = torch.cdist(torch.tensor(coords), torch.tensor(coords))
    _, topk_indices = torch.topk(distances, k=k, dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(len(coords)), k),
        topk_indices.flatten()
    ])

    # Node features (gene expression)
    x = adata.X

    return Data(x=x, edge_index=edge_index)


sc_cellline_data = sc.read("preprocessed/sc-cell-line/GSE117872_HN120.h5ad") # Cell line (drug labels)

print(sc_cellline_data.obs['sensitive'])