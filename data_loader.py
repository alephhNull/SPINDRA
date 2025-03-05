import scanpy as sc
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import deque


def load_data(folder='preprocessed'):
    # Load datasets
    spatial_data = sc.read(f"{folder}/spatial/GSM6592061_M15.h5ad")
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


def prepare_tensors(spatial_data, bulk_data, sc_tumor_data, sc_cellline_data, device, k=10, test_size=0.2):
    """
    Prepare tensors with a balanced train-test split for labeled data and random splits for unlabeled data.
    
    Args:
        spatial_data: AnnData object for spatial data (unlabeled).
        bulk_data: DataFrame with bulk data and 'PACLITAXEL' labels.
        sc_tumor_data: AnnData object for single-cell tumor data (unlabeled).
        sc_cellline_data: AnnData object with single-cell cell line data and 'sensitive' labels.
        device: Torch device (e.g., 'cpu' or 'cuda').
        test_size: Proportion of data to use for validation (default: 0.2).
    
    Returns:
        dict: Dictionary with tensors for spatial (train/val), bulk (train/val), sc_tumor (train/val), and sc_cellline (train/val).
    """
    # Split bulk data (labeled)
    bulk_data['label'] = (bulk_data['PACLITAXEL'] == 'sensitive').astype(int)
    bulk_train, bulk_val = train_test_split(
        bulk_data, 
        test_size=test_size, 
        stratify=bulk_data['label'],
        random_state=42  # For reproducibility
    )

    # Split single-cell cell line data (labeled)
    sc_cellline_labels = sc_cellline_data.obs['sensitive'].values
    indices = range(sc_cellline_data.shape[0])
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=test_size, 
        stratify=sc_cellline_labels,
        random_state=42  # For reproducibility
    )
    sc_cellline_train = sc_cellline_data[train_idx, :]
    sc_cellline_val = sc_cellline_data[val_idx, :]


    # Split sc_tumor data (unlabeled)
    sc_tumor_labels = sc_tumor_data.obs['condition'].values
    sc_tumor_indices = range(sc_tumor_data.shape[0])
    sc_tumor_train_idx, sc_tumor_val_idx = train_test_split(
        sc_tumor_indices, 
        test_size=test_size,
        stratify=sc_tumor_labels, 
        random_state=42  # For reproducibility
    )
    sc_tumor_train = sc_tumor_data[sc_tumor_train_idx, :]
    sc_tumor_val = sc_tumor_data[sc_tumor_val_idx, :]

    # Prepare tensors for bulk data
    bulk_train_X = torch.tensor(bulk_train.drop(columns=['PACLITAXEL', 'label']).values).float().to(device)
    bulk_train_y = torch.tensor(bulk_train['label'].values).float().to(device)
    bulk_val_X = torch.tensor(bulk_val.drop(columns=['PACLITAXEL', 'label']).values).float().to(device)
    bulk_val_y = torch.tensor(bulk_val['label'].values).float().to(device)

    # Prepare tensors for single-cell cell line data
    sc_cellline_train_X = torch.tensor(sc_cellline_train.X).float().to(device)
    sc_cellline_train_y = torch.tensor(sc_cellline_train.obs['sensitive'].values).float().to(device)
    sc_cellline_val_X = torch.tensor(sc_cellline_val.X).float().to(device)
    sc_cellline_val_y = torch.tensor(sc_cellline_val.obs['sensitive'].values).float().to(device)

    spatial_train, spatial_val, edge_index_train, edge_index_val, edge_weights_train, edge_weights_val = neighborhood_based_split(spatial_data, k, test_size, device)

    # Prepare tensors for sc_tumor data (unlabeled)
    sc_tumor_train_X = torch.tensor(sc_tumor_train.X).float().to(device)
    sc_tumor_train_y = torch.tensor(sc_tumor_train.obs['condition'] == 'sensitive').float().to(device)
    sc_tumor_val_X = torch.tensor(sc_tumor_val.X).float().to(device)
    sc_tumor_val_y = torch.tensor(sc_tumor_val.obs['condition'] == 'sensitive').float().to(device)

    # Return dictionary with train and validation tensors
    return {
        'spatial_train': (spatial_train, None),
        'spatial_val': (spatial_val, None),
        'bulk_train': (bulk_train_X, bulk_train_y),
        'bulk_val': (bulk_val_X, bulk_val_y),
        'sc_tumor_train': (sc_tumor_train_X, sc_tumor_train_y),
        'sc_tumor_val': (sc_tumor_val_X, sc_tumor_val_y),
        'sc_cellline_train': (sc_cellline_train_X, sc_cellline_train_y),
        'sc_cellline_val': (sc_cellline_val_X, sc_cellline_val_y),
        'edge_index_train': edge_index_train,
        'edge_index_val': edge_index_val,
        'edge_weights_train': edge_weights_train,
        'edge_weights_val': edge_weights_val
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


def neighborhood_based_split(adata, k=10, test_size=0.2, device='cpu'):
    """
    Split spatial data into training and test sets, ensuring each forms a connected subgraph.

    Args:
        adata: AnnData object with spatial coordinates in adata.obsm['spatial_scaled'].
        k: Number of neighbors for k-NN graph (default: 10).
        test_size: Proportion of nodes for the test set (default: 0.2).

    Returns:
        tuple: (train_features, test_features, train_edges, train_weights, test_edges, test_weights)
    """
    # Extract spatial coordinates
    coords = torch.tensor(adata.obsm['spatial_scaled'], dtype=torch.float32)

    # Compute pairwise distances and build k-NN graph
    distances = torch.cdist(coords, coords)
    _, topk_indices = torch.topk(distances, k=k, dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(len(coords)), k),
        topk_indices.flatten()
    ])

    # Build adjacency list for BFS
    adj_list = {i: [] for i in range(len(coords))}
    for src, tgt in edge_index.t().tolist():
        adj_list[src].append(tgt)
        adj_list[tgt].append(src)  # Undirected graph

    # Use BFS to select a connected test set
    test_nodes = set()
    queue = deque([np.random.randint(0, len(coords))])  # Random starting node
    target_test_size = int(test_size * len(coords))

    while queue and len(test_nodes) < target_test_size:
        current = queue.popleft()
        if current not in test_nodes:
            test_nodes.add(current)
            for neighbor in adj_list[current]:
                if neighbor not in test_nodes:
                    queue.append(neighbor)

    test_nodes = list(test_nodes)
    train_nodes = list(set(range(len(coords))) - set(test_nodes))

    # Create training subgraph
    train_mask = np.isin(edge_index[0].numpy(), train_nodes) & np.isin(edge_index[1].numpy(), train_nodes)
    edge_index_train = edge_index[:, train_mask]
    edge_weights_train = 1 / (distances[edge_index[0, train_mask], edge_index[1, train_mask]] + 1e-6)
    edge_weights_train = edge_weights_train.to(device)

    # Create test subgraph
    test_mask = np.isin(edge_index[0].numpy(), test_nodes) & np.isin(edge_index[1].numpy(), test_nodes)
    edge_index_test = edge_index[:, test_mask]
    edge_weights_test = 1 / (distances[edge_index[0, test_mask], edge_index[1, test_mask]] + 1e-6)
    edge_weights_test = edge_weights_test.to(device)

    # Extract features
    train_features = torch.tensor(adata[train_nodes, :].X, dtype=torch.float32).to(device)
    test_features = torch.tensor(adata[test_nodes, :].X, dtype=torch.float32).to(device)

    # Remap edge indices to subset indices
    train_node_map = {old: new for new, old in enumerate(train_nodes)}
    test_node_map = {old: new for new, old in enumerate(test_nodes)}
    edge_index_train = torch.stack([
        torch.tensor([train_node_map[src.item()] for src in edge_index[0, train_mask]], dtype=torch.long),
        torch.tensor([train_node_map[tgt.item()] for tgt in edge_index[1, train_mask]], dtype=torch.long)
    ]).to(device)

    edge_index_test = torch.stack([
        torch.tensor([test_node_map[src.item()] for src in edge_index[0, test_mask]], dtype=torch.long),
        torch.tensor([test_node_map[tgt.item()] for tgt in edge_index[1, test_mask]], dtype=torch.long)
    ]).to(device)

    return (train_features, test_features, edge_index_train, edge_index_test,
            edge_weights_train, edge_weights_test)

