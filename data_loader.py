import scanpy as sc
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import deque
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import os
import umap

def load_data(spatial_file_path, sc_tumor_file_path, sc_cellline_file_path):
    # Load datasets
    spatial_file_name = spatial_file_path.split('/')[-1]
    sc_tumor_file_name = sc_tumor_file_path.split('/')[-1]
    sc_cellline_file_name = sc_cellline_file_path.split('/')[-1]

    spatial_data = sc.read(f'preprocessed/spatial/{spatial_file_name}')
    bulk_data = pd.read_csv(f"preprocessed/bulk/bulk_data.csv")
    sc_tumor_data = sc.read(f'preprocessed/sc-tumor/{sc_tumor_file_name}')
    sc_cellline_data = sc.read(f'preprocessed/sc-cell-line/{sc_cellline_file_name}')

    print(f'Shapes of preprocessed data:\n Spatial: {spatial_data.shape}, Bulk: {bulk_data.shape}, Single-Cell Tumor: {sc_tumor_data.shape}, Single-Cell Cellline: {sc_cellline_data.shape}')
    
    # Extract common genes
    common_genes = list(
        set(spatial_data.var_names) &
        set(bulk_data.columns) &
        set(sc_tumor_data.var_names) &
        set(sc_cellline_data.var_names)
    )

    print(f'Length of common genes: {len(common_genes)}')

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

    save_dir = "figures_umap"
    os.makedirs(save_dir, exist_ok=True)

    feature_cols = [col for col in bulk_data.columns if col not in ['label', 'PACLITAXEL']]
    label_map = {0: 'Resistant', 1: 'Sensitive'}
    color_dict = {'Sensitive': 'red', 'Resistant': 'blue'}

    def get_labels_named(df):
        return df['label'].map(label_map).values

    X_bulk = bulk_data[feature_cols].values
    X_train = bulk_train[feature_cols].values
    X_val = bulk_val[feature_cols].values

    bulk_labels_named = get_labels_named(bulk_data)
    train_labels_named = get_labels_named(bulk_train)
    val_labels_named = get_labels_named(bulk_val)

    # -------- UMAP --------
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap_bulk = reducer.fit_transform(X_bulk)
    X_umap_train = reducer.transform(X_train)
    X_umap_val = reducer.transform(X_val)

    df_umap_bulk = pd.DataFrame(X_umap_bulk, columns=['UMAP1', 'UMAP2'])
    df_umap_bulk['Label'] = bulk_labels_named
    df_umap_bulk['Split'] = 'None'
    df_umap_bulk.loc[bulk_train.index, 'Split'] = 'Train'
    df_umap_bulk.loc[bulk_val.index, 'Split'] = 'Val'

    df_umap_train = pd.DataFrame(X_umap_train, columns=['UMAP1', 'UMAP2'])
    df_umap_train['Label'] = train_labels_named

    df_umap_val = pd.DataFrame(X_umap_val, columns=['UMAP1', 'UMAP2'])
    df_umap_val['Label'] = val_labels_named

    # --------- پلات bulk (Train/Val) ----------
    plt.figure(figsize=(8, 7))
    for label in color_dict:
        for split, marker in zip(['Train', 'Val', 'None'], ['o', 's', 'x']):
            sub = df_umap_bulk[(df_umap_bulk['Label'] == label) & (df_umap_bulk['Split'] == split)]
            if not sub.empty:
                plt.scatter(sub['UMAP1'], sub['UMAP2'],
                            c=color_dict[label],
                            label=f'{label}-{split}' if split != 'None' else f'{label}-Other',
                            s=70 if split != 'None' else 30,
                            marker=marker,
                            alpha=0.7,
                            edgecolor='k' if split != 'None' else None)
    plt.title('UMAP of All Bulk Data (Train/Val split shown)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_bulk_split.png"), dpi=300)
    plt.close()

    # --------- پلات فقط train ----------
    plt.figure(figsize=(6, 5))
    for label in color_dict:
        sub = df_umap_train[df_umap_train['Label'] == label]
        plt.scatter(sub['UMAP1'], sub['UMAP2'],
                    c=color_dict[label], label=label, s=70, alpha=0.7, edgecolor='k')
    plt.title('UMAP of Train Data')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_train.png"), dpi=300)
    plt.close()

    # --------- پلات فقط validation ----------
    plt.figure(figsize=(6, 5))
    for label in color_dict:
        sub = df_umap_val[df_umap_val['Label'] == label]
        plt.scatter(sub['UMAP1'], sub['UMAP2'],
                    c=color_dict[label], label=label, s=70, alpha=0.7, edgecolor='k')
    plt.title('UMAP of Validation Data')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_val.png"), dpi=300)
    plt.close()

    print(f"UMAP plots for bulk data saved in {os.path.abspath(save_dir)}")


    # Split single-cell cell line data (labeled)
    # لیبل و رنگ‌ها
    sc_cellline_labels = sc_cellline_data.obs['condition'].values
    indices = range(sc_cellline_data.shape[0])
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=test_size, 
        stratify=sc_cellline_labels,
        random_state=42  # For reproducibility
    )
    sc_cellline_train = sc_cellline_data[train_idx, :]
    sc_cellline_val = sc_cellline_data[val_idx, :]
    label_map = {'sensitive': 'Sensitive', 'resistant': 'Resistant'}
    color_dict = {'Sensitive': 'red', 'Resistant': 'blue'}
    labels_named = pd.Series(sc_cellline_data.obs['condition']).map(label_map).values

    # کل دیتا
    X_all = sc_cellline_data.X.toarray() if hasattr(sc_cellline_data.X, "toarray") else sc_cellline_data.X
    X_train = X_all[train_idx]
    X_val = X_all[val_idx]

    labels_train = labels_named[train_idx]
    labels_val = labels_named[val_idx]

    # -------- UMAP --------
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap_all = reducer.fit_transform(X_all)
    X_umap_train = reducer.transform(X_train)
    X_umap_val = reducer.transform(X_val)

    df_umap_all = pd.DataFrame(X_umap_all, columns=['UMAP1', 'UMAP2'])
    df_umap_all['Label'] = labels_named
    df_umap_all['Split'] = 'None'
    df_umap_all.loc[train_idx, 'Split'] = 'Train'
    df_umap_all.loc[val_idx, 'Split'] = 'Val'

    df_umap_train = pd.DataFrame(X_umap_train, columns=['UMAP1', 'UMAP2'])
    df_umap_train['Label'] = labels_train

    df_umap_val = pd.DataFrame(X_umap_val, columns=['UMAP1', 'UMAP2'])
    df_umap_val['Label'] = labels_val

    # --------- پلات UMAP کل دیتا (Train/Val) ----------
    plt.figure(figsize=(8, 7))
    for label in color_dict:
        for split, marker in zip(['Train', 'Val', 'None'], ['o', 's', 'x']):
            sub = df_umap_all[(df_umap_all['Label'] == label) & (df_umap_all['Split'] == split)]
            if not sub.empty:
                plt.scatter(sub['UMAP1'], sub['UMAP2'],
                            c=color_dict[label],
                            label=f'{label}-{split}' if split != 'None' else f'{label}-Other',
                            s=70 if split != 'None' else 30,
                            marker=marker,
                            alpha=0.7,
                            edgecolor='k' if split != 'None' else None)
    plt.title('UMAP of sc_cellline_data (Train/Val split shown)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_sc_cellline_all_split.png"), dpi=300)
    plt.close()

    # --------- پلات فقط train ----------
    plt.figure(figsize=(6, 5))
    for label in color_dict:
        sub = df_umap_train[df_umap_train['Label'] == label]
        plt.scatter(sub['UMAP1'], sub['UMAP2'],
                    c=color_dict[label], label=label, s=70, alpha=0.7, edgecolor='k')
    plt.title('UMAP of sc_cellline_data (Train)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_sc_cellline_train.png"), dpi=300)
    plt.close()

    # --------- پلات فقط validation ----------
    plt.figure(figsize=(6, 5))
    for label in color_dict:
        sub = df_umap_val[df_umap_val['Label'] == label]
        plt.scatter(sub['UMAP1'], sub['UMAP2'],
                    c=color_dict[label], label=label, s=70, alpha=0.7, edgecolor='k')
    plt.title('UMAP of sc_cellline_data (Validation)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_sc_cellline_val.png"), dpi=300)
    plt.close()

    print(f"UMAP plots for sc_cellline_data saved in {os.path.abspath(save_dir)}")






    # Split sc_tumor data (labeled)
    
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

    label_map = {'sensitive': 'Sensitive', 'resistant': 'Resistant'}
    color_dict = {'Sensitive': 'red', 'Resistant': 'blue'}
    labels_named = pd.Series(sc_tumor_data.obs['condition']).map(label_map).values

    # داده عددی
    X_all = sc_tumor_data.X.toarray() if hasattr(sc_tumor_data.X, "toarray") else sc_tumor_data.X
    X_train = X_all[sc_tumor_train_idx]
    X_val = X_all[sc_tumor_val_idx]

    labels_train = labels_named[sc_tumor_train_idx]
    labels_val = labels_named[sc_tumor_val_idx]

    # -------- UMAP --------
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap_all = reducer.fit_transform(X_all)
    X_umap_train = reducer.transform(X_train)
    X_umap_val = reducer.transform(X_val)

    df_umap_all = pd.DataFrame(X_umap_all, columns=['UMAP1', 'UMAP2'])
    df_umap_all['Label'] = labels_named
    df_umap_all['Split'] = 'None'
    df_umap_all.loc[sc_tumor_train_idx, 'Split'] = 'Train'
    df_umap_all.loc[sc_tumor_val_idx, 'Split'] = 'Val'

    df_umap_train = pd.DataFrame(X_umap_train, columns=['UMAP1', 'UMAP2'])
    df_umap_train['Label'] = labels_train

    df_umap_val = pd.DataFrame(X_umap_val, columns=['UMAP1', 'UMAP2'])
    df_umap_val['Label'] = labels_val

    # --------- پلات UMAP کل داده (Train/Val) ----------
    plt.figure(figsize=(8, 7))
    for label in color_dict:
        for split, marker in zip(['Train', 'Val', 'None'], ['o', 's', 'x']):
            sub = df_umap_all[(df_umap_all['Label'] == label) & (df_umap_all['Split'] == split)]
            if not sub.empty:
                plt.scatter(sub['UMAP1'], sub['UMAP2'],
                            c=color_dict[label],
                            label=f'{label}-{split}' if split != 'None' else f'{label}-Other',
                            s=70 if split != 'None' else 30,
                            marker=marker,
                            alpha=0.7,
                            edgecolor='k' if split != 'None' else None)
    plt.title('UMAP of sc_tumor_data (Train/Val split shown)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_sc_tumor_all_split.png"), dpi=300)
    plt.close()

    # --------- پلات فقط train ----------
    plt.figure(figsize=(6, 5))
    for label in color_dict:
        sub = df_umap_train[df_umap_train['Label'] == label]
        plt.scatter(sub['UMAP1'], sub['UMAP2'],
                    c=color_dict[label], label=label, s=70, alpha=0.7, edgecolor='k')
    plt.title('UMAP of sc_tumor_data (Train)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_sc_tumor_train.png"), dpi=300)
    plt.close()

    # --------- پلات فقط validation ----------
    plt.figure(figsize=(6, 5))
    for label in color_dict:
        sub = df_umap_val[df_umap_val['Label'] == label]
        plt.scatter(sub['UMAP1'], sub['UMAP2'],
                    c=color_dict[label], label=label, s=70, alpha=0.7, edgecolor='k')
    plt.title('UMAP of sc_tumor_data (Validation)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_sc_tumor_val.png"), dpi=300)
    plt.close()

    print(f"UMAP plots for sc_tumor_data saved in {os.path.abspath(save_dir)}")

    # Prepare tensors for bulk data
    bulk_train_X = torch.tensor(bulk_train.drop(columns=['PACLITAXEL', 'label']).values).float().to(device)
    bulk_train_y = torch.tensor(bulk_train['label'].values).float().to(device)
    bulk_val_X = torch.tensor(bulk_val.drop(columns=['PACLITAXEL', 'label']).values).float().to(device)
    bulk_val_y = torch.tensor(bulk_val['label'].values).float().to(device)
    
    
    # Prepare tensors for single-cell cell line data
    sc_cellline_train_X = torch.tensor(sc_cellline_train.X).float().to(device)
    sc_cellline_train_y = torch.tensor(sc_cellline_train.obs['condition'] == 'sensitive').float().to(device)
    sc_cellline_val_X = torch.tensor(sc_cellline_val.X).float().to(device)
    sc_cellline_val_y = torch.tensor(sc_cellline_val.obs['condition'] == 'sensitive').float().to(device)

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

