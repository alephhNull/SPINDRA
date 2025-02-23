import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.autograd import Function
import numpy as np
import random

folder = 'preprocessed'

# Load data
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

spatial_data = spatial_data[:, common_genes]
bulk_data = bulk_data.loc[:, common_genes + ['PACLITAXEL']]
sc_tumor_data = sc_tumor_data[:, common_genes]
sc_cellline_data = sc_cellline_data[:, common_genes]

# Prepare tensors
bulk_data_X = torch.tensor(bulk_data.drop(columns='PACLITAXEL').values).float()
bulk_data_y = torch.tensor((bulk_data['PACLITAXEL'] == 'sensitive').values).float()

cell_line_X = torch.tensor(sc_cellline_data.X).float()
cell_line_y = torch.tensor((sc_cellline_data.obs['response'] == 'R').values).float()

tumor_X = torch.tensor(sc_tumor_data.X).float()
spatial_X = torch.tensor(spatial_data.X).float()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move tensors to device
bulk_data_X, bulk_data_y = bulk_data_X.to(device), bulk_data_y.to(device)
cell_line_X, cell_line_y = cell_line_X.to(device), cell_line_y.to(device)
tumor_X = tumor_X.to(device)
spatial_X = spatial_X.to(device)

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
    return edge_index.to(device)

edge_index = spatial_to_graph(spatial_data)

# Define a Gradient Reversal Layer (GRL)
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)

# Define Models
class DrugResponsePredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.classifier(x)

class SpatialEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.use_gnn = False
        if edge_index is not None:
            self.use_gnn = True
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        else:
            self.conv1 = nn.Linear(input_dim, hidden_dim)
            self.conv2 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index=None):
        if self.use_gnn and edge_index is not None:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
        return x

class BulkEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.residual = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) + residual
        return F.relu(x)

class SingleCellEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class DomainDiscriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 domains: spatial, bulk, sc_tumor, sc_cellline
        )
    
    def forward(self, x):
        return self.classifier(x)


# Initialize models
spatial_encoder = SpatialEncoder(input_dim=len(common_genes)).to(device)
bulk_encoder = BulkEncoder(input_dim=len(common_genes)).to(device)
sc_encoder = SingleCellEncoder(input_dim=len(common_genes)).to(device)
drug_predictor = DrugResponsePredictor(hidden_dim=128).to(device)
discriminator = DomainDiscriminator(hidden_dim=128).to(device)

# Optimizers
optimizer_G = torch.optim.Adam(
    list(spatial_encoder.parameters()) +
    list(bulk_encoder.parameters()) +
    list(sc_encoder.parameters()) +
    list(drug_predictor.parameters()),
    lr=0.001
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# Domain weights for balanced loss
N_spatial = spatial_X.size(0)
N_bulk = bulk_data_X.size(0)
N_sc_tumor = tumor_X.size(0)
N_sc_cellline = cell_line_X.size(0)
total_N = N_spatial + N_bulk + N_sc_tumor + N_sc_cellline
weights = [total_N / (4 * N_spatial), total_N / (4 * N_bulk), total_N / (4 * N_sc_tumor), total_N / (4 * N_sc_cellline)]
weights = torch.tensor(weights).to(device)

ce_loss = nn.CrossEntropyLoss(weight=weights)
bce_loss = nn.BCEWithLogitsLoss()

num_epochs = 1000
progress_bar = tqdm(range(num_epochs), desc="Training")

# Balanced batch sampling setup
domains = ['spatial', 'bulk', 'sc_tumor', 'sc_cellline']

domain_data = {
    'spatial': (spatial_X, None),  # No labels for spatial
    'bulk': (bulk_data_X, bulk_data_y),
    'sc_tumor': (tumor_X, None),  # Assuming no labels for sc_tumor
    'sc_cellline': (cell_line_X, cell_line_y)
}

for epoch in progress_bar:
    # Schedule lambda_adv
    p = float(epoch) / num_epochs
    lambda_adv = 2. / (1. + np.exp(-10. * p)) - 1
    
    # Feature extraction
    spatial_z = spatial_encoder(spatial_X, edge_index) if edge_index is not None else spatial_encoder(spatial_X)
    bulk_z = bulk_encoder(bulk_data_X)
    sc_tumor_z = sc_encoder(tumor_X)
    sc_cellline_z = sc_encoder(cell_line_X)
    
    # Domain labels: 0: spatial, 1: bulk, 2: sc_tumor, 3: sc_cellline
    domain_labels = torch.cat([
        torch.zeros(N_spatial),
        torch.ones(N_bulk),
        2 * torch.ones(N_sc_tumor),
        3 * torch.ones(N_sc_cellline)
    ]).long().to(device)
    
    
    # Train Domain Discriminator
    optimizer_D.zero_grad()
    features_detached = torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z]).detach()
    domain_preds = discriminator(features_detached)
    loss_D = ce_loss(domain_preds, domain_labels)
    loss_D.backward()
    optimizer_D.step()
    
    # Train Generators with Adversarial Loss using GRL
    optimizer_G.zero_grad()
    features = torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z])
    features_reversed = grad_reverse(features, lambda_adv)
    domain_preds_adv = discriminator(features_reversed)
    loss_G_adv = ce_loss(domain_preds_adv, domain_labels)
    
    # Shared predictor
    bulk_pred = drug_predictor(bulk_z).squeeze()
    cellline_pred = drug_predictor(sc_cellline_z).squeeze()
    loss_bulk = bce_loss(bulk_pred, bulk_data_y)
    loss_cellline = bce_loss(cellline_pred, cell_line_y)
    loss_pred = loss_bulk + loss_cellline
    
    total_loss = loss_G_adv + loss_pred
    total_loss.backward()
    optimizer_G.step()
    
    # Calculate accuracies
    with torch.no_grad():
        bulk_pred_labels = (torch.sigmoid(bulk_pred) > 0.5).float()
        bulk_accuracy = (bulk_pred_labels == bulk_data_y).float().mean().item()
        cellline_pred_labels = (torch.sigmoid(cellline_pred) > 0.5).float()
        cellline_accuracy = (cellline_pred_labels == cell_line_y).float().mean().item()
        domain_accuracy = (torch.argmax(domain_preds, dim=1) == domain_labels).float().mean().item()
    
    progress_bar.set_postfix({
        'Bulk Acc': f"{bulk_accuracy:.4f}",
        'CellLine Acc': f"{cellline_accuracy:.4f}",
        'Domain Acc': f"{domain_accuracy:.4f}",
        'G Loss': f"{total_loss.item():.4f}",
        'D Loss': f"{loss_D.item():.4f}"
    })

# Previous code remains unchanged until the end of the training loop
# ... (your existing code up to the prediction step)

# Make predictions on spatial data
with torch.no_grad():
    spatial_z = spatial_encoder(spatial_X, edge_index) if edge_index is not None else spatial_encoder(spatial_X)
    spatial_pred = drug_predictor(spatial_z).squeeze()
    spatial_pred_labels = (torch.sigmoid(spatial_pred) > 0.5).float()

print(f"Total spatial cells: {spatial_pred_labels.shape[0]}")
print(f"Sensitive cells: {spatial_pred_labels.sum().item()}")

# Add the following for visualization and evaluation
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from umap import UMAP

# Ensure spatial_pred_labels are stored in the AnnData object for visualization
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

# Plot UMAP colored by predictions
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