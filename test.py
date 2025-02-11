import scanpy as sc
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.nn.functional as F


spatial_data = sc.read("preprocessed/spatial/visium_breast_cancer.h5ad")  # Spatial graph data
bulk_data = pd.read_csv('preprocessed/bulk/bulk_data.csv')
sc_tumor_data = sc.read("preprocessed/sc-tumor/GSE169246.h5ad") # Single-cell tumor (no labels)
sc_cellline_data = sc.read("preprocessed/sc-cell-line/GSE131984.h5ad") # Cell line (drug labels)
spatial_data.var_names = spatial_data.var['gene_symbol']

# Extract common genes (adjust based on your data)
common_genes = list(
    set(spatial_data.var_names) &
    set(bulk_data.columns) &
    set(sc_tumor_data.var_names) &
    set(sc_cellline_data.var_names)
)

print(common_genes)
exit()


# Subset data to common genes (if possible)
for data in [spatial_data, bulk_data, sc_tumor_data, sc_cellline_data]:
    data = data[:, common_genes].copy()


class SpatialEncoder(nn.Module):
    """Graph-based encoder for spatial data"""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # Replace with GNN layers if needed
        return x


class BulkEncoder(nn.Module):
    """Self-attention encoder for bulk RNA"""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=4)
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        x = F.relu(self.fc(x))
        return x


class SingleCellEncoder(nn.Module):
    """MLP encoder for single-cell data (tumor/cell line)"""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class DomainDiscriminator(nn.Module):
    """Adversarial discriminator to align latent spaces"""

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
spatial_encoder = SpatialEncoder(input_dim=len(common_genes))
bulk_encoder = BulkEncoder(input_dim=len(common_genes))
sc_encoder = SingleCellEncoder(input_dim=len(common_genes))
discriminator = DomainDiscriminator(hidden_dim=128)

# Optimizers
optimizer_G = torch.optim.Adam(
    list(spatial_encoder.parameters()) +
    list(bulk_encoder.parameters()) +
    list(sc_encoder.parameters()),
    lr=0.001
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# Loss functions
ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(100):
    # Sample batches from all domains
    # ---------------------------------------------------------------
    # Spatial data (graph)
    spatial_x = torch.tensor(spatial_data.X).float()
    spatial_edge_index = None  # Your spatial graph edges

    # Bulk RNA
    bulk_x = torch.tensor(bulk_data.drop(columns='CISPLATIN')).float()
    bulk_y = torch.tensor(bulk_data['CISPLATIN']).float()

    # Single-cell tumor
    sc_tumor_x = torch.tensor(sc_tumor_data.X).float()

    # Cell line
    sc_cellline_x = torch.tensor(sc_cellline_data.X).float()
    sc_cellline_y = torch.tensor(sc_cellline_data.obs["sensitive"].values).float()

    # Forward pass
    spatial_z = spatial_encoder(spatial_x, spatial_edge_index)
    bulk_z = bulk_encoder(bulk_x)
    sc_tumor_z = sc_encoder(sc_tumor_x)
    sc_cellline_z = sc_encoder(sc_cellline_x)

    # Adversarial domain alignment
    domain_labels = torch.cat([
        torch.zeros(len(spatial_z)),  # Spatial = 0
        torch.ones(len(bulk_z)),  # Bulk = 1
        2 * torch.ones(len(sc_tumor_z)),  # sc_tumor = 2
        3 * torch.ones(len(sc_cellline_z))  # sc_cellline = 3
    ]).long()

    # Train discriminator
    optimizer_D.zero_grad()
    domain_preds = discriminator(torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z]))
    loss_D = ce_loss(domain_preds, domain_labels)
    loss_D.backward()
    optimizer_D.step()

    # Train generators (encoders) to fool discriminator
    optimizer_G.zero_grad()
    domain_preds = discriminator(torch.cat([spatial_z, bulk_z, sc_tumor_z, sc_cellline_z]))
    loss_G_adv = ce_loss(domain_preds, torch.zeros_like(domain_labels))  # Try to predict wrong domain
    loss_G = loss_G_adv

    # Drug response prediction (labeled domains: bulk and cell line)
    pred_bulk = torch.sigmoid(nn.Linear(128, 1)(bulk_z))
    loss_bulk = bce_loss(pred_bulk.squeeze(), bulk_y)

    pred_cellline = torch.sigmoid(nn.Linear(128, 1)(sc_cellline_z))
    loss_cellline = bce_loss(pred_cellline.squeeze(), sc_cellline_y)

    loss_G += loss_bulk + loss_cellline
    loss_G.backward()
    optimizer_G.step()