import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch.autograd import Function

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

class DrugResponsePredictor(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout added after ReLU
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.classifier(x)

class SpatialEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, use_edge=False, dropout_rate=0.5):
        super().__init__()
        self.use_gnn = False
        if use_edge:
            self.use_gnn = True
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        else:
            self.conv1 = nn.Linear(input_dim, hidden_dim)
            self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index=None, edge_weights=None):
        if self.use_gnn and edge_index is not None:
            # Use edge_weights if provided
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
        else:
            x = F.relu(self.conv1(x))
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.dropout(x)
        # x = self.fc(x)
        return x


class ImprovedSpatialEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout_rate=0.5, use_gat=False, edge_index=None):
        super().__init__()
        self.use_gnn = False
        if edge_index is not None:
            self.use_gnn = True
            if use_gat:
                self.conv_layers = nn.ModuleList([
                    GATConv(input_dim if i == 0 else hidden_dim, hidden_dim // 4, heads=4, dropout=dropout_rate)
                    for i in range(num_layers)
                ])
            else:
                self.conv_layers = nn.ModuleList([
                    GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim)
                    for i in range(num_layers)
                ])
            # Residual connection from input to hidden dimension
            self.residual = nn.Linear(input_dim, hidden_dim) if num_layers > 1 else None
        else:
            # Fully connected layers if no graph structure is provided
            self.fc_layers = nn.ModuleList([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])
            self.residual = nn.Linear(input_dim, hidden_dim*4) if num_layers > 1 else None
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, edge_index=None, edge_weights=None):
        if self.use_gnn and edge_index is not None:
            # Apply residual connection if available
            residual = self.residual(x) if self.residual else None
            for i, conv in enumerate(self.conv_layers):
                if isinstance(conv, GATConv):
                    # Handle multi-head output from GAT
                    x = conv(x, edge_index)
                    x = x.view(x.size(0), -1)  # Flatten multi-head output
                else:
                    # Use edge weights if provided for GCN
                    x = conv(x, edge_index, edge_weight=edge_weights) if edge_weights is not None else conv(x, edge_index)
                x = F.relu(x)  # Activation
                x = self.dropout(x)  # Dropout
                # Add residual connection after the first layer
                if i == 0 and residual is not None:
                    x += residual
        else:
            # Fully connected path without GNN
            residual = self.residual(x) if self.residual else None
            for i, fc in enumerate(self.fc_layers):
                x = fc(x)
                x = F.relu(x)
                x = self.dropout(x)
                if i == 0 and residual is not None:
                    x += residual
        return x

class BulkEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after ReLU
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after ReLU
        return x

class SingleCellEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after ReLU
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after ReLU
        return x

class TumorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after ReLU
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after ReLU
        return x

class DomainDiscriminator(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Additional hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 4 domains
        )
    def forward(self, x):
        return self.classifier(x)