# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
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
    def __init__(self, input_dim, hidden_dim=128, edge_index=None):
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