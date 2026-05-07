"""GNN model definition for graph-level fraud classification."""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.pool = global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.pool(x, batch)
        return self.classifier(x)
