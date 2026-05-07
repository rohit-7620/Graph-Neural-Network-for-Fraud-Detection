"""Synthetic dataset utilities for GNN fraud detection example.

Generates simple graph-level classification graphs where the label has a
structural signal (e.g., presence of a high-degree hub indicates 'fraud').
"""
from typing import List
import random
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def generate_synthetic_graph(num_nodes: int = 50, feat_dim: int = 16, p: float = 0.05, seed: int | None = None) -> Data:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    G = nx.erdos_renyi_graph(num_nodes, p)
    if G.number_of_edges() == 0:
        G = nx.watts_strogatz_graph(num_nodes, 4, 0.1)

    x = np.random.normal(size=(num_nodes, feat_dim)).astype(np.float32)
    degs = np.array([d for _, d in G.degree()])

    # Simple labeling rule: if a graph contains a high-degree node, mark as fraud.
    label = 1 if (degs.max() > num_nodes * 0.2) else 0

    # Attach features to networkx nodes so `from_networkx` picks them up if needed.
    for i in G.nodes():
        G.nodes[i]["x"] = x[i].tolist()

    data = from_networkx(G)
    data.x = torch.tensor(x)
    data.y = torch.tensor([label], dtype=torch.long)
    return data


def create_synthetic_dataset(num_graphs: int = 200, **kwargs) -> List[Data]:
    return [generate_synthetic_graph(**kwargs) for _ in range(num_graphs)]
