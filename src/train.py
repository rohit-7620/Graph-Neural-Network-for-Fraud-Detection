"""Training and evaluation script for the GNN fraud detection example.

Example:
    python -m src.train --epochs 10 --num-graphs 200
"""
import argparse
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score

from src.data import create_synthetic_dataset
from src.model import GNNClassifier


def train_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = torch.nn.functional.cross_entropy(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def eval_model(model, loader, device):
    model.eval()
    ys = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            p = out.argmax(dim=-1).cpu().numpy()
            preds.extend(p.tolist())
            ys.extend(batch.y.view(-1).cpu().numpy().tolist())
    if len(ys) == 0:
        return 0.0
    return float(accuracy_score(ys, preds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-graphs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = create_synthetic_dataset(num_graphs=args.num_graphs, num_nodes=50, feat_dim=16, p=0.05, seed=args.seed)

    # simple random split
    split = int(0.8 * len(dataset))
    train_ds = dataset[:split]
    val_ds = dataset[split:]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = GNNClassifier(in_channels=16, hidden=64, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training on {len(train_ds)} graphs, validating on {len(val_ds)} graphs")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc = eval_model(model, val_loader, device)
        print(f"Epoch {epoch:03d}: loss={train_loss:.4f} val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
