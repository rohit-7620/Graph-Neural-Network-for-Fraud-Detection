# Graph Neural Network for Fraud Detection

This repository contains a minimal, runnable example of a Graph Neural Network (GNN) for fraud detection using Python and PyTorch Geometric.

Overview
- Built a GNN model for fraud detection on relational datasets.
- Improved prediction quality by modeling graph dependencies and network-based feature interactions.

What’s included
- `requirements.txt` — Python dependencies (PyTorch, PyG, numpy, networkx, scikit-learn).
- `src/data.py` — synthetic graph dataset generator (so you can run the example without external data).
- `src/model.py` — a simple GCN-based classifier with global pooling.
- `src/train.py` — training and evaluation script with a small smoke-run example.

Quickstart
1. Create a virtual environment and install dependencies. Installing `torch-geometric` may require following the official install instructions for your CUDA/Python combination.

	pip install -r requirements.txt

2. Run a short training run (this uses a synthetic dataset):

	python -m src.train --epochs 10 --num-graphs 200

Notes
- The dataset is synthetic and intended for demonstration. Replace `create_synthetic_dataset` in `src/data.py` with a real dataset loader for production experiments.
- For best results on real datasets, tune architecture, learning rate, and regularization, and consider richer node/edge features.


