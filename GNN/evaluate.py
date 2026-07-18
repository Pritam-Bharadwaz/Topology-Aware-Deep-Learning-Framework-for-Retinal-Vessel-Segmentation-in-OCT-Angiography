"""Evaluation utilities for a trained vessel-graph GCN.

Computes standard node-classification metrics (accuracy, precision, recall,
F1 score, confusion matrix) over one or more test graphs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.loader import DataLoader

from GNN.dataset import VesselGraphDataset
from GNN.gcn_model import VesselGCN
from GNN.utils import get_device, load_checkpoint, move_data_to_device

logger = logging.getLogger(__name__)


def evaluate_model(
    model: VesselGCN,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    """Evaluate a trained model on a set of graphs.

    Args:
        model: A trained ``VesselGCN`` instance.
        loader: DataLoader yielding batched evaluation graphs.
        device: Device to run inference on.

    Returns:
        A dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1_score`` (all macro-averaged across classes), and
        ``confusion_matrix`` (as a nested list of ints).
    """
    model.eval()
    all_predictions: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_data_to_device(batch, device)
            edge_weight = getattr(batch, "edge_weight", None)
            logits = model(batch.x, batch.edge_index, edge_weight)
            predictions = logits.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    metrics: Dict[str, object] = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "f1_score": f1_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions,labels=[0, 1, 2],).tolist(),
    }
    return metrics


def evaluate_from_checkpoint(
    checkpoint_path: Union[str, Path],
    dataset_dir: Union[str, Path],
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float = 0.5,
    batch_size: int = 4,
) -> Dict[str, object]:
    """Load a checkpoint and evaluate it on a directory of test graphs.

    Args:
        checkpoint_path: Path to a checkpoint saved by
            ``GNN.utils.save_checkpoint``.
        dataset_dir: Directory containing test ``*.graphml`` files.
        input_dim: Number of input node features (must match the trained
            model).
        hidden_dim: Hidden channel size used at training time.
        output_dim: Number of output classes used at training time.
        dropout: Dropout probability used at training time.
        batch_size: Number of graphs per evaluation batch.

    Returns:
        The evaluation metrics dictionary from :func:`evaluate_model`.

    Raises:
        ValueError: If no GraphML files are found in ``dataset_dir``.
    """
    device = get_device()
    model = VesselGCN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
    ).to(device)
    load_checkpoint(checkpoint_path, model, device=device)

    dataset = VesselGraphDataset(root_dir=dataset_dir)
    if len(dataset) == 0:
        raise ValueError(f"No GraphML files found in {dataset_dir}.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metrics = evaluate_model(model, loader, device)

    logger.info("Evaluation metrics: %s", metrics)
    return metrics

if __name__ == "__main__":
    metrics = evaluate_from_checkpoint(
        checkpoint_path="Checkpoints/best_model.pth",
        dataset_dir="Graph/Graphs",
        input_dim=6,
        hidden_dim=64,
        output_dim=3,
    )

    print("\nEvaluation Results")
    print("-" * 40)

    for key, value in metrics.items():
        print(f"{key}: {value}")