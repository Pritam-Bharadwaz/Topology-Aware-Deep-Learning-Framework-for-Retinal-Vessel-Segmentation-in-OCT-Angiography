"""Training entry point for the vessel-graph GCN.

Loads GraphML-derived graphs via :mod:`GNN.dataset`, trains
:class:`GNN.gcn_model.VesselGCN`, and saves checkpoints and training history.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from GNN.dataset import VesselGraphDataset
from GNN.gcn_model import VesselGCN
from GNN.utils import (
    count_parameters,
    get_device,
    move_data_to_device,
    save_checkpoint,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_dataset(
    dataset: VesselGraphDataset, val_ratio: float = 0.2, seed: int = 42
) -> Tuple[Subset, Subset]:
    """Split a dataset of graphs into training and validation subsets.

    The split operates over whole graphs (not individual nodes), so a
    validation graph is never partially seen during training.

    Args:
        dataset: The full ``VesselGraphDataset``.
        val_ratio: Fraction of graphs reserved for validation.
        seed: Random seed governing the split, for reproducibility.

    Returns:
        A ``(train_subset, val_subset)`` tuple.

    Raises:
        ValueError: If the dataset does not contain enough graphs to split.
    """
    num_graphs = len(dataset)
    if num_graphs < 2:
        raise ValueError(
            f"Need at least 2 graphs to create a train/validation split, got {num_graphs}."
        )

    num_val = max(1, int(round(num_graphs * val_ratio)))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_graphs, generator=generator).tolist()

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def run_epoch(
    model: VesselGCN,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    """Run one epoch of training or evaluation.

    Args:
        model: The GCN model.
        loader: DataLoader yielding batched graphs.
        device: Device to run computation on.
        optimizer: If provided, the model is put in training mode and
            gradients are applied. If ``None``, the model runs in
            evaluation mode with no gradient updates.

    Returns:
        A ``(average_loss, accuracy)`` tuple over the epoch, both averaged
        per node.
    """
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0
    total_nodes = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_data_to_device(batch, device)
            edge_weight = getattr(batch, "edge_weight", None)

            logits = model(batch.x, batch.edge_index, edge_weight)
            loss = F.cross_entropy(
              logits,
              batch.y.long(),
            )

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()

            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == batch.y).sum().item())
            total_nodes += batch.y.size(0)
            total_loss += loss.item() * batch.y.size(0)

    avg_loss = total_loss / max(total_nodes, 1)
    accuracy = total_correct / max(total_nodes, 1)
    return avg_loss, accuracy


def train(
    dataset_dir: Union[str, Path],
    output_dir: Union[str, Path] = "Checkpoints",
    hidden_dim: int = 64,
    output_dim: Optional[int] = None,
    dropout: float = 0.5,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    batch_size: int = 4,
    max_epochs: int = 200,
    val_ratio: float = 0.2,
    early_stopping_patience: int = 20,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Train the vessel-graph GCN and save checkpoints.

    Args:
        dataset_dir: Directory containing vessel-graph ``*.graphml`` files.
        output_dir: Directory to write checkpoints and training history to.
        hidden_dim: Hidden channel size for the GCN.
        output_dim: Number of output classes.
        dropout: Dropout probability.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay (L2 regularization).
        batch_size: Number of graphs per training batch.
        max_epochs: Maximum number of training epochs.
        val_ratio: Fraction of graphs used for validation.
        early_stopping_patience: Number of epochs without validation loss
            improvement before training stops early.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary containing per-epoch training/validation loss and
        accuracy history.

    Raises:
        ValueError: If no GraphML files are found in ``dataset_dir``.
    """
    set_seed(seed)
    device = get_device()
    logger.info("Using device: %s", device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = VesselGraphDataset(root_dir=dataset_dir)
    if len(dataset) == 0:
        raise ValueError(f"No GraphML files found in {dataset_dir}.")

    train_subset, val_subset = split_dataset(dataset, val_ratio=val_ratio, seed=seed)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    sample = dataset[0]
    input_dim = sample.x.size(1)

    if output_dim is None:
        output_dim = int(sample.y.max().item()) + 1

        logger.info("Input feature dimension : %d", input_dim)
        logger.info("Number of output classes: %d", output_dim)
        logger.info("Training graphs         : %d", len(train_subset))
        logger.info("Validation graphs       : %d", len(val_subset))

    model = VesselGCN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
    ).to(device)
    logger.info("Model has %d trainable parameters", count_parameters(model))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay

    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "best_model.pth"

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        logger.info(
            "Epoch %03d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        save_checkpoint(
            output_dir / "last_model.pth",
            model,
            optimizer,
            epoch=epoch,
            metrics={"val_loss": val_loss, "val_accuracy": val_acc},
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch=epoch,
                metrics={"val_loss": val_loss, "val_accuracy": val_acc},
            )
            logger.info("New best model saved (val_loss=%.4f)", val_loss)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    return history


if __name__ == "__main__":
    train(
        dataset_dir="Graph/Graphs",
        output_dir="Checkpoints",
    )