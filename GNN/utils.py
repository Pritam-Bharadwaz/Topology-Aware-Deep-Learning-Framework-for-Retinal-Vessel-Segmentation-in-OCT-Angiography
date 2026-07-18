"""Reusable, model-agnostic utilities for the GNN training and inference
pipeline: seeding, device selection, checkpointing, and parameter counting.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch_geometric.data import Data


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: The seed value to use across all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select the best available compute device.

    Returns:
        A ``cuda`` device if a GPU is available, otherwise ``cpu``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_data_to_device(data: Data, device: torch.device) -> Data:
    """Move a PyG ``Data`` (or batched ``Data``) object to a device.

    Args:
        data: The graph data object to move.
        device: Target device.

    Returns:
        The data object, moved to ``device``.
    """
    return data.to(device)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a model.

    Args:
        model: The PyTorch model to inspect.
        trainable_only: If True, count only parameters with
            ``requires_grad=True``.

    Returns:
        The total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint.

    Args:
        filepath: Destination path for the checkpoint file. Parent
            directories are created if they do not exist.
        model: The model whose state should be saved.
        optimizer: Optional optimizer whose state should be saved.
        epoch: The epoch number associated with this checkpoint.
        metrics: Optional dictionary of metrics to store alongside the
            checkpoint (for example validation loss/accuracy).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint into a model (and optionally an optimizer).

    Args:
        filepath: Path to the checkpoint file.
        model: Model instance to load weights into. Must have the same
            architecture as the model the checkpoint was saved from.
        optimizer: Optional optimizer instance to restore state into.
        device: Device to map the checkpoint tensors to. Defaults to CPU.

    Returns:
        The full checkpoint dictionary (including ``epoch`` and ``metrics``).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    map_location = device or torch.device("cpu")
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint