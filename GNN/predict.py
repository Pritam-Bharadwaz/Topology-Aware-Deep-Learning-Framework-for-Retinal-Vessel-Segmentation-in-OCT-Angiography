"""Single-graph inference utilities for a trained vessel-graph GCN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Union

import torch

from GNN.dataset import graphml_to_pyg_data
from GNN.gcn_model import VesselGCN
from GNN.utils import get_device, load_checkpoint, move_data_to_device

logger = logging.getLogger(__name__)


def predict_graph(
    graphml_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    
    hidden_dim: int = 64,
    output_dim: int = 3,
    dropout: float = 0.5,
) -> Dict[str, object]:
    """Run inference on a single vessel-graph GraphML file.

    Args:
        graphml_path: Path to the ``.graphml`` file to run inference on.
        checkpoint_path: Path to a trained model checkpoint.
        input_dim: Number of input node features (must match the trained
            model).
        hidden_dim: Hidden channel size used at training time.
        output_dim: Number of output classes used at training time.
        dropout: Dropout probability used at training time.

    Returns:
        A dictionary with:
            * ``graph_path``: the input GraphML path.
            * ``node_ids``: original NetworkX node identifiers, in the same
              order as ``predictions``.
            * ``predictions``: predicted class index per node.
            * ``probabilities``: per-class softmax probabilities per node.
    """
    device = get_device()

    data = graphml_to_pyg_data(graphml_path)
    input_dim = data.x.size(1)

    model = VesselGCN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
    ).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    
    node_ids = data.node_ids
    data = move_data_to_device(data, device)
    edge_weight = getattr(data, "edge_weight", None)

    with torch.no_grad():
        logits = model(data.x, data.edge_index, edge_weight)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    return {
        "graph_path": str(graphml_path),
        "node_ids": node_ids,
        "predictions": predictions.cpu().tolist(),
        "probabilities": probabilities.cpu().tolist(),
    }


if __name__ == "__main__":
    result = predict_graph(
        graphml_path="Graph/Graphs/10451.graphml",
        checkpoint_path="Checkpoints/best_model.pth",
        hidden_dim=64,
        output_dim=3,
    )

    print(f"Graph: {result['graph_path']}")
    print(f"Total Nodes: {len(result['node_ids'])}")
    print(f"First 20 Predictions: {result['predictions'][:20]}")

    logger.info("Predictions: %s", result["predictions"])