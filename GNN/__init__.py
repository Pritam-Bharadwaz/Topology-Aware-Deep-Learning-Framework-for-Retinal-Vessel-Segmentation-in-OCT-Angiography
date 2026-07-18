"""Graph Neural Network module for vessel-graph node classification.

Exposes the primary dataset, model, training, evaluation, and inference
entry points for downstream use in the Topology_OCTA pipeline.
"""

from GNN.dataset import VesselGraphDataset, graphml_to_pyg_data
from GNN.evaluate import evaluate_from_checkpoint, evaluate_model
from GNN.gcn_model import VesselGCN
from GNN.predict import predict_graph
from GNN.train_gnn import train

__all__ = [
    "VesselGraphDataset",
    "graphml_to_pyg_data",
    "VesselGCN",
    "evaluate_from_checkpoint",
    "evaluate_model",
    "predict_graph",
    "train",
]