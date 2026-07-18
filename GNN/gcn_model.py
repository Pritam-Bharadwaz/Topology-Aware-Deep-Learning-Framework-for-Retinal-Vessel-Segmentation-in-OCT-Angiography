"""Graph Convolutional Network architecture for vessel-graph node
classification.

Contains only the model definition. Training, evaluation, and data loading
logic live in their own modules.
"""

from __future__ import annotations

from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv


class VesselGCN(nn.Module):
    """A two-layer Graph Convolutional Network for vessel-graph node classification.

    Architecture::

        Input -> GCNConv -> ReLU -> Dropout -> GCNConv -> ReLU -> Linear -> Output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.5,
    ) -> None:
        """Initialize the model.

        Args:
            input_dim: Number of input node features.
            hidden_dim: Number of hidden channels in each GCN layer.
            output_dim: Number of output classes.
            dropout: Dropout probability applied after the first GCN layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Run a forward pass.

        Args:
            x: Node feature matrix of shape ``(num_nodes, input_dim)``.
            edge_index: Graph connectivity of shape ``(2, num_edges)``.
            edge_weight: Optional edge weights of shape ``(num_edges,)``.

        Returns:
            Raw (pre-softmax) class logits of shape ``(num_nodes, output_dim)``.
        """
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_weight)
        h = F.relu(h)
        out = self.linear(h)
        return out