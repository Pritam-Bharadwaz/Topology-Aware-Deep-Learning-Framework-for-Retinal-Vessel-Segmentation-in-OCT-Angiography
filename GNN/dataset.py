"""Dataset utilities for converting vessel graphs (GraphML) into PyTorch
Geometric ``Data`` objects for graph neural network training and inference.

This module is intentionally decoupled from the ``Graph`` and ``Biomarkers``
packages: it reads GraphML files directly with NetworkX (as required for the
GNN pipeline) and produces ``torch_geometric.data.Data`` objects. It performs
no training, no model definition, and no biomarker computation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# Default values used when an expected node/edge attribute is missing.
DEFAULT_NODE_TYPE = "regular"
DEFAULT_LABEL = 0
NODE_TYPE_TO_INDEX: Dict[str, int] = {"regular": 0, "endpoint": 1, "branch": 2}

# Number of columns produced by build_node_features:
# [degree, is_endpoint, is_branch, is_regular, x, y]
NODE_FEATURE_DIM = 6


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert a value to float, falling back to a default on failure.

    Args:
        value: The value to convert.
        default: The fallback value used when conversion fails.

    Returns:
        The converted float, or ``default`` if conversion is not possible.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _node_type_index(node_data: Dict[str, object]) -> int:
    """Resolve a categorical node-type index from raw node attributes.

    Looks for a ``type`` (or ``node_type``) attribute produced by the vessel
    graph builder (for example ``"endpoint"`` or ``"branch"``). Falls back to
    ``"regular"`` when the attribute is absent or unrecognized.

    Args:
        node_data: Raw NetworkX node attribute dictionary.

    Returns:
        Integer index of the node type.
    """
    raw_type = node_data.get("type", node_data.get("node_type", DEFAULT_NODE_TYPE))
    raw_type = str(raw_type).lower()
    return NODE_TYPE_TO_INDEX.get(raw_type, NODE_TYPE_TO_INDEX[DEFAULT_NODE_TYPE])


def build_node_features(
    graph: nx.Graph, normalize_coordinates: bool = True
) -> np.ndarray:
    """Build a node feature matrix from a vessel graph.

    Each node's feature vector is composed of::

        [degree, is_endpoint, is_branch, is_regular, x, y]

    Coordinates default to ``0.0`` and degree/type flags default to safe
    values whenever the underlying GraphML does not provide them, so the
    function never raises on missing attributes.

    Args:
        graph: A NetworkX graph loaded from a vessel-graph GraphML file.
        normalize_coordinates: If True, min-max normalize x/y coordinates
            across the graph's nodes to the range [0, 1].

    Returns:
        A ``(num_nodes, 6)`` float32 NumPy array of node features, ordered
        to match ``list(graph.nodes())``.
    """
    nodes = list(graph.nodes(data=True))
    degrees = dict(graph.degree())

    raw_coords = np.zeros((len(nodes), 2), dtype=np.float32)
    features = np.zeros((len(nodes), NODE_FEATURE_DIM), dtype=np.float32)

    for i, (node_id, data) in enumerate(nodes):
        degree = float(degrees.get(node_id, 0))
        type_index = _node_type_index(data)

        x = _safe_float(data.get("x", data.get("pos_x", 0.0)))
        y = _safe_float(data.get("y", data.get("pos_y", 0.0)))
        raw_coords[i] = (x, y)

        features[i, 0] = degree
        features[i, 1] = 1.0 if type_index == NODE_TYPE_TO_INDEX["endpoint"] else 0.0
        features[i, 2] = 1.0 if type_index == NODE_TYPE_TO_INDEX["branch"] else 0.0
        features[i, 3] = 1.0 if type_index == NODE_TYPE_TO_INDEX["regular"] else 0.0

    if normalize_coordinates and len(nodes) > 0:
        coord_min = raw_coords.min(axis=0)
        coord_max = raw_coords.max(axis=0)
        coord_range = np.where(coord_max > coord_min, coord_max - coord_min, 1.0)
        normalized = (raw_coords - coord_min) / coord_range
    else:
        normalized = raw_coords

    features[:, 4:6] = normalized
    return features


def build_edge_index_and_weight(
    graph: nx.Graph, node_order: Sequence[object]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build a PyG-style edge index (and optional edge weight) from a graph.

    Both directions of each undirected edge are included, since
    ``torch_geometric.nn.GCNConv`` expects a symmetric ``edge_index`` for
    undirected message passing.

    Args:
        graph: A NetworkX graph loaded from a vessel-graph GraphML file.
        node_order: The ordered sequence of node ids matching row order of
            the node feature matrix.

    Returns:
        A tuple of ``(edge_index, edge_weight)`` where ``edge_index`` has
        shape ``(2, 2 * num_edges)`` and ``edge_weight`` has shape
        ``(2 * num_edges,)`` or is ``None`` if no edge carries a usable
        ``length`` attribute.
    """
    node_to_index = {node_id: i for i, node_id in enumerate(node_order)}
    sources: List[int] = []
    targets: List[int] = []
    weights: List[float] = []
    has_weight = False

    for u, v, data in graph.edges(data=True):
        if u not in node_to_index or v not in node_to_index:
            continue
        ui, vi = node_to_index[u], node_to_index[v]
        length = data.get("length", None)
        weight = _safe_float(length, default=1.0)
        has_weight = has_weight or length is not None

        sources.extend([ui, vi])
        targets.extend([vi, ui])
        weights.extend([weight, weight])

    if not sources:
        return torch.zeros((2, 0), dtype=torch.long), None

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32) if has_weight else None
    return edge_index, edge_weight


def build_node_labels(
    graph: nx.Graph,
    node_order: Sequence[object],
) -> torch.Tensor:
    """Build node labels from the GraphML node_type attribute.

    Mapping:
        regular  -> 0
        endpoint -> 1
        branch   -> 2
    """

    labels = []

    for node_id in node_order:
        node_type = str(
            graph.nodes[node_id].get("node_type", DEFAULT_NODE_TYPE)
        ).lower()

        label = NODE_TYPE_TO_INDEX.get(
            node_type,
            NODE_TYPE_TO_INDEX[DEFAULT_NODE_TYPE],
        )

        labels.append(label)

    return torch.tensor(labels, dtype=torch.long)


def graphml_to_pyg_data(
    filepath: Union[str, Path],
    
    normalize_coordinates: bool = True,
) -> Data:
    """Load a vessel-graph GraphML file and convert it to a PyG ``Data`` object.

    Args:
        filepath: Path to a ``.graphml`` file produced by the Graph package.
        label_key: Node attribute name holding the ground-truth class, used
            for supervised node classification. Missing attributes default
            to class 0 — populate this attribute in your GraphML (or via a
            pre-processing step) before training for meaningful supervision.
        normalize_coordinates: Whether to min-max normalize node coordinate
            features.

    Returns:
        A ``torch_geometric.data.Data`` object with ``x``, ``edge_index``,
        ``edge_weight`` (optional), ``y``, and ``node_ids`` populated.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
        ValueError: If the loaded graph has zero nodes.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GraphML file not found: {filepath}")

    graph = nx.read_graphml(str(filepath))

    if graph.is_directed():
          graph = graph.to_undirected()

    if graph.number_of_nodes() == 0:
        raise ValueError(f"Graph loaded from {filepath} has no nodes.")

    node_order = list(graph.nodes())
    x = torch.from_numpy(
        build_node_features(graph, normalize_coordinates=normalize_coordinates)
    )
    edge_index, edge_weight = build_edge_index_and_weight(graph, node_order)
    y = build_node_labels(
        graph,
        node_order,
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    if edge_weight is not None:
        data.edge_weight = edge_weight
    data.node_ids = node_order
    data.graph_path = str(filepath)
    data.graph_name = filepath.stem
    return data


class VesselGraphDataset(TorchDataset):
    """Dataset over a collection of vessel-graph GraphML files.

    Each item is one graph converted to a ``torch_geometric.data.Data``
    object. Compatible with ``torch_geometric.loader.DataLoader`` for
    mini-batching multiple graphs. Supports an arbitrary number of GraphML
    files, discovered either explicitly via ``graph_paths`` or by scanning
    ``root_dir`` for ``*.graphml`` files.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        graph_paths: Optional[Sequence[Union[str, Path]]] = None,
        
        normalize_coordinates: bool = True,
        cache_in_memory: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            root_dir: Root directory scanned for ``*.graphml`` files when
                ``graph_paths`` is not provided.
            graph_paths: Optional explicit list of GraphML file paths. When
                omitted, all ``*.graphml`` files under ``root_dir`` are used.
            label_key: Node attribute name holding ground-truth labels.
            normalize_coordinates: Whether to min-max normalize coordinates.
            cache_in_memory: If True, converted ``Data`` objects are cached
                after first access to avoid re-parsing GraphML on every
                epoch. Disable for very large datasets that do not fit in
                memory.
        """
        self.root_dir = Path(root_dir)
        
        self.normalize_coordinates = normalize_coordinates
        self.cache_in_memory = cache_in_memory
        self._cache: Dict[int, Data] = {}

        if graph_paths is not None:
            self._graph_paths: List[Path] = [Path(p) for p in graph_paths]
        else:
            self._graph_paths = sorted(self.root_dir.glob("*.graphml"))

        if not self._graph_paths:
            logger.warning(
                "No GraphML files found for VesselGraphDataset at %s", self.root_dir
            )

    def __len__(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self._graph_paths)

    def __getitem__(self, idx: int) -> Data:
        """Load and convert the graph at ``idx`` to a PyG ``Data`` object.

        Args:
            idx: Index of the graph to retrieve.

        Returns:
            The corresponding ``Data`` object.
        """
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        data = graphml_to_pyg_data(
            self._graph_paths[idx],
            
            normalize_coordinates=self.normalize_coordinates,
        )
        if self.cache_in_memory:
            self._cache[idx] = data
        return data

    @property
    def graph_paths(self) -> List[Path]:
        """Return the list of GraphML file paths backing this dataset."""
        return list(self._graph_paths)