"""Reusable OCTA vessel graph helper functions.

This module provides small, independent helpers for GraphML I/O, graph access,
validation, and geometry. It intentionally contains no graph construction,
visualization, report generation, CSV output, command-line interface, or
hard-coded project paths.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import networkx as nx
import numpy as np

NodeId: TypeAlias = Hashable
Point: TypeAlias = Sequence[float]
VALID_NODE_TYPES = {"endpoint", "branch"}


def load_graph(graph_path: Path) -> nx.Graph:
    """Load a GraphML graph.

    Args:
        graph_path: Path to an existing GraphML file.

    Returns:
        The graph represented by the GraphML file.

    Raises:
        networkx.NetworkXError: If the GraphML content cannot be parsed.
        OSError: If the source file cannot be read.
    """
    graph = nx.read_graphml(graph_path)
    return graph.to_undirected() if graph.is_directed() else graph


def save_graph(graph: nx.Graph, graph_path: Path) -> None:
    """Save a graph as GraphML, creating its destination directory if needed.

    Args:
        graph: Graph to serialize.
        graph_path: Destination GraphML path.
    """

    if graph_path.suffix.lower() != ".graphml":
        raise ValueError("Output file must use a .graphml extension.")

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, graph_path, named_key_ids=True)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float without propagating conversion errors.

    Args:
        value: Value to convert.
        default: Value returned for missing, invalid, or non-finite input.

    Returns:
        A finite float or ``default``.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert a finite numeric value to an integer safely.

    Args:
        value: Value to convert.
        default: Value returned for missing, invalid, or non-finite input.

    Returns:
        An integer or ``default``.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return int(result) if np.isfinite(result) else default


def get_node_position(graph: nx.Graph, node: NodeId) -> tuple[float, float]:
    """Return a node's V3 image coordinates as ``(x, y)``.

    Args:
        graph: Graph containing the node.
        node: Node identifier.

    Returns:
        The node's finite x and y coordinates.

    Raises:
        KeyError: If the node or either coordinate attribute is unavailable.
        ValueError: If either coordinate is not finite numeric data.
    """
    attributes = graph.nodes[node]
    if "x" not in attributes or "y" not in attributes:
        raise KeyError(f"Node {node!r} must contain 'x' and 'y' attributes")
    x, y = float(attributes["x"]), float(attributes["y"])
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError(f"Node {node!r} has non-finite coordinates")
    return x, y


def get_node_degree(graph: nx.Graph, node: NodeId) -> int:
    """Return the graph-theoretic degree of a node.

    Args:
        graph: Graph containing the node.
        node: Node identifier.

    Returns:
        Node degree.
    """
    return int(graph.degree(node))


def get_endpoints(graph: nx.Graph) -> list[NodeId]:
    """Return nodes classified as endpoints by Graph Builder V3.

    Args:
        graph: Input graph.

    Returns:
        Endpoint node identifiers in graph iteration order.
    """
    return [
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == "endpoint"
    ]


def get_branchpoints(graph: nx.Graph) -> list[NodeId]:
    """Return nodes classified as branch points by Graph Builder V3.

    Args:
        graph: Input graph.

    Returns:
        Branch-point node identifiers in graph iteration order.
    """
    return [
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == "branch"
    ]


def get_isolated_nodes(graph: nx.Graph) -> list[NodeId]:
    """Return all nodes having zero graph-theoretic degree.

    Args:
        graph: Input graph.

    Returns:
        Isolated node identifiers.
    """
    return [node for node, degree in graph.degree() if degree == 0]


def _edge_attributes(
    graph: nx.Graph, source: NodeId, target: NodeId, key: NodeId | None = None
) -> dict[str, Any]:
    """Return one edge attribute dictionary, checking multigraph ambiguity."""
    if graph.is_multigraph():
        edge_data = graph.get_edge_data(source, target)
        if edge_data is None:
            raise nx.NetworkXError(f"Edge ({source!r}, {target!r}) does not exist")
        if key is None:
            if len(edge_data) != 1:
                raise ValueError(
                    "A multigraph edge key is required when parallel edges exist"
                )
            return next(iter(edge_data.values()))
        return edge_data[key]
    return graph.edges[source, target]


def get_edge_length(
    graph: nx.Graph, source: NodeId, target: NodeId, key: NodeId | None = None
) -> float:
    """Return the non-negative V3 ``length`` attribute of one edge.

    Args:
        graph: Input graph.
        source: Source node identifier.
        target: Target node identifier.
        key: Edge key for a multigraph with parallel edges.

    Returns:
        Edge length in pixels.

    Raises:
        KeyError: If the edge has no ``length`` attribute.
        ValueError: If the length is invalid or negative.
    """
    attributes = _edge_attributes(graph, source, target, key)
    if "length" not in attributes:
        raise KeyError(f"Edge ({source!r}, {target!r}) has no 'length' attribute")
    length = float(attributes["length"])
    if not np.isfinite(length) or length < 0:
        raise ValueError(f"Edge ({source!r}, {target!r}) has invalid length {length!r}")
    return length


def _all_edge_lengths(graph: nx.Graph) -> np.ndarray:
    """Extract all valid length attributes, ignoring malformed optional edges."""
    values: list[float] = []
    for _, _, attributes in graph.edges(data=True):
        value = safe_float(attributes.get("length"), default=np.nan)
        if np.isfinite(value) and value >= 0:
            values.append(value)
    return np.asarray(values, dtype=float)


def get_total_edge_length(graph: nx.Graph) -> float:
    """Return the sum of all valid V3 edge lengths.

    Args:
        graph: Input graph.

    Returns:
        Total vessel length in pixels, or zero when no valid lengths exist.
    """
    return float(np.sum(_all_edge_lengths(graph)))


def get_average_edge_length(graph: nx.Graph) -> float:
    """Return the mean of all valid V3 edge lengths.

    Args:
        graph: Input graph.

    Returns:
        Mean vessel segment length in pixels, or zero when unavailable.
    """
    lengths = _all_edge_lengths(graph)
    return float(np.mean(lengths)) if lengths.size else 0.0


def _undirected_simple_graph(graph: nx.Graph) -> nx.Graph:
    """Convert a graph to a simple undirected representation for topology APIs."""
    undirected = graph.to_undirected(as_view=True) if graph.is_directed() else graph
    return nx.Graph(undirected) if undirected.is_multigraph() else undirected


def connected_components(graph: nx.Graph) -> list[set[NodeId]]:
    """Return connected components as node sets.

    Directed graphs are treated as undirected for vessel-network connectivity.

    Args:
        graph: Input graph.

    Returns:
        List of connected-component node sets.
    """
    return [
        set(component)
        for component in nx.connected_components(_undirected_simple_graph(graph))
    ]


def largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return a copied subgraph containing the largest connected component.

    Args:
        graph: Input graph.

    Returns:
        Independent graph copy of the largest component, or an empty graph copy.
    """
    components = connected_components(graph)
    largest = max(components, key=len, default=set())
    return graph.subgraph(largest).copy()


def graph_density(graph: nx.Graph) -> float:
    """Return graph density, with zero for empty graphs.

    Args:
        graph: Input graph.

    Returns:
        Graph density.
    """
    simple_graph = _undirected_simple_graph(graph)
    return float(nx.density(simple_graph)) if simple_graph.number_of_nodes() else 0.0


def graph_transitivity(graph: nx.Graph) -> float:
    """Return global transitivity, with zero for empty graphs.

    Args:
        graph: Input graph.

    Returns:
        Graph transitivity.
    """
    simple_graph = _undirected_simple_graph(graph)
    return (
        float(nx.transitivity(simple_graph)) if simple_graph.number_of_nodes() else 0.0
    )


def average_clustering(graph: nx.Graph) -> float:
    """Return average clustering coefficient, with zero for empty graphs.

    Args:
        graph: Input graph.

    Returns:
        Average clustering coefficient.
    """
    simple_graph = _undirected_simple_graph(graph)
    return (
        float(nx.average_clustering(simple_graph))
        if simple_graph.number_of_nodes()
        else 0.0
    )


def average_degree(graph: nx.Graph) -> float:
    """Return the mean graph-theoretic degree.

    Args:
        graph: Input graph.

    Returns:
        Average node degree, or zero for an empty graph.
    """
    if graph.number_of_nodes() == 0:
        return 0.0
    return float(np.mean([degree for _, degree in graph.degree()]))


def validate_node_attributes(graph: nx.Graph) -> list[str]:
    """Validate required Graph Builder V3 node attributes.

    Args:
        graph: Input graph.

    Returns:
        Human-readable validation errors; an empty list means valid nodes.
    """
    errors: list[str] = []
    for node, attributes in graph.nodes(data=True):
        for name in ("x", "y", "degree", "node_type"):
            if name not in attributes:
                errors.append(f"Node {node!r} is missing required '{name}' attribute")
        if "x" in attributes and "y" in attributes:
            try:
                get_node_position(graph, node)
            except (KeyError, TypeError, ValueError) as error:
                errors.append(str(error))
        if "degree" in attributes and safe_int(attributes["degree"], default=-1) < 0:
            errors.append(f"Node {node!r} has invalid 'degree' attribute")
        if (
            "node_type" in attributes
            and attributes["node_type"] not in VALID_NODE_TYPES
        ):
            errors.append(f"Node {node!r} has invalid 'node_type' attribute")
    return errors


def validate_edge_attributes(graph: nx.Graph) -> list[str]:
    """Validate required Graph Builder V3 edge attributes.

    Args:
        graph: Input graph.

    Returns:
        Human-readable validation errors; an empty list means valid edges.
    """
    errors: list[str] = []
    required = ("length", "path_coordinates", "euclidean_distance", "tortuosity")
    for source, target, attributes in graph.edges(data=True):
        for name in required:
            if name not in attributes:
                errors.append(
                    f"Edge ({source!r}, {target!r}) is missing required '{name}' attribute"
                )
        for name in ("length", "euclidean_distance", "tortuosity"):
            if name in attributes:
                value = safe_float(attributes[name], default=np.nan)

                if not np.isfinite(value) or value < 0:
                    errors.append(
                        f"Edge ({source!r}, {target!r}) has invalid '{name}' attribute"
                    )
        if (
            "path_coordinates" in attributes
            and not str(attributes["path_coordinates"]).strip()
        ):
            errors.append(
                f"Edge ({source!r}, {target!r}) has an empty 'path_coordinates' attribute"
            )
    return errors


def validate_graph(graph: nx.Graph) -> tuple[bool, list[str]]:
    """Validate Graph Builder V3 structural attributes.

    Args:
        graph: Input graph.

    Returns:
        A validity flag and all discovered validation errors.
    """
    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        return False, ["Object is not a NetworkX graph"]
    errors = validate_node_attributes(graph) + validate_edge_attributes(graph)
    return not errors, errors


def check_empty_graph(graph: nx.Graph) -> bool:
    """Return whether a graph has no nodes.

    Args:
        graph: Input graph.

    Returns:
        ``True`` when the graph contains zero nodes.
    """
    return graph.number_of_nodes() == 0


def euclidean_distance(point_a: Point, point_b: Point) -> float:
    """Calculate Euclidean distance between two planar ``(x, y)`` points.

    Args:
        point_a: First point.
        point_b: Second point.

    Returns:
        Euclidean distance.

    Raises:
        ValueError: If either point is not a finite two-coordinate point.
    """
    first = np.asarray(point_a, dtype=float)
    second = np.asarray(point_b, dtype=float)
    if (
        first.shape != (2,)
        or second.shape != (2,)
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("Both points must contain exactly two finite coordinates")
    return float(np.linalg.norm(second - first))


def path_length(path: Iterable[Point]) -> float:
    """Calculate the cumulative Euclidean length of an ordered coordinate path.

    Args:
        path: Iterable of planar ``(x, y)`` coordinates.

    Returns:
        Sum of consecutive Euclidean distances, or zero for fewer than two points.
    """
    points = list(path)
    return float(
        sum(
            euclidean_distance(first, second)
            for first, second in zip(points, points[1:])
        )
    )


def graph_summary(graph: nx.Graph) -> dict[str, float | int]:
    """Return compact in-memory graph metadata for callers and logs.

    Args:
        graph: Input graph.

    Returns:
        Basic reusable graph summary values. No file output is performed.
    """
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "connected_components": len(connected_components(graph)),
        "endpoints": len(get_endpoints(graph)),
        "branchpoints": len(get_branchpoints(graph)),
        "isolated_nodes": len(get_isolated_nodes(graph)),
        "total_edge_length": get_total_edge_length(graph),
        "average_edge_length": get_average_edge_length(graph),
        "average_degree": average_degree(graph),
        "density": graph_density(graph),
        "transitivity": graph_transitivity(graph),
        "average_clustering": average_clustering(graph),
    }
