# Revised biomarkers.py

"""Compute in-memory retinal vessel biomarkers from Graph Builder V3 GraphML."""

from __future__ import annotations

from typing import TypeAlias
import networkx as nx
import numpy as np

from Graph.graph_utils import (
    average_clustering,
    average_degree,
    connected_components,
    get_average_edge_length,
    get_branchpoints,
    get_endpoints,
    get_isolated_nodes,
    get_total_edge_length,
    graph_density,
    graph_transitivity,
    largest_connected_component,
    safe_float,
)

BiomarkerValue: TypeAlias = float | int


def image_area(image_shape: tuple[int, int] | None) -> int:
    if image_shape is None or len(image_shape) != 2:
        return 0
    h, w = image_shape
    if not isinstance(h, (int, np.integer)) or not isinstance(w, (int, np.integer)):
        return 0
    return int(h * w) if h > 0 and w > 0 else 0


def compute_total_vessel_length(graph: nx.Graph) -> float:
    return get_total_edge_length(graph)


def compute_average_vessel_length(graph: nx.Graph) -> float:
    return get_average_edge_length(graph)


def compute_vessel_length_density(
    graph: nx.Graph, image_shape: tuple[int, int] | None
) -> float:
    area = image_area(image_shape)
    return compute_total_vessel_length(graph) / area if area else 0.0


def compute_endpoint_count(graph: nx.Graph) -> int:
    return len(get_endpoints(graph))


def compute_branch_point_count(graph: nx.Graph) -> int:
    return len(get_branchpoints(graph))


def compute_branch_point_density(
    graph: nx.Graph, image_shape: tuple[int, int] | None
) -> float:
    area = image_area(image_shape)
    return compute_branch_point_count(graph) / area if area else 0.0


def compute_connected_component_count(graph: nx.Graph) -> int:
    return len(connected_components(graph))


def compute_largest_component_size(graph: nx.Graph) -> int:
    return largest_connected_component(graph).number_of_nodes()


def _valid_tortuosities(graph: nx.Graph) -> np.ndarray:
    vals = []
    for _, _, a in graph.edges(data=True):
        t = safe_float(a.get("tortuosity"), default=np.nan)
        if np.isfinite(t) and t >= 1.0:
            vals.append(t)
    return np.asarray(vals, dtype=float)


def compute_average_edge_tortuosity(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    v = _valid_tortuosities(graph)
    return float(np.mean(v)) if v.size else 0.0


def compute_maximum_tortuosity(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    v = _valid_tortuosities(graph)
    return float(np.max(v)) if v.size else 0.0


def compute_minimum_tortuosity(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    v = _valid_tortuosities(graph)
    return float(np.min(v)) if v.size else 0.0


def compute_tortuosity_standard_deviation(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    v = _valid_tortuosities(graph)
    return float(np.std(v)) if v.size > 1 else 0.0


def compute_biomarkers(
    graph: nx.Graph, image_shape: tuple[int, int] | None = None
) -> dict[str, BiomarkerValue]:
    return {
        "total_vessel_length": compute_total_vessel_length(graph),
        "average_vessel_length": compute_average_vessel_length(graph),
        "average_edge_length": get_average_edge_length(graph),
        "vessel_length_density": compute_vessel_length_density(graph, image_shape),
        "endpoint_count": compute_endpoint_count(graph),
        "branch_point_count": compute_branch_point_count(graph),
        "branch_point_density": compute_branch_point_density(graph, image_shape),
        "number_of_isolated_nodes": len(get_isolated_nodes(graph)),
        "connected_component_count": compute_connected_component_count(graph),
        "largest_connected_component_size": compute_largest_component_size(graph),
        "average_node_degree": average_degree(graph),
        "graph_density": graph_density(graph),
        "graph_transitivity": graph_transitivity(graph),
        "average_clustering_coefficient": average_clustering(graph),
        "average_edge_tortuosity": compute_average_edge_tortuosity(graph),
        "maximum_tortuosity": compute_maximum_tortuosity(graph),
        "minimum_tortuosity": compute_minimum_tortuosity(graph),
        "tortuosity_standard_deviation": compute_tortuosity_standard_deviation(graph),
    }


__all__ = [
    "compute_biomarkers",
    "compute_total_vessel_length",
    "compute_average_vessel_length",
    "compute_vessel_length_density",
    "compute_endpoint_count",
    "compute_branch_point_count",
    "compute_branch_point_density",
    "compute_connected_component_count",
    "compute_largest_component_size",
    "compute_average_edge_tortuosity",
    "compute_maximum_tortuosity",
    "compute_minimum_tortuosity",
    "compute_tortuosity_standard_deviation",
]
