"""Compute graph-level statistics for OCTA retinal vessel GraphML files.

This module only calculates and exports statistics. It does not build or
visualize graphs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

NUMERIC_COLUMNS: tuple[str, ...] = (
    "number_of_nodes", "number_of_edges", "number_of_connected_components",
    "number_of_endpoints", "number_of_branch_points", "number_of_isolated_nodes",
    "average_node_degree", "maximum_node_degree", "minimum_node_degree",
    "average_edge_length", "maximum_edge_length", "minimum_edge_length",
    "total_vessel_length", "graph_density", "average_clustering_coefficient",
    "graph_transitivity", "diameter", "average_shortest_path_length",
    "largest_connected_component_size", "number_of_small_connected_components",
    "endpoint_to_branch_ratio", "branch_point_density",
    "average_vessel_segment_length", "number_of_vessel_segments",
)
CSV_COLUMNS: tuple[str, ...] = ("filename", *NUMERIC_COLUMNS, "processing_status", "error")


def load_graph(graph_path: Path) -> tuple[nx.Graph | None, str | None]:
    """Load one GraphML file without stopping the batch on parsing errors.

    Args:
        graph_path: Path to the source GraphML file.

    Returns:
        A loaded undirected graph and ``None`` on success, or ``None`` and a
        descriptive error message when the file cannot be parsed.
    """
    try:
        graph = nx.read_graphml(graph_path)
        return (graph.to_undirected() if graph.is_directed() else graph), None
    except Exception as error:  # XML parsers raise multiple exception types.
        return None, f"{type(error).__name__}: {error}"


def compute_basic_statistics(graph: nx.Graph) -> dict[str, float]:
    """Calculate basic counts and endpoint/branch classifications.

    Args:
        graph: Input vessel graph.

    Returns:
        Basic count metrics for one graph.
    """
    return {
        "number_of_nodes": float(graph.number_of_nodes()),
        "number_of_edges": float(graph.number_of_edges()),
        "number_of_connected_components": float(nx.number_connected_components(graph)),
        "number_of_endpoints": float(sum(d.get("node_type") == "endpoint" for _, d in graph.nodes(data=True))),
        "number_of_branch_points": float(sum(d.get("node_type") == "branch" for _, d in graph.nodes(data=True))),
        "number_of_isolated_nodes": float(sum(degree == 0 for _, degree in graph.degree())),
    }


def compute_node_statistics(graph: nx.Graph) -> dict[str, float]:
    """Calculate degree statistics.

    Args:
        graph: Input vessel graph.

    Returns:
        Mean, maximum, and minimum node degree. Empty graphs return zeroes.
    """
    degrees = np.fromiter((degree for _, degree in graph.degree()), dtype=float)
    if degrees.size == 0:
        return {"average_node_degree": 0.0, "maximum_node_degree": 0.0, "minimum_node_degree": 0.0}
    return {
        "average_node_degree": float(np.mean(degrees)),
        "maximum_node_degree": float(np.max(degrees)),
        "minimum_node_degree": float(np.min(degrees)),
    }


def extract_edge_lengths(graph: nx.Graph) -> np.ndarray:
    """Extract finite numeric V3 ``length`` edge attributes.

    Args:
        graph: Input vessel graph.

    Returns:
        Array of usable edge lengths in pixels.
    """
    lengths: list[float] = []
    for _, _, attributes in graph.edges(data=True):
        try:
            length = float(attributes["length"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(length):
            lengths.append(length)
    return np.asarray(lengths, dtype=float)


def compute_edge_statistics(graph: nx.Graph) -> dict[str, float]:
    """Calculate vessel edge-length statistics.

    Args:
        graph: Input vessel graph.

    Returns:
        Mean, maximum, minimum, and total length of valid vessel segments.
    """
    lengths = extract_edge_lengths(graph)
    if lengths.size == 0:
        return {"average_edge_length": 0.0, "maximum_edge_length": 0.0, "minimum_edge_length": 0.0, "total_vessel_length": 0.0}
    return {
        "average_edge_length": float(np.mean(lengths)),
        "maximum_edge_length": float(np.max(lengths)),
        "minimum_edge_length": float(np.min(lengths)),
        "total_vessel_length": float(np.sum(lengths)),
    }


def compute_topology_statistics(graph: nx.Graph) -> dict[str, float]:
    """Calculate density and triangle-based topology statistics.

    Args:
        graph: Input vessel graph.

    Returns:
        Density, average clustering coefficient, and transitivity.
    """
    if graph.number_of_nodes() == 0:
        return {"graph_density": 0.0, "average_clustering_coefficient": 0.0, "graph_transitivity": 0.0}
    return {
        "graph_density": float(nx.density(graph)),
        "average_clustering_coefficient": float(nx.average_clustering(graph)),
        "graph_transitivity": float(nx.transitivity(graph)),
    }


def compute_connectivity_statistics(graph: nx.Graph) -> dict[str, float]:
    """Calculate path metrics on a connected graph or its largest component.

    Args:
        graph: Input vessel graph.

    Returns:
        Diameter and average shortest-path length. Empty graphs use NaN.
    """
    if graph.number_of_nodes() == 0:
        return {"diameter": np.nan, "average_shortest_path_length": np.nan}
    largest = max(nx.connected_components(graph), key=len)
    component = graph.subgraph(largest)
    if component.number_of_nodes() == 1:
        return {"diameter": 0.0, "average_shortest_path_length": 0.0}
    return {
        "diameter": float(nx.diameter(component)),
        "average_shortest_path_length": float(nx.average_shortest_path_length(component)),
    }


def compute_component_statistics(graph: nx.Graph) -> dict[str, float]:
    """Calculate connected-component size metrics.

    Args:
        graph: Input vessel graph.

    Returns:
        Largest component size and count of components with fewer than five nodes.
    """
    sizes = [len(component) for component in nx.connected_components(graph)]
    return {
        "largest_connected_component_size": float(max(sizes, default=0)),
        "number_of_small_connected_components": float(sum(size < 5 for size in sizes)),
    }


def compute_research_statistics(basic: dict[str, float], edges: dict[str, float]) -> dict[str, float]:
    """Calculate OCTA vessel-network research metrics.

    Branch-point density is the proportion of graph nodes that are branch
    points, because Graph Builder V3 does not store image-area metadata.

    Args:
        basic: Basic count metrics.
        edges: Edge-length metrics.

    Returns:
        Endpoint/branch ratio, branch density, and segment metrics.
    """
    endpoints = basic["number_of_endpoints"]
    branches = basic["number_of_branch_points"]
    nodes = basic["number_of_nodes"]
    return {
        "endpoint_to_branch_ratio": float(endpoints / branches) if branches else np.nan,
        "branch_point_density": float(branches / nodes) if nodes else 0.0,
        "average_vessel_segment_length": edges["average_edge_length"],
        "number_of_vessel_segments": basic["number_of_edges"],
    }


def compute_graph_statistics(graph: nx.Graph, filename: str) -> dict[str, Any]:
    """Combine all metric groups into one CSV-ready graph record.

    Args:
        graph: Input vessel graph.
        filename: Name of the source GraphML file.

    Returns:
        One complete statistics record.
    """
    basic = compute_basic_statistics(graph)
    edges = compute_edge_statistics(graph)
    record: dict[str, Any] = {"filename": filename}
    record.update(basic)
    record.update(compute_node_statistics(graph))
    record.update(edges)
    record.update(compute_topology_statistics(graph))
    record.update(compute_connectivity_statistics(graph))
    record.update(compute_component_statistics(graph))
    record.update(compute_research_statistics(basic, edges))
    record.update(processing_status="ok", error="")
    return record


def failed_record(filename: str, error: str) -> dict[str, Any]:
    """Create a row for an invalid GraphML input.

    Args:
        filename: Name of the invalid source file.
        error: Explanation of the loading failure.

    Returns:
        Record with missing numerical values and failure status.
    """
    record: dict[str, Any] = {"filename": filename, **{column: np.nan for column in NUMERIC_COLUMNS}}
    record.update(processing_status="failed", error=error)
    return record


def save_csv(records: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    """Save per-graph records to the required CSV file.

    Args:
        records: Graph-statistics records.
        output_path: Destination CSV path.

    Returns:
        The saved dataframe.
    """
    dataframe = pd.DataFrame(records, columns=CSV_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return dataframe


def save_summary_json(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save mean, sample standard deviation, minimum, and maximum per metric.

    Args:
        dataframe: Per-graph statistics dataframe.
        output_path: Destination JSON path.
    """
    successful = dataframe[dataframe["processing_status"] == "ok"]
    summary: dict[str, dict[str, float | None]] = {}
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(successful[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary[column] = {"mean": None, "std": None, "minimum": None, "maximum": None}
        else:
            summary[column] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)


def process_graphs(input_directory: Path, output_directory: Path) -> int:
    """Process all GraphML inputs and write the CSV and JSON outputs.

    Args:
        input_directory: Directory containing input GraphML files.
        output_directory: Directory for the two statistics artifacts.

    Returns:
        Number of GraphML files attempted.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "graph_statistics.csv"
    summary_path = output_directory / "summary_statistics.json"
    graph_paths = sorted(input_directory.glob("*.graphml")) if input_directory.is_dir() else []
    if not input_directory.is_dir():
        print(f"Input directory not found: {input_directory}")

    records: list[dict[str, Any]] = []
    for graph_path in graph_paths:
        graph, error = load_graph(graph_path)
        if graph is None:
            print(f"Skipping invalid GraphML: {graph_path.name} ({error})")
            records.append(failed_record(graph_path.name, error or "Unknown loading error"))
        else:
            records.append(compute_graph_statistics(graph, graph_path.name))

    dataframe = save_csv(records, csv_path)
    save_summary_json(dataframe, summary_path)
    print(f"Processed {len(graph_paths)} graphs")
    print(f"CSV saved to {csv_path}")
    print(f"Summary JSON saved to {summary_path}")
    return len(graph_paths)


def parse_arguments() -> argparse.Namespace:
    """Parse batch input and output directories.

    Returns:
        Parsed command-line arguments.
    """
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Compute OCTA vessel graph statistics")
    parser.add_argument("--input-directory", type=Path, default=project_root / "Graph" / "Graphs", help="Directory containing GraphML files.")
    parser.add_argument("--output-directory", type=Path, default=project_root / "Graph" / "Statistics", help="Directory for CSV and JSON outputs.")
    return parser.parse_args()


def main() -> None:
    """Run graph-level statistics computation."""
    arguments = parse_arguments()
    process_graphs(arguments.input_directory, arguments.output_directory)


if __name__ == "__main__":
    main()
