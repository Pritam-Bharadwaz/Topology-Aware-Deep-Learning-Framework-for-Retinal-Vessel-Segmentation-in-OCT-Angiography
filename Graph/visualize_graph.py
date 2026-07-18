"""Render Graph Builder V3 GraphML vessel graphs over their OCTA vessel masks.

This module intentionally does not build graphs or export statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np


def load_graph(graph_path: Path) -> nx.Graph:
    """Read one GraphML vessel graph."""
    try:
        return nx.read_graphml(graph_path)
    except (OSError, nx.NetworkXError) as error:
        raise RuntimeError(f"Could not load GraphML file: {graph_path}") from error


def load_vessel_image(image_path: Path) -> np.ndarray:
    """Load the corresponding binary OCTA vessel mask as a grayscale image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read corresponding vessel image: {image_path}")
    return image


def edge_path(graph: nx.Graph, source: str, target: str, data: dict[str, object]) -> np.ndarray:
    """Read V3 ordered path coordinates, with a direct-line fallback."""
    raw_path = data.get("path_coordinates")
    if raw_path is not None:
        try:
            coordinates = np.asarray(json.loads(str(raw_path)), dtype=float)
            if coordinates.ndim == 2 and coordinates.shape[1] == 2:
                return coordinates
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    return np.asarray(
        [
            [float(graph.nodes[source]["x"]), float(graph.nodes[source]["y"])],
            [float(graph.nodes[target]["x"]), float(graph.nodes[target]["y"])],
        ],
        dtype=float,
    )


def positions_for_type(graph: nx.Graph, node_type: str) -> np.ndarray:
    """Return all ``(x, y)`` node positions having ``node_type``."""
    positions = [
        (float(data["x"]), float(data["y"]))
        for _, data in graph.nodes(data=True)
        if data.get("node_type") == node_type
    ]
    return np.asarray(positions, dtype=float) if positions else np.empty((0, 2))


def plot_graph(
    graph: nx.Graph,
    vessel_image: np.ndarray,
    output_path: Path,
    title: str,
    dpi: int,
    show_nodes: bool,
    show_node_labels: bool,
    image_alpha: float,
    graph_alpha: float,
    publication: bool,
) -> None:
    """Overlay a V3 graph on its vessel mask and save a PNG image."""
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=not publication)
    height, width = vessel_image.shape[:2]
    axis.imshow(vessel_image, cmap="gray", origin="upper", alpha=image_alpha)
    axis.set_aspect("equal", adjustable="box")

    segments = [
        edge_path(graph, source, target, data)
        for source, target, data in graph.edges(data=True)
    ]
    if segments:
        axis.add_collection(
            LineCollection(segments, colors="blue", linewidths=0.8, alpha=graph_alpha)
        )

    if show_nodes:
        endpoints = positions_for_type(graph, "endpoint")
        branches = positions_for_type(graph, "branch")
        if len(endpoints):
            axis.scatter(
                endpoints[:, 0], endpoints[:, 1], s=20, c="green", marker="o",
                label="Endpoint", alpha=graph_alpha, zorder=3,
            )
        if len(branches):
            axis.scatter(
                branches[:, 0], branches[:, 1], s=24, c="red", marker="s",
                label="Branch point", alpha=graph_alpha, zorder=3,
            )
        if (len(endpoints) or len(branches)) and not publication:
            axis.legend(loc="upper right", fontsize=8)

    if show_node_labels:
        for node_id, data in graph.nodes(data=True):
            axis.annotate(
                str(data.get("id", node_id)),
                (float(data["x"]), float(data["y"])),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5,
            )

    # Use the full source image extent so graph coordinates remain aligned.
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    if publication:
        axis.set_axis_off()
    else:
        axis.set_title(title)
        axis.set_xlabel("x (pixels)")
        axis.set_ylabel("y (pixels)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(figure)


def visualise_directory(
    input_directory: Path,
    image_directory: Path,
    output_directory: Path,
    dpi: int,
    show_nodes: bool,
    show_node_labels: bool,
    image_alpha: float,
    graph_alpha: float,
    publication: bool,
) -> None:
    """Overlay every GraphML graph on its same-stem BMP vessel image."""
    if not input_directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")
    if not image_directory.is_dir():
        raise FileNotFoundError(f"Vessel-image directory does not exist: {image_directory}")

    graph_paths = sorted(input_directory.glob("*.graphml"))
    if not graph_paths:
        print(f"No GraphML files found in: {input_directory}")
        return

    for graph_path in graph_paths:
        image_path = image_directory / f"{graph_path.stem}.bmp"
        output_path = output_directory / f"{graph_path.stem}.png"
        plot_graph(
            graph=load_graph(graph_path),
            vessel_image=load_vessel_image(image_path),
            output_path=output_path,
            title=graph_path.stem,
            dpi=dpi,
            show_nodes=show_nodes,
            show_node_labels=show_node_labels,
            image_alpha=image_alpha,
            graph_alpha=graph_alpha,
            publication=publication,
        )
        print(f"Saved: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line batch visualization options."""
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="OCTA vessel graph visualizer")
    parser.add_argument("--input-directory", type=Path, default=project_root / "Graph" / "Graphs", help="Directory containing .graphml files.")
    parser.add_argument("--image-directory", type=Path, default=project_root / "Dataset" / "OCTA-500" / "Output_clDice", help="Directory containing corresponding vessel .bmp images.")
    parser.add_argument("--output-directory", type=Path, default=project_root / "Graph" / "Visualizations", help="Directory for PNG visualizations.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300).")
    parser.add_argument("--hide-nodes", action="store_true", help="Hide endpoint and branch-point markers.")
    parser.add_argument("--show-node-labels", action="store_true", help="Show GraphML node IDs.")
    parser.add_argument("--image-alpha", type=float, default=0.65, help="Vessel image opacity from 0 to 1 (default: 0.65).")
    parser.add_argument("--graph-alpha", type=float, default=0.85, help="Graph overlay opacity from 0 to 1 (default: 0.85).")
    parser.add_argument("--publication", action="store_true", help="Remove axes, ticks, labels, title, and legend.")
    return parser.parse_args()


def main() -> None:
    """Run batch graph visualization."""
    arguments = parse_arguments()
    if arguments.dpi < 1:
        raise ValueError("dpi must be a positive integer")
    if not 0.0 <= arguments.image_alpha <= 1.0:
        raise ValueError("image_alpha must be between 0 and 1")
    if not 0.0 <= arguments.graph_alpha <= 1.0:
        raise ValueError("graph_alpha must be between 0 and 1")
    visualise_directory(
        arguments.input_directory,
        arguments.image_directory,
        arguments.output_directory,
        arguments.dpi,
        not arguments.hide_nodes,
        arguments.show_node_labels,
        arguments.image_alpha,
        arguments.graph_alpha,
        arguments.publication,
    )


if __name__ == "__main__":
    main()
