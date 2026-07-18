"""Graph Builder V3: convert OCTA binary vessel masks to GraphML graphs.

This module deliberately excludes visualization and CSV statistics export.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

# Internal pixels use (row, column); exported coordinates use (x=column, y=row).
Pixel = tuple[int, int]
NEIGHBOUR_OFFSETS: tuple[Pixel, ...] = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
)


@dataclass(frozen=True)
class GraphBuildResult:
    """Graph and requested counts produced for one vessel mask."""

    graph: nx.Graph
    endpoint_count: int
    branch_point_count: int
    skeleton_component_count: int


def load_binary_mask(image_path: Path) -> np.ndarray:
    """Load a BMP mask as a boolean foreground image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image > 0


def remove_tiny_components(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    """Remove 8-connected vessel components smaller than the given size."""
    if min_component_size < 1:
        raise ValueError("min_component_size must be at least 1")

    labels, component_count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_component_size
    keep[0] = False  # Never retain the background label.
    return keep[labels]


def skeleton_neighbours(pixel: Pixel, skeleton: np.ndarray) -> list[Pixel]:
    """Return all in-bounds 8-connected skeleton neighbours of ``pixel``."""
    row, column = pixel
    height, width = skeleton.shape
    neighbours: list[Pixel] = []
    for dr, dc in NEIGHBOUR_OFFSETS:
        next_row, next_column = row + dr, column + dc
        if 0 <= next_row < height and 0 <= next_column < width:
            if skeleton[next_row, next_column]:
                neighbours.append((next_row, next_column))
    return neighbours


def count_skeleton_neighbours(skeleton: np.ndarray) -> np.ndarray:
    """Explicitly count 8-neighbours; no convolution threshold is used."""
    counts = np.zeros(skeleton.shape, dtype=np.uint8)
    for row, column in np.argwhere(skeleton):
        counts[row, column] = len(skeleton_neighbours((int(row), int(column)), skeleton))
    return counts


def detect_graph_nodes(neighbour_counts: np.ndarray) -> tuple[set[Pixel], set[Pixel]]:
    """Detect endpoints (one neighbour) and branch points (three or more)."""
    endpoints = {tuple(map(int, p)) for p in np.argwhere(neighbour_counts == 1)}
    branches = {tuple(map(int, p)) for p in np.argwhere(neighbour_counts >= 3)}
    return endpoints, branches


def trace_edge_dfs(
    start: Pixel,
    first_pixel: Pixel,
    skeleton: np.ndarray,
    node_pixels: set[Pixel],
) -> list[Pixel] | None:
    """Trace one node-to-node vessel segment with iterative DFS.

    A local visited set prevents a malformed skeleton loop from causing an
    infinite search. The caller maintains global visited pixels for accepted
    edges, so the same vessel segment is never traced twice.
    """
    if first_pixel in node_pixels:
        return [start, first_pixel]

    stack: list[tuple[Pixel, list[Pixel]]] = [(first_pixel, [start, first_pixel])]
    local_visited: set[Pixel] = {start, first_pixel}

    while stack:
        current, path = stack.pop()
        for neighbour in skeleton_neighbours(current, skeleton):
            if neighbour == start or neighbour in local_visited:
                continue
            candidate = path + [neighbour]
            if neighbour in node_pixels:
                return candidate
            local_visited.add(neighbour)
            stack.append((neighbour, candidate))
    return None


def canonical_path(path: list[Pixel]) -> tuple[Pixel, ...]:
    """Return orientation-independent path coordinates for duplicate checks."""
    forward = tuple(path)
    backward = tuple(reversed(path))
    return min(forward, backward)


def edge_attributes(path: list[Pixel]) -> dict[str, float | str]:
    """Calculate edge length, ordered path, Euclidean distance and tortuosity."""
    length = float(
        sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
    )
    euclidean_distance = float(np.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1]))
    tortuosity = length / euclidean_distance if euclidean_distance > 0 else 1.0
    # GraphML supports scalar values, therefore the ordered coordinate list is JSON.
    path_coordinates = json.dumps([[column, row] for row, column in path], separators=(",", ":"))
    return {
        "length": length,
        "path_coordinates": path_coordinates,
        "euclidean_distance": euclidean_distance,
        "tortuosity": float(tortuosity),
    }


def build_vessel_graph(mask: np.ndarray, min_component_size: int) -> GraphBuildResult:
    """Build a vessel graph from one binary mask.

    Nodes are only endpoints and branch points. DFS is launched from every
    node-neighbour pair. Interior pixels of each accepted edge are added to a
    global visited-pixel set; direct node-node links are tracked separately.
    """
    cleaned_mask = remove_tiny_components(mask, min_component_size)
    skeleton = skeletonize(cleaned_mask)
    _, skeleton_component_count = ndi.label(
        skeleton, structure=np.ones((3, 3), dtype=np.uint8)
    )

    neighbour_counts = count_skeleton_neighbours(skeleton)
    endpoints, branches = detect_graph_nodes(neighbour_counts)
    node_pixels = endpoints | branches

    graph = nx.Graph()
    pixel_to_node: dict[Pixel, str] = {}
    for index, pixel in enumerate(sorted(node_pixels)):
        row, column = pixel
        node_id = f"node_{index}"
        graph.add_node(
            node_id,
            id=node_id,
            x=int(column),
            y=int(row),
            degree=int(neighbour_counts[row, column]),
            node_type="endpoint" if pixel in endpoints else "branch",
        )
        pixel_to_node[pixel] = node_id

    visited_pixels: set[Pixel] = set()
    visited_direct_links: set[frozenset[Pixel]] = set()
    seen_paths: set[tuple[Pixel, ...]] = set()

    for start in sorted(node_pixels):
        for first_pixel in skeleton_neighbours(start, skeleton):
            direct_link = frozenset((start, first_pixel))
            if first_pixel not in node_pixels and first_pixel in visited_pixels:
                continue
            if first_pixel in node_pixels and direct_link in visited_direct_links:
                continue

            path = trace_edge_dfs(start, first_pixel, skeleton, node_pixels)
            if path is None or len(path) < 2:
                continue

            end = path[-1]
            unique_path = canonical_path(path)
            if end == start or unique_path in seen_paths:
                continue

            source, target = pixel_to_node[start], pixel_to_node[end]
            # Simple Graph plus canonical paths prevents duplicated graph edges.
            if graph.has_edge(source, target):
                continue

            graph.add_edge(source, target, **edge_attributes(path))
            seen_paths.add(unique_path)
            visited_pixels.update(pixel for pixel in path[1:-1] if pixel not in node_pixels)
            if first_pixel in node_pixels:
                visited_direct_links.add(direct_link)

    return GraphBuildResult(
        graph=graph,
        endpoint_count=len(endpoints),
        branch_point_count=len(branches),
        skeleton_component_count=int(skeleton_component_count),
    )


def print_summary(image_path: Path, result: GraphBuildResult) -> None:
    """Print the requested graph-construction summary for one input mask."""
    graph = result.graph
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    average_degree = sum(dict(graph.degree()).values()) / node_count if node_count else 0.0
    average_length = (
        sum(float(data["length"]) for _, _, data in graph.edges(data=True)) / edge_count
        if edge_count else 0.0
    )
    print(f"\n{image_path.name}")
    print(f"  nodes: {node_count}")
    print(f"  edges: {edge_count}")
    print(f"  endpoints: {result.endpoint_count}")
    print(f"  branch points: {result.branch_point_count}")
    print(f"  connected components: {result.skeleton_component_count}")
    print(f"  average node degree: {average_degree:.3f}")
    print(f"  average edge length: {average_length:.3f}")


def process_masks(input_directory: Path, output_directory: Path, min_component_size: int) -> None:
    """Build and save one ``.graphml`` file for every input BMP mask."""
    if not input_directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")

    output_directory.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(input_directory.glob("*.bmp"))
    if not image_paths:
        print(f"No BMP masks found in: {input_directory}")
        return

    for image_path in image_paths:
        result = build_vessel_graph(load_binary_mask(image_path), min_component_size)
        output_path = output_directory / f"{image_path.stem}.graphml"
        nx.write_graphml(result.graph, output_path, named_key_ids=True)
        print_summary(image_path, result)
        print(f"  saved: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse batch-processing options."""
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="OCTA vessel Graph Builder V3")
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=project_root / "Dataset" / "OCTA-500" / "Output_clDice",
        help="Directory containing binary .bmp vessel masks.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=project_root / "Graph" / "Graphs",
        help="Directory where .graphml files will be written.",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=10,
        help="Remove vessel components smaller than this many pixels (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    """Run Graph Builder V3."""
    arguments = parse_arguments()
    process_masks(arguments.input_directory, arguments.output_directory, arguments.min_component_size)


if __name__ == "__main__":
    main()