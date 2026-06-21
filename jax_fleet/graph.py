from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import jax.numpy as jnp
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from jax_fleet.types import GraphArrays


@dataclass(frozen=True)
class RoutingCacheInfo:
    status: str
    table_path: Path
    manifest_path: Path
    fingerprint: str
    num_nodes: int
    num_edges: int


def _hourly_profile(edge: dict[str, Any]) -> np.ndarray:
    if "travel_time_profile_s" in edge:
        values = np.asarray(edge["travel_time_profile_s"], dtype=np.float32)
    else:
        base = float(edge.get("travel_time_s", 1.0))
        values = np.full((24,), base, dtype=np.float32)
        for hour, multiplier in edge.get("hourly_multiplier", {}).items():
            values[int(hour) % 24] = base * float(multiplier)
    if values.shape != (24,):
        raise ValueError("edge travel-time profiles must have exactly 24 hourly values")
    return values


def build_synthetic_graph(
    node_lonlat: Iterable[tuple[float, float]],
    edges: Iterable[dict[str, Any]],
) -> GraphArrays:
    nodes = np.asarray(list(node_lonlat), dtype=np.float32)
    edge_records = list(edges)
    sources = np.asarray([int(edge["source"]) for edge in edge_records], dtype=np.int32)
    targets = np.asarray([int(edge["target"]) for edge in edge_records], dtype=np.int32)
    profiles = np.stack([_hourly_profile(edge) for edge in edge_records], axis=0).astype(np.float32)
    lengths = np.asarray(
        [float(edge.get("length_m", edge.get("travel_time_s", 1.0))) for edge in edge_records],
        dtype=np.float32,
    )
    free_flow = np.maximum(profiles.min(axis=1, keepdims=True), 1e-3)
    congestion = (profiles / free_flow).astype(np.float32)
    node_grid_rows, node_grid_cols = _synthetic_grid_indices(nodes)
    next_edge_table, travel_time_table = build_routing_tables_numpy(
        num_nodes=len(nodes),
        sources=sources,
        targets=targets,
        travel_time_s=profiles.mean(axis=1),
    )
    return _assemble_graph(
        node_lonlat=nodes,
        sources=sources,
        targets=targets,
        lengths=lengths,
        travel_times=profiles,
        congestion=congestion,
        original_node_ids=np.arange(len(nodes), dtype=np.int64),
        node_grid_rows=node_grid_rows,
        node_grid_cols=node_grid_cols,
        node_population_density=np.zeros((len(nodes),), dtype=np.float32),
        routing_next_edge=next_edge_table,
        routing_travel_time_s=travel_time_table,
    )


def load_public_data_graph(
    data_dir: str | Path = "public/data",
    *,
    include_routing: bool = True,
    cache_dir: str | Path = "cache/jax_fleet",
    routing_chunk_size: int = 512,
    routing_progress: Callable[[int, int], None] | None = None,
) -> GraphArrays:
    data_dir = Path(data_dir)
    nodes_payload = json.loads((data_dir / "ppo_nodes.json").read_text(encoding="utf-8"))
    edges_payload = json.loads((data_dir / "ppo_edges.json").read_text(encoding="utf-8"))

    nodes_by_id = {
        int(node["node_id"]): node
        for node in nodes_payload.get("nodes", [])
    }

    best_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for edge in edges_payload.get("edges", []):
        u = int(edge["u"])
        v = int(edge["v"])
        weights = np.asarray(edge.get("dynamic_weights_travel_time_s", [1.0] * 24), dtype=np.float32)
        existing = best_by_pair.get((u, v))
        if existing is None or float(weights.mean()) < float(existing["_weights"].mean()):
            item = dict(edge)
            item["_weights"] = weights
            best_by_pair[(u, v)] = item

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes_by_id.keys())
    graph.add_edges_from(best_by_pair.keys())
    component = max(nx.strongly_connected_components(graph), key=len)
    compact_ids = {node_id: idx for idx, node_id in enumerate(sorted(component))}

    selected_edges = [
        edge
        for (u, v), edge in sorted(best_by_pair.items())
        if u in compact_ids and v in compact_ids
    ]
    sources = np.asarray([compact_ids[int(edge["u"])] for edge in selected_edges], dtype=np.int32)
    targets = np.asarray([compact_ids[int(edge["v"])] for edge in selected_edges], dtype=np.int32)
    travel_times = np.stack([edge["_weights"] for edge in selected_edges], axis=0).astype(np.float32)
    lengths = np.asarray(
        [
            float(edge.get("features", {}).get("length_m", travel_times[idx].mean()))
            for idx, edge in enumerate(selected_edges)
        ],
        dtype=np.float32,
    )
    free_flow = np.maximum(travel_times.min(axis=1, keepdims=True), 1e-3)
    congestion = (travel_times / free_flow).astype(np.float32)
    ordered_original_ids = np.asarray(sorted(component), dtype=np.int64)
    node_lonlat = np.asarray(
        [
            [float(nodes_by_id[node_id]["lon"]), float(nodes_by_id[node_id]["lat"])]
            for node_id in ordered_original_ids
        ],
        dtype=np.float32,
    )
    node_grid_rows = np.asarray(
        [int(nodes_by_id[node_id].get("grid_row", -1)) for node_id in ordered_original_ids],
        dtype=np.int32,
    )
    node_grid_cols = np.asarray(
        [int(nodes_by_id[node_id].get("grid_col", -1)) for node_id in ordered_original_ids],
        dtype=np.int32,
    )
    node_population_density = np.asarray(
        [
            float(
                nodes_by_id[node_id].get(
                    "population_density_noisy",
                    nodes_by_id[node_id].get("population_density_base", 0.0),
                )
            )
            for node_id in ordered_original_ids
        ],
        dtype=np.float32,
    )

    if include_routing:
        cache_info = ensure_routing_cache(
            num_nodes=len(node_lonlat),
            sources=sources,
            targets=targets,
            travel_time_s=travel_times.mean(axis=1),
            cache_dir=cache_dir,
            graph_key="sf_largest_scc",
            chunk_size=routing_chunk_size,
            progress=routing_progress,
        )
        routing_next_edge, routing_travel_time_s = load_routing_cache(cache_info.table_path)
    else:
        routing_next_edge = np.zeros((0, 0), dtype=np.int32)
        routing_travel_time_s = np.zeros((0, 0), dtype=np.float32)

    return _assemble_graph(
        node_lonlat=node_lonlat,
        sources=sources,
        targets=targets,
        lengths=lengths,
        travel_times=travel_times,
        congestion=congestion,
        original_node_ids=ordered_original_ids,
        node_grid_rows=node_grid_rows,
        node_grid_cols=node_grid_cols,
        node_population_density=node_population_density,
        routing_next_edge=routing_next_edge,
        routing_travel_time_s=routing_travel_time_s,
    )


def _assemble_graph(
    *,
    node_lonlat: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    lengths: np.ndarray,
    travel_times: np.ndarray,
    congestion: np.ndarray,
    original_node_ids: np.ndarray,
    node_grid_rows: np.ndarray,
    node_grid_cols: np.ndarray,
    node_population_density: np.ndarray,
    routing_next_edge: np.ndarray,
    routing_travel_time_s: np.ndarray,
) -> GraphArrays:
    num_nodes = int(node_lonlat.shape[0])
    num_edges = int(sources.shape[0])
    outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
    for edge_id, source in enumerate(sources):
        outgoing[int(source)].append(edge_id)

    max_degree = max((len(items) for items in outgoing), default=0)
    outgoing_edge_ids = np.full((num_nodes, max_degree), -1, dtype=np.int32)
    outgoing_target_nodes = np.full((num_nodes, max_degree), -1, dtype=np.int32)
    outgoing_mask = np.zeros((num_nodes, max_degree), dtype=bool)
    for node, edge_ids in enumerate(outgoing):
        for slot, edge_id in enumerate(edge_ids):
            outgoing_edge_ids[node, slot] = int(edge_id)
            outgoing_target_nodes[node, slot] = int(targets[edge_id])
            outgoing_mask[node, slot] = True

    bounds = np.asarray(
        [
            float(node_lonlat[:, 0].min()),
            float(node_lonlat[:, 1].min()),
            float(node_lonlat[:, 0].max()),
            float(node_lonlat[:, 1].max()),
        ],
        dtype=np.float32,
    )
    return GraphArrays(
        num_nodes=num_nodes,
        num_edges=num_edges,
        max_degree=max_degree,
        node_lonlat=jnp.asarray(node_lonlat, dtype=jnp.float32),
        edge_sources=jnp.asarray(sources, dtype=jnp.int32),
        edge_targets=jnp.asarray(targets, dtype=jnp.int32),
        edge_lengths_m=jnp.asarray(lengths, dtype=jnp.float32),
        edge_travel_time_s=jnp.asarray(travel_times, dtype=jnp.float32),
        edge_congestion=jnp.asarray(congestion, dtype=jnp.float32),
        outgoing_edge_ids=jnp.asarray(outgoing_edge_ids, dtype=jnp.int32),
        outgoing_target_nodes=jnp.asarray(outgoing_target_nodes, dtype=jnp.int32),
        outgoing_mask=jnp.asarray(outgoing_mask, dtype=jnp.bool_),
        routing_next_edge=jnp.asarray(routing_next_edge, dtype=jnp.int32),
        routing_travel_time_s=jnp.asarray(routing_travel_time_s, dtype=jnp.float32),
        original_node_ids=jnp.asarray(original_node_ids, dtype=jnp.int64),
        node_grid_rows=jnp.asarray(node_grid_rows, dtype=jnp.int32),
        node_grid_cols=jnp.asarray(node_grid_cols, dtype=jnp.int32),
        node_population_density=jnp.asarray(node_population_density, dtype=jnp.float32),
        bounds=jnp.asarray(bounds, dtype=jnp.float32),
    )


def _synthetic_grid_indices(node_lonlat: np.ndarray, *, grid_size: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if len(node_lonlat) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    min_lon = float(node_lonlat[:, 0].min())
    max_lon = float(node_lonlat[:, 0].max())
    min_lat = float(node_lonlat[:, 1].min())
    max_lat = float(node_lonlat[:, 1].max())
    span_lon = max(1e-9, max_lon - min_lon)
    span_lat = max(1e-9, max_lat - min_lat)
    cols = np.floor((node_lonlat[:, 0] - min_lon) / span_lon * grid_size).astype(np.int32)
    rows = np.floor((node_lonlat[:, 1] - min_lat) / span_lat * grid_size).astype(np.int32)
    return np.clip(rows, 0, grid_size - 1), np.clip(cols, 0, grid_size - 1)


def build_routing_tables_numpy(
    *,
    num_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
    travel_time_s: np.ndarray,
    chunk_size: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.maximum(np.asarray(travel_time_s, dtype=np.float32), 1e-3)
    matrix = csr_matrix((weights, (sources, targets)), shape=(num_nodes, num_nodes))
    edge_by_pair = {
        (int(source), int(target)): int(edge_id)
        for edge_id, (source, target) in enumerate(zip(sources, targets, strict=True))
    }
    next_edges = np.full((num_nodes, num_nodes), -1, dtype=np.int32)
    travel_times = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    chunk = max(1, int(chunk_size or num_nodes))

    for start in range(0, num_nodes, chunk):
        stop = min(num_nodes, start + chunk)
        indices = np.arange(start, stop, dtype=np.int32)
        dist, predecessors = dijkstra(
            matrix,
            directed=True,
            indices=indices,
            return_predecessors=True,
            unweighted=False,
        )
        dist = np.atleast_2d(dist)
        predecessors = np.atleast_2d(predecessors)
        travel_times[start:stop] = dist.astype(np.float32)
        _fill_next_edges_for_chunk(
            next_edges=next_edges,
            predecessors=predecessors,
            source_indices=indices,
            edge_by_pair=edge_by_pair,
            num_nodes=num_nodes,
        )
        if progress is not None:
            progress(stop, num_nodes)

    return next_edges, travel_times


def ensure_routing_cache(
    *,
    num_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
    travel_time_s: np.ndarray,
    cache_dir: str | Path,
    graph_key: str,
    chunk_size: int = 512,
    progress: Callable[[int, int], None] | None = None,
) -> RoutingCacheInfo:
    cache_dir = Path(cache_dir)
    table_path = cache_dir / f"{graph_key}_routing.npz"
    manifest_path = cache_dir / f"{graph_key}_routing_manifest.json"
    sources = np.asarray(sources, dtype=np.int32)
    targets = np.asarray(targets, dtype=np.int32)
    travel_time_s = np.asarray(travel_time_s, dtype=np.float32)
    fingerprint = routing_fingerprint(
        num_nodes=num_nodes,
        sources=sources,
        targets=targets,
        travel_time_s=travel_time_s,
    )

    if _routing_cache_is_valid(
        table_path=table_path,
        manifest_path=manifest_path,
        fingerprint=fingerprint,
        num_nodes=num_nodes,
        num_edges=len(sources),
    ):
        return RoutingCacheInfo(
            status="hit",
            table_path=table_path,
            manifest_path=manifest_path,
            fingerprint=fingerprint,
            num_nodes=num_nodes,
            num_edges=len(sources),
        )

    next_edges, travel_times = build_routing_tables_numpy(
        num_nodes=num_nodes,
        sources=sources,
        targets=targets,
        travel_time_s=travel_time_s,
        chunk_size=chunk_size,
        progress=progress,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        table_path,
        routing_next_edge=next_edges.astype(np.int32),
        routing_travel_time_s=travel_times.astype(np.float32),
    )
    manifest = {
        "graph_key": graph_key,
        "fingerprint": fingerprint,
        "num_nodes": int(num_nodes),
        "num_edges": int(len(sources)),
        "chunk_size": int(chunk_size),
        "table_path": str(table_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return RoutingCacheInfo(
        status="built",
        table_path=table_path,
        manifest_path=manifest_path,
        fingerprint=fingerprint,
        num_nodes=num_nodes,
        num_edges=len(sources),
    )


def load_routing_cache(table_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.load(Path(table_path))
    return (
        table["routing_next_edge"].astype(np.int32),
        table["routing_travel_time_s"].astype(np.float32),
    )


def routing_fingerprint(
    *,
    num_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
    travel_time_s: np.ndarray,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.asarray([num_nodes], dtype=np.int64).tobytes())
    hasher.update(np.ascontiguousarray(sources, dtype=np.int32).tobytes())
    hasher.update(np.ascontiguousarray(targets, dtype=np.int32).tobytes())
    hasher.update(np.ascontiguousarray(travel_time_s, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def _routing_cache_is_valid(
    *,
    table_path: Path,
    manifest_path: Path,
    fingerprint: str,
    num_nodes: int,
    num_edges: int,
) -> bool:
    if not table_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            return False
        if int(manifest.get("num_nodes", -1)) != int(num_nodes):
            return False
        if int(manifest.get("num_edges", -1)) != int(num_edges):
            return False
        next_edges, travel_times = load_routing_cache(table_path)
        return next_edges.shape == (num_nodes, num_nodes) and travel_times.shape == (
            num_nodes,
            num_nodes,
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _fill_next_edges_for_chunk(
    *,
    next_edges: np.ndarray,
    predecessors: np.ndarray,
    source_indices: np.ndarray,
    edge_by_pair: dict[tuple[int, int], int],
    num_nodes: int,
) -> None:
    for row, source in enumerate(source_indices):
        source = int(source)
        predecessor_row = predecessors[row]
        children: list[list[int]] = [[] for _ in range(num_nodes)]
        for node, parent in enumerate(predecessor_row):
            parent = int(parent)
            if parent < 0 or node == source:
                continue
            children[parent].append(node)

        for first_hop in children[source]:
            edge_id = edge_by_pair[(source, first_hop)]
            stack = [first_hop]
            while stack:
                node = stack.pop()
                next_edges[source, node] = edge_id
                stack.extend(children[node])
