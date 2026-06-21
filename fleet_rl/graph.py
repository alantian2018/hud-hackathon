from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from .types import GraphData, NO_EDGE


@dataclass(frozen=True)
class RouteOracleResult:
    node_path: list[int]
    edge_path: list[int]
    cost: float


def _normalize_xy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_span = max(1e-9, float(lon.max() - lon.min()))
    lat_span = max(1e-9, float(lat.max() - lat.min()))
    return (lon - lon.min()) / lon_span, (lat - lat.min()) / lat_span


def _node_demand_weights(node_list: list[dict], num_nodes: int) -> np.ndarray:
    raw = np.array(
        [
            float(
                node.get(
                    "demand_weight",
                    node.get("population_density_noisy", node.get("population_density_base", 1.0)),
                )
            )
            for node in node_list
        ],
        dtype=np.float32,
    )
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 1.0)
    max_value = max(1e-9, float(raw.max(initial=1.0)))
    weights = raw / max_value
    if not np.any(weights > 0.0):
        weights = np.ones((num_nodes,), dtype=np.float32)
    return weights.astype(np.float32)


def _precompute_routes(num_nodes: int, edges: list[dict], max_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(num_nodes)]
    for edge_idx, edge in enumerate(edges):
        adjacency[int(edge["from"])].append((int(edge["to"]), edge_idx, float(edge["time"])))

    travel = np.full((max_nodes, max_nodes), np.inf, dtype=np.float32)
    next_hop = np.full((max_nodes, max_nodes), -1, dtype=np.int32)
    next_edge = np.full((max_nodes, max_nodes), NO_EDGE, dtype=np.int32)

    for source in range(num_nodes):
        travel[source, source] = 0.0
        next_hop[source, source] = source
        frontier: list[tuple[float, int]] = [(0.0, source)]
        best = {source: 0.0}
        parent: dict[int, tuple[int, int]] = {}
        while frontier:
            cost, node = heapq.heappop(frontier)
            if cost > best.get(node, math.inf):
                continue
            for to_node, edge_idx, weight in adjacency[node]:
                next_cost = cost + max(1e-6, weight)
                if next_cost < best.get(to_node, math.inf):
                    best[to_node] = next_cost
                    parent[to_node] = (node, edge_idx)
                    heapq.heappush(frontier, (next_cost, to_node))

        for target, cost in best.items():
            travel[source, target] = np.float32(cost)
            if target == source:
                continue
            node = target
            edge_idx = NO_EDGE
            while parent[node][0] != source:
                node = parent[node][0]
            first_parent, edge_idx = parent[node]
            if first_parent != source:
                raise AssertionError("route reconstruction failed")
            next_hop[source, target] = node
            next_edge[source, target] = edge_idx

    return next_hop, next_edge, travel


def _dijkstra_first_edges(
    num_nodes: int,
    adjacency: list[list[tuple[int, int, float]]],
    source: int,
) -> tuple[np.ndarray, np.ndarray]:
    dist = np.full((num_nodes,), np.inf, dtype=np.float32)
    first_edge = np.full((num_nodes,), NO_EDGE, dtype=np.int32)
    dist[source] = 0.0
    frontier: list[tuple[float, int]] = [(0.0, source)]
    while frontier:
        cost, node = heapq.heappop(frontier)
        if cost > float(dist[node]):
            continue
        for to_node, edge_idx, weight in adjacency[node]:
            next_cost = cost + max(1e-6, float(weight))
            if next_cost < float(dist[to_node]):
                dist[to_node] = np.float32(next_cost)
                first_edge[to_node] = edge_idx if node == source else first_edge[node]
                heapq.heappush(frontier, (next_cost, to_node))
    first_edge[source] = NO_EDGE
    return dist, first_edge


def _dijkstra_reverse_next_edges(
    num_nodes: int,
    reverse_adjacency: list[list[tuple[int, int, float]]],
    target: int,
) -> tuple[np.ndarray, np.ndarray]:
    dist = np.full((num_nodes,), np.inf, dtype=np.float32)
    next_edge = np.full((num_nodes,), NO_EDGE, dtype=np.int32)
    dist[target] = 0.0
    frontier: list[tuple[float, int]] = [(0.0, target)]
    while frontier:
        cost, node = heapq.heappop(frontier)
        if cost > float(dist[node]):
            continue
        for prev_node, edge_idx, weight in reverse_adjacency[node]:
            next_cost = cost + max(1e-6, float(weight))
            if next_cost < float(dist[prev_node]):
                dist[prev_node] = np.float32(next_cost)
                next_edge[prev_node] = edge_idx
                heapq.heappush(frontier, (next_cost, prev_node))
    next_edge[target] = NO_EDGE
    return dist, next_edge


def _precompute_landmark_routes(
    num_nodes: int,
    edges: list[dict],
    max_nodes: int,
    num_landmarks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_landmarks = max(1, min(int(num_landmarks), num_nodes))
    landmark_nodes = np.linspace(0, num_nodes - 1, num_landmarks, dtype=np.int32)
    forward: list[list[tuple[int, int, float]]] = [[] for _ in range(num_nodes)]
    reverse: list[list[tuple[int, int, float]]] = [[] for _ in range(num_nodes)]
    for edge_idx, edge in enumerate(edges):
        src = int(edge["from"])
        dst = int(edge["to"])
        weight = float(edge["time"])
        forward[src].append((dst, edge_idx, weight))
        reverse[dst].append((src, edge_idx, weight))

    landmark_to_node_time = np.full((num_landmarks, max_nodes), np.inf, dtype=np.float32)
    node_to_landmark_time = np.full((max_nodes, num_landmarks), np.inf, dtype=np.float32)
    landmark_to_node_next_edge = np.full((num_landmarks, max_nodes), NO_EDGE, dtype=np.int32)
    node_to_landmark_next_edge = np.full((max_nodes, num_landmarks), NO_EDGE, dtype=np.int32)

    for landmark_idx, node in enumerate(landmark_nodes.tolist()):
        forward_dist, forward_first_edge = _dijkstra_first_edges(num_nodes, forward, int(node))
        reverse_dist, reverse_next_edge = _dijkstra_reverse_next_edges(num_nodes, reverse, int(node))
        landmark_to_node_time[landmark_idx, :num_nodes] = forward_dist
        landmark_to_node_next_edge[landmark_idx, :num_nodes] = forward_first_edge
        node_to_landmark_time[:num_nodes, landmark_idx] = reverse_dist
        node_to_landmark_next_edge[:num_nodes, landmark_idx] = reverse_next_edge

    return (
        landmark_nodes,
        landmark_to_node_time,
        node_to_landmark_time,
        landmark_to_node_next_edge,
        node_to_landmark_next_edge,
    )


def build_graph_from_edges(
    nodes: Iterable[dict],
    edges: Iterable[dict],
    *,
    raster_size: int = 50,
    max_nodes: int | None = None,
    max_edges: int | None = None,
    pickup_prob: Iterable[float] | None = None,
    dropoff_prob: Iterable[float] | None = None,
    route_mode: str = "dense",
    num_landmarks: int = 1,
) -> GraphData:
    node_list = list(nodes)
    raw_edges = list(edges)
    if not node_list:
        raise ValueError("graph requires at least one node")
    if not raw_edges:
        raise ValueError("graph requires at least one directed edge")

    compact_by_original = {int(node["id"]): idx for idx, node in enumerate(node_list)}
    num_nodes = len(node_list)
    converted_edges: list[dict] = []
    for idx, edge in enumerate(raw_edges):
        src = compact_by_original[int(edge["from"])]
        dst = compact_by_original[int(edge["to"])]
        converted_edges.append(
            {
                "id": int(edge.get("id", idx)),
                "from": src,
                "to": dst,
                "length": float(edge.get("length_m", edge.get("length", 1.0))),
                "time": float(edge.get("base_travel_time_s", edge.get("time", 1.0))),
                "traffic_profile": list(edge.get("traffic_profile", [1.0] * 24)),
                "congestion": float(edge.get("congestion", 1.0)),
            }
        )

    num_edges = len(converted_edges)
    max_nodes = int(max_nodes or num_nodes)
    max_edges = int(max_edges or num_edges)
    if num_nodes > max_nodes or num_edges > max_edges:
        raise ValueError("graph exceeds requested static capacity")

    out_degree_actual = np.zeros((num_nodes,), dtype=np.int32)
    for edge in converted_edges:
        out_degree_actual[edge["from"]] += 1
    if np.any(out_degree_actual == 0):
        sinks = np.flatnonzero(out_degree_actual == 0).tolist()
        raise ValueError(f"policy-controllable nodes must have outgoing edges; sinks={sinks[:8]}")

    max_degree = int(max(out_degree_actual.max(), 1))
    out_edges = np.full((max_nodes, max_degree), NO_EDGE, dtype=np.int32)
    out_edge_mask = np.zeros((max_nodes, max_degree), dtype=bool)
    cursor = np.zeros((num_nodes,), dtype=np.int32)
    for edge_idx, edge in enumerate(converted_edges):
        src = int(edge["from"])
        slot = int(cursor[src])
        out_edges[src, slot] = edge_idx
        out_edge_mask[src, slot] = True
        cursor[src] += 1

    lon = np.array([float(node["lon"]) for node in node_list], dtype=np.float32)
    lat = np.array([float(node["lat"]) for node in node_list], dtype=np.float32)
    x, y = _normalize_xy(lon, lat)

    node_lon = np.zeros((max_nodes,), dtype=np.float32)
    node_lat = np.zeros((max_nodes,), dtype=np.float32)
    node_x = np.zeros((max_nodes,), dtype=np.float32)
    node_y = np.zeros((max_nodes,), dtype=np.float32)
    node_raster_row = np.zeros((max_nodes,), dtype=np.int32)
    node_raster_col = np.zeros((max_nodes,), dtype=np.int32)
    node_demand_weight = np.zeros((max_nodes,), dtype=np.float32)
    original_node_ids = np.full((max_nodes,), -1, dtype=np.int64)
    demand_weights = _node_demand_weights(node_list, num_nodes)
    node_lon[:num_nodes] = lon
    node_lat[:num_nodes] = lat
    node_x[:num_nodes] = x
    node_y[:num_nodes] = y
    node_raster_row[:num_nodes] = np.clip((y * (raster_size - 1)).astype(np.int32), 0, raster_size - 1)
    node_raster_col[:num_nodes] = np.clip((x * (raster_size - 1)).astype(np.int32), 0, raster_size - 1)
    node_demand_weight[:num_nodes] = demand_weights
    original_node_ids[:num_nodes] = [int(node["id"]) for node in node_list]

    edge_from = np.zeros((max_edges,), dtype=np.int32)
    edge_to = np.zeros((max_edges,), dtype=np.int32)
    edge_raster_row = np.zeros((max_edges,), dtype=np.int32)
    edge_raster_col = np.zeros((max_edges,), dtype=np.int32)
    edge_length = np.ones((max_edges,), dtype=np.float32)
    edge_time = np.ones((max_edges,), dtype=np.float32)
    edge_profile = np.ones((max_edges, 24), dtype=np.float32)
    edge_congestion = np.ones((max_edges,), dtype=np.float32)
    edge_original_ids = np.full((max_edges,), -1, dtype=np.int64)
    for idx, edge in enumerate(converted_edges):
        edge_from[idx] = int(edge["from"])
        edge_to[idx] = int(edge["to"])
        edge_raster_row[idx] = node_raster_row[int(edge["to"])]
        edge_raster_col[idx] = node_raster_col[int(edge["to"])]
        edge_length[idx] = np.float32(edge["length"])
        edge_time[idx] = np.float32(edge["time"])
        profile = np.array(edge["traffic_profile"], dtype=np.float32)
        if profile.size != 24:
            profile = np.resize(profile, 24).astype(np.float32)
        edge_profile[idx] = profile
        edge_congestion[idx] = np.float32(edge["congestion"])
        edge_original_ids[idx] = int(edge["id"])

    real_congestion_profile = edge_profile[:num_edges] * edge_congestion[:num_edges, None]
    traffic_heat = np.clip((real_congestion_profile - 1.0) / 3.0, 0.0, 1.0)
    traffic_mean_profile = traffic_heat.mean(axis=0).astype(np.float32)
    traffic_max_profile = traffic_heat.max(axis=0).astype(np.float32)

    if route_mode not in {"dense", "landmark"}:
        raise ValueError("route_mode must be 'dense' or 'landmark'")
    if route_mode == "dense":
        next_hop, next_edge, travel = _precompute_routes(num_nodes, converted_edges, max_nodes)
        landmark_nodes = np.zeros((1,), dtype=np.int32)
        landmark_to_node_time = np.zeros((1, max_nodes), dtype=np.float32)
        node_to_landmark_time = np.zeros((max_nodes, 1), dtype=np.float32)
        landmark_to_node_next_edge = np.full((1, max_nodes), NO_EDGE, dtype=np.int32)
        node_to_landmark_next_edge = np.full((max_nodes, 1), NO_EDGE, dtype=np.int32)
        actual_landmarks = 1
    else:
        next_hop = np.full((1, 1), -1, dtype=np.int32)
        next_edge = np.full((1, 1), NO_EDGE, dtype=np.int32)
        travel = np.full((1, 1), np.inf, dtype=np.float32)
        (
            landmark_nodes,
            landmark_to_node_time,
            node_to_landmark_time,
            landmark_to_node_next_edge,
            node_to_landmark_next_edge,
        ) = _precompute_landmark_routes(num_nodes, converted_edges, max_nodes, num_landmarks)
        actual_landmarks = int(landmark_nodes.shape[0])

    if pickup_prob is None:
        probs = demand_weights / max(1e-9, float(demand_weights.sum()))
    else:
        probs = np.array(list(pickup_prob), dtype=np.float32)
        probs = probs / max(1e-9, float(probs.sum()))
    if dropoff_prob is None:
        drop_probs = probs.copy()
    else:
        drop_probs = np.array(list(dropoff_prob), dtype=np.float32)
        drop_probs = drop_probs / max(1e-9, float(drop_probs.sum()))

    pickup = np.zeros((max_nodes,), dtype=np.float32)
    dropoff = np.zeros((max_nodes,), dtype=np.float32)
    demand = np.zeros((max_nodes,), dtype=np.float32)
    pickup[:num_nodes] = probs
    dropoff[:num_nodes] = drop_probs
    demand[:num_nodes] = probs
    demand_mean = np.array(float(demand_weights.mean()), dtype=np.float32)
    demand_max = np.array(float(demand_weights.max(initial=1.0)), dtype=np.float32)
    controllable = np.zeros((max_nodes,), dtype=bool)
    controllable[:num_nodes] = out_degree_actual > 0
    out_degree = np.zeros((max_nodes,), dtype=np.int32)
    out_degree[:num_nodes] = out_degree_actual

    return GraphData(
        num_nodes=jnp.array(num_nodes, dtype=jnp.int32),
        num_edges=jnp.array(num_edges, dtype=jnp.int32),
        node_lon=jnp.asarray(node_lon),
        node_lat=jnp.asarray(node_lat),
        node_x=jnp.asarray(node_x),
        node_y=jnp.asarray(node_y),
        node_raster_row=jnp.asarray(node_raster_row),
        node_raster_col=jnp.asarray(node_raster_col),
        node_demand_weight=jnp.asarray(node_demand_weight),
        original_node_ids=jnp.asarray(original_node_ids),
        edge_from=jnp.asarray(edge_from),
        edge_to=jnp.asarray(edge_to),
        edge_raster_row=jnp.asarray(edge_raster_row),
        edge_raster_col=jnp.asarray(edge_raster_col),
        edge_length_m=jnp.asarray(edge_length),
        edge_base_travel_time_s=jnp.asarray(edge_time),
        edge_traffic_profile=jnp.asarray(edge_profile),
        edge_congestion_base=jnp.asarray(edge_congestion),
        traffic_mean_profile=jnp.asarray(traffic_mean_profile),
        traffic_max_profile=jnp.asarray(traffic_max_profile),
        edge_original_ids=jnp.asarray(edge_original_ids),
        out_edges=jnp.asarray(out_edges),
        out_degree=jnp.asarray(out_degree),
        out_edge_mask=jnp.asarray(out_edge_mask),
        next_hop_table=jnp.asarray(next_hop),
        next_edge_table=jnp.asarray(next_edge),
        travel_time_table=jnp.asarray(travel),
        landmark_nodes=jnp.asarray(landmark_nodes),
        landmark_to_node_time=jnp.asarray(landmark_to_node_time),
        node_to_landmark_time=jnp.asarray(node_to_landmark_time),
        landmark_to_node_next_edge=jnp.asarray(landmark_to_node_next_edge),
        node_to_landmark_next_edge=jnp.asarray(node_to_landmark_next_edge),
        pickup_prob=jnp.asarray(pickup),
        dropoff_prob=jnp.asarray(dropoff),
        demand_prob=jnp.asarray(demand),
        demand_mean=jnp.asarray(demand_mean),
        demand_max=jnp.asarray(demand_max),
        controllable_mask=jnp.asarray(controllable),
        route_mode=route_mode,
        raster_size=int(raster_size),
        max_nodes=max_nodes,
        max_edges=max_edges,
        max_degree=max_degree,
        num_landmarks=actual_landmarks,
    )


def build_synthetic_debug_graph(name: str = "line") -> GraphData:
    if name == "line":
        nodes = [
            {"id": 0, "lon": 0.0, "lat": 0.0},
            {"id": 1, "lon": 1.0, "lat": 0.0},
            {"id": 2, "lon": 2.0, "lat": 0.0},
        ]
        edges = [
            {"id": 0, "from": 0, "to": 1, "length_m": 100.0, "base_travel_time_s": 10.0},
            {"id": 1, "from": 1, "to": 2, "length_m": 100.0, "base_travel_time_s": 10.0},
            {"id": 2, "from": 2, "to": 0, "length_m": 200.0, "base_travel_time_s": 20.0},
        ]
    elif name == "asymmetric":
        nodes = [
            {"id": 0, "lon": 0.0, "lat": 0.0},
            {"id": 1, "lon": 1.0, "lat": 0.0},
            {"id": 2, "lon": 2.0, "lat": 0.0},
        ]
        edges = [
            {"id": 0, "from": 0, "to": 1, "length_m": 1.0, "base_travel_time_s": 1.0},
            {"id": 1, "from": 1, "to": 2, "length_m": 1.0, "base_travel_time_s": 1.0},
            {"id": 2, "from": 2, "to": 1, "length_m": 1.0, "base_travel_time_s": 10.0},
            {"id": 3, "from": 1, "to": 0, "length_m": 1.0, "base_travel_time_s": 10.0},
        ]
    elif name == "directed_assignment":
        nodes = [
            {"id": 0, "lon": 0.0, "lat": 0.0},
            {"id": 1, "lon": 1.0, "lat": 0.0},
            {"id": 2, "lon": 2.0, "lat": 0.0},
        ]
        edges = [
            {"id": 0, "from": 0, "to": 1, "length_m": 100.0, "base_travel_time_s": 50.0},
            {"id": 1, "from": 1, "to": 0, "length_m": 100.0, "base_travel_time_s": 1.0},
            {"id": 2, "from": 2, "to": 1, "length_m": 100.0, "base_travel_time_s": 2.0},
            {"id": 3, "from": 1, "to": 2, "length_m": 100.0, "base_travel_time_s": 30.0},
        ]
    elif name == "variable_degree":
        nodes = [
            {"id": 0, "lon": 0.0, "lat": 0.0},
            {"id": 1, "lon": 1.0, "lat": 0.0},
            {"id": 2, "lon": 0.0, "lat": 1.0},
        ]
        edges = [
            {"id": 0, "from": 0, "to": 1, "length_m": 100.0, "base_travel_time_s": 10.0},
            {"id": 1, "from": 1, "to": 0, "length_m": 100.0, "base_travel_time_s": 10.0},
            {"id": 2, "from": 1, "to": 2, "length_m": 150.0, "base_travel_time_s": 12.0},
            {"id": 3, "from": 2, "to": 0, "length_m": 100.0, "base_travel_time_s": 10.0},
        ]
    elif name == "grid3":
        nodes = []
        edges = []
        edge_id = 0
        for r in range(3):
            for c in range(3):
                nodes.append({"id": r * 3 + c, "lon": float(c), "lat": float(r)})
        for r in range(3):
            for c in range(3):
                src = r * 3 + c
                for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < 3 and 0 <= cc < 3:
                        edges.append(
                            {
                                "id": edge_id,
                                "from": src,
                                "to": rr * 3 + cc,
                                "length_m": 80.0,
                                "base_travel_time_s": 8.0,
                            }
                        )
                        edge_id += 1
    else:
        raise ValueError(f"unknown synthetic graph: {name}")
    return build_graph_from_edges(nodes, edges, raster_size=50)


def python_shortest_path(graph: GraphData, source: int, target: int) -> RouteOracleResult:
    source = int(source)
    target = int(target)
    if source == target:
        return RouteOracleResult([source], [], 0.0)

    num_edges = int(graph.num_edges)
    adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(int(graph.num_nodes))]
    for edge_idx in range(num_edges):
        adjacency[int(graph.edge_from[edge_idx])].append(
            (int(graph.edge_to[edge_idx]), edge_idx, float(graph.edge_base_travel_time_s[edge_idx]))
        )

    frontier = [(0.0, source)]
    best = {source: 0.0}
    parent: dict[int, tuple[int, int]] = {}
    while frontier:
        cost, node = heapq.heappop(frontier)
        if node == target:
            break
        if cost > best.get(node, math.inf):
            continue
        for to_node, edge_idx, weight in adjacency[node]:
            next_cost = cost + weight
            if next_cost < best.get(to_node, math.inf):
                best[to_node] = next_cost
                parent[to_node] = (node, edge_idx)
                heapq.heappush(frontier, (next_cost, to_node))

    if target not in best:
        return RouteOracleResult([], [], math.inf)

    node_path = [target]
    edge_path = []
    while node_path[-1] != source:
        prev, edge_idx = parent[node_path[-1]]
        edge_path.append(edge_idx)
        node_path.append(prev)
    node_path.reverse()
    edge_path.reverse()
    return RouteOracleResult(node_path, edge_path, best[target])


def load_ppo_json_graph(
    nodes_path: str | Path,
    edges_path: str | Path,
    *,
    max_nodes: int | None = None,
    max_edges: int | None = None,
    raster_size: int = 50,
    node_limit: int | None = None,
    route_mode: str = "landmark",
    num_landmarks: int = 64,
) -> GraphData:
    nodes_payload = json.loads(Path(nodes_path).read_text(encoding="utf-8"))
    edges_payload = json.loads(Path(edges_path).read_text(encoding="utf-8"))
    raw_nodes = nodes_payload.get("nodes", [])
    if node_limit is not None:
        raw_nodes = raw_nodes[:node_limit]
    allowed = {int(node["node_id"]) for node in raw_nodes}

    nodes = [
        {
            "id": int(node["node_id"]),
            "lon": float(node["lon"]),
            "lat": float(node["lat"]),
            "population_density_base": float(node.get("population_density_base", 1.0)),
            "population_density_noisy": float(node.get("population_density_noisy", 1.0)),
        }
        for node in raw_nodes
    ]
    edges = []
    for edge in edges_payload.get("edges", []):
        u = int(edge["u"])
        v = int(edge["v"])
        if u not in allowed or v not in allowed:
            continue
        weights = edge.get("dynamic_weights_travel_time_s") or [1.0] * 24
        base_time = float(weights[0])
        features = edge.get("features", {})
        edges.append(
            {
                "id": len(edges),
                "from": u,
                "to": v,
                "length_m": float(features.get("length_m", 1.0)),
                "base_travel_time_s": base_time,
                "traffic_profile": [float(w) / max(1e-6, base_time) for w in weights],
            }
        )

    nodes, edges = _largest_strongly_connected_subgraph(nodes, edges)
    nodes, edges = _prune_sink_nodes(nodes, edges)
    return build_graph_from_edges(
        nodes,
        edges,
        raster_size=raster_size,
        max_nodes=max_nodes,
        max_edges=max_edges,
        route_mode=route_mode,
        num_landmarks=num_landmarks,
    )


def _prune_sink_nodes(nodes: list[dict], edges: list[dict]) -> tuple[list[dict], list[dict]]:
    active = {int(node["id"]) for node in nodes}
    changed = True
    while changed:
        out_nodes = {int(edge["from"]) for edge in edges if int(edge["from"]) in active and int(edge["to"]) in active}
        next_active = active & out_nodes
        changed = next_active != active
        active = next_active
        edges = [edge for edge in edges if int(edge["from"]) in active and int(edge["to"]) in active]
    if not active or not edges:
        raise ValueError("subgraph has no directed edges after sink pruning")
    return [node for node in nodes if int(node["id"]) in active], edges


def _largest_strongly_connected_subgraph(nodes: list[dict], edges: list[dict]) -> tuple[list[dict], list[dict]]:
    active = {int(node["id"]) for node in nodes}
    adjacency = {node_id: [] for node_id in active}
    reverse = {node_id: [] for node_id in active}
    for edge in edges:
        src = int(edge["from"])
        dst = int(edge["to"])
        if src in active and dst in active:
            adjacency[src].append(dst)
            reverse[dst].append(src)

    visited: set[int] = set()
    order: list[int] = []

    for start in active:
        if start in visited:
            continue
        stack: list[tuple[int, bool]] = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for nxt in adjacency.get(node, []):
                if nxt not in visited:
                    stack.append((nxt, False))

    assigned: set[int] = set()
    components: list[list[int]] = []
    for start in reversed(order):
        if start in assigned:
            continue
        component = []
        stack = [start]
        assigned.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in reverse.get(node, []):
                if nxt not in assigned:
                    assigned.add(nxt)
                    stack.append(nxt)
        components.append(component)

    if not components:
        raise ValueError("graph has no strongly connected component")
    largest = set(max(components, key=len))
    kept_nodes = [node for node in nodes if int(node["id"]) in largest]
    kept_edges = [edge for edge in edges if int(edge["from"]) in largest and int(edge["to"]) in largest]
    if not kept_edges:
        raise ValueError("largest strongly connected component has no edges")
    return kept_nodes, kept_edges
