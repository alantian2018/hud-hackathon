from __future__ import annotations

import jax.numpy as jnp

from jax_fleet.types import GraphArrays


def next_edge(graph: GraphArrays, current_node: int | jnp.ndarray, target_node: int | jnp.ndarray):
    return graph.routing_next_edge[jnp.asarray(current_node, jnp.int32), jnp.asarray(target_node, jnp.int32)]


def travel_time_estimate(
    graph: GraphArrays,
    source_node: int | jnp.ndarray,
    target_node: int | jnp.ndarray,
):
    return graph.routing_travel_time_s[
        jnp.asarray(source_node, jnp.int32),
        jnp.asarray(target_node, jnp.int32),
    ]


def shortest_path_edges(graph: GraphArrays, source_node: int, target_node: int) -> list[int]:
    source = int(source_node)
    target = int(target_node)
    if source == target:
        return []

    path: list[int] = []
    current = source
    for _ in range(graph.num_nodes + 1):
        edge_id = int(graph.routing_next_edge[current, target])
        if edge_id < 0:
            return []
        path.append(edge_id)
        current = int(graph.edge_targets[edge_id])
        if current == target:
            return path
    raise RuntimeError("routing table contains a cycle")
