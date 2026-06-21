from __future__ import annotations

from flax import struct
import jax.numpy as jnp

from .types import GraphData
from .types import NO_EDGE


@struct.dataclass
class TableRouter:
    """JAX-friendly baseline router backed by precomputed directed tables."""

    graph: GraphData

    def next_hop(self, current_node: jnp.ndarray, target_node: jnp.ndarray, traffic_state=None) -> jnp.ndarray:
        edge = self.next_edge(current_node, target_node, traffic_state)
        safe_edge = jnp.clip(edge, 0, self.graph.max_edges - 1)
        return jnp.where(edge == NO_EDGE, current_node, self.graph.edge_to[safe_edge])

    def next_edge(self, current_node: jnp.ndarray, target_node: jnp.ndarray, traffic_state=None) -> jnp.ndarray:
        del traffic_state
        if self.graph.route_mode == "dense":
            return self.graph.next_edge_table[current_node, target_node]
        via_landmarks = self.graph.node_to_landmark_time[current_node, :] + self.graph.landmark_to_node_time[:, target_node]
        landmark_idx = jnp.argmin(via_landmarks).astype(jnp.int32)
        landmark_node = self.graph.landmark_nodes[landmark_idx]
        to_landmark = self.graph.node_to_landmark_next_edge[current_node, landmark_idx]
        from_landmark = self.graph.landmark_to_node_next_edge[landmark_idx, target_node]
        edge = jnp.where(current_node != landmark_node, to_landmark, from_landmark)
        return jnp.where(current_node == target_node, NO_EDGE, edge)

    def travel_time_estimate(self, source_node: jnp.ndarray, target_node: jnp.ndarray, traffic_state=None) -> jnp.ndarray:
        del traffic_state
        if self.graph.route_mode == "dense":
            return self.graph.travel_time_table[source_node, target_node]
        via_landmarks = self.graph.node_to_landmark_time[source_node, :] + jnp.moveaxis(
            self.graph.landmark_to_node_time[:, target_node],
            0,
            -1,
        )
        best = jnp.min(via_landmarks, axis=-1)
        return jnp.where(source_node == target_node, 0.0, best)
