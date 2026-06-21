from __future__ import annotations

from typing import Any

from flax import struct
import jax
import jax.numpy as jnp


Array = jax.Array


@struct.dataclass
class GraphArrays:
    num_nodes: int = struct.field(pytree_node=False)
    num_edges: int = struct.field(pytree_node=False)
    max_degree: int = struct.field(pytree_node=False)
    node_lonlat: Array
    edge_sources: Array
    edge_targets: Array
    edge_lengths_m: Array
    edge_travel_time_s: Array
    edge_congestion: Array
    outgoing_edge_ids: Array
    outgoing_target_nodes: Array
    outgoing_mask: Array
    routing_next_edge: Array
    routing_travel_time_s: Array
    original_node_ids: Array
    node_grid_rows: Array
    node_grid_cols: Array
    node_population_density: Array
    bounds: Array


@struct.dataclass
class EnvParams:
    graph: GraphArrays
    max_cars: int = struct.field(pytree_node=False)
    max_requests: int = struct.field(pytree_node=False)
    raster_size: int = struct.field(pytree_node=False, default=50)
    max_event_steps: int = struct.field(pytree_node=False, default=512)
    target_active_requests: int = struct.field(pytree_node=False, default=0)
    assignment_max_route_edges: int = struct.field(pytree_node=False, default=15)
    initial_car_nodes: Array = struct.field(default_factory=lambda: jnp.zeros((1,), jnp.int32))
    start_time_seconds: Array = struct.field(default_factory=lambda: jnp.asarray(0.0, jnp.float32))
    episode_seconds: Array = struct.field(default_factory=lambda: jnp.asarray(3600.0, jnp.float32))
    spawn_rate_per_minute: Array = struct.field(default_factory=lambda: jnp.asarray(0.0, jnp.float32))
    density_spawn_patience_seconds: Array = struct.field(
        default_factory=lambda: jnp.asarray(jnp.inf, jnp.float32)
    )
    density_destination_time_shift_seconds: Array = struct.field(
        default_factory=lambda: jnp.asarray(2.0 * 3600.0, jnp.float32)
    )
    wait_time_scale: Array = struct.field(default_factory=lambda: jnp.asarray(1.0 / 60.0, jnp.float32))
    gamma: Array = struct.field(default_factory=lambda: jnp.asarray(0.99, jnp.float32))
    preplanned_spawn_times: Array = struct.field(default_factory=lambda: jnp.zeros((0,), jnp.float32))
    preplanned_origin_nodes: Array = struct.field(default_factory=lambda: jnp.zeros((0,), jnp.int32))
    preplanned_dest_nodes: Array = struct.field(default_factory=lambda: jnp.zeros((0,), jnp.int32))
    preplanned_deadline_times: Array = struct.field(default_factory=lambda: jnp.zeros((0,), jnp.float32))
    node_density_by_hour: Array = struct.field(default_factory=lambda: jnp.ones((24, 1), jnp.float32))
    edge_raster_by_hour: Array = struct.field(default_factory=lambda: jnp.zeros((24, 1, 1), jnp.float32))


@struct.dataclass
class Observation:
    raster: Array
    local_raster: Array
    structured: Array
    candidate_edges: Array
    action_mask: Array


@struct.dataclass
class EnvMetrics:
    invalid_actions: Array
    dropped_requests: Array
    completed_requests: Array
    queued_requests: Array
    pickup_wait_seconds: Array
    aggregate_reward: Array
    recent_pickup_wait_seconds: Array
    recent_pickup_wait_count: Array
    recent_pickup_wait_index: Array


@struct.dataclass
class EnvState:
    rng: Array
    time_seconds: Array
    car_nodes: Array
    car_status: Array
    car_edge_ids: Array
    car_target_nodes: Array
    car_goal_nodes: Array
    car_request_ids: Array
    car_ready_times: Array
    car_departure_times: Array
    car_edge_durations: Array
    request_status: Array
    request_origin_nodes: Array
    request_dest_nodes: Array
    request_spawn_times: Array
    request_deadline_times: Array
    request_assigned_car_ids: Array
    request_pickup_times: Array
    current_car_id: Array
    decision_required: Array
    done: Array
    next_random_spawn_time: Array
    next_scheduled_request_index: Array
    step_count: Array
    metrics: EnvMetrics


@struct.dataclass
class Timestep:
    observation: Observation
    reward: Array
    discount: Array
    done: Array
    dt_seconds: Array
    metrics: EnvMetrics


def tree_to_numpy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: x.copy() if hasattr(x, "copy") else x, tree)
