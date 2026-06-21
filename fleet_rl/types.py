from __future__ import annotations

from flax import struct
import jax.numpy as jnp


POLICY_CONTROLLED = 0
TO_PICKUP = 1
TO_DROPOFF = 2

REQ_EMPTY = 0
REQ_QUEUED = 1
REQ_ASSIGNED = 2
REQ_PICKED_UP = 3
REQ_COMPLETED = 4

EVENT_REQUEST_SPAWNED = 1
EVENT_REQUEST_ASSIGNED = 2
EVENT_REQUEST_PICKED_UP = 3
EVENT_REQUEST_COMPLETED = 4
EVENT_REQUEST_DROPPED = 5
EVENT_OVERFLOW = 6

NO_EDGE = -1
NO_REQUEST = -1
NO_CAR = -1


@struct.dataclass
class GraphData:
    num_nodes: jnp.ndarray
    num_edges: jnp.ndarray
    node_lon: jnp.ndarray
    node_lat: jnp.ndarray
    node_x: jnp.ndarray
    node_y: jnp.ndarray
    node_raster_row: jnp.ndarray
    node_raster_col: jnp.ndarray
    node_demand_weight: jnp.ndarray
    original_node_ids: jnp.ndarray
    edge_from: jnp.ndarray
    edge_to: jnp.ndarray
    edge_raster_row: jnp.ndarray
    edge_raster_col: jnp.ndarray
    edge_length_m: jnp.ndarray
    edge_base_travel_time_s: jnp.ndarray
    edge_traffic_profile: jnp.ndarray
    edge_congestion_base: jnp.ndarray
    traffic_mean_profile: jnp.ndarray
    traffic_max_profile: jnp.ndarray
    edge_original_ids: jnp.ndarray
    out_edges: jnp.ndarray
    out_degree: jnp.ndarray
    out_edge_mask: jnp.ndarray
    next_hop_table: jnp.ndarray
    next_edge_table: jnp.ndarray
    travel_time_table: jnp.ndarray
    landmark_nodes: jnp.ndarray
    landmark_to_node_time: jnp.ndarray
    node_to_landmark_time: jnp.ndarray
    landmark_to_node_next_edge: jnp.ndarray
    node_to_landmark_next_edge: jnp.ndarray
    pickup_prob: jnp.ndarray
    dropoff_prob: jnp.ndarray
    demand_prob: jnp.ndarray
    demand_mean: jnp.ndarray
    demand_max: jnp.ndarray
    controllable_mask: jnp.ndarray
    route_mode: str = struct.field(pytree_node=False)
    raster_size: int = struct.field(pytree_node=False)
    max_nodes: int = struct.field(pytree_node=False)
    max_edges: int = struct.field(pytree_node=False)
    max_degree: int = struct.field(pytree_node=False)
    num_landmarks: int = struct.field(pytree_node=False)


@struct.dataclass
class Metrics:
    requests_spawned: jnp.ndarray
    requests_queued: jnp.ndarray
    requests_assigned: jnp.ndarray
    requests_picked_up: jnp.ndarray
    requests_completed: jnp.ndarray
    dropped_requests: jnp.ndarray
    queue_length: jnp.ndarray
    total_pickup_wait_time: jnp.ndarray
    avg_pickup_wait_time: jnp.ndarray
    p50_pickup_wait_time: jnp.ndarray
    p90_pickup_wait_time: jnp.ndarray
    p95_pickup_wait_time: jnp.ndarray
    pickup_wait_samples: jnp.ndarray
    pickup_wait_sample_times: jnp.ndarray
    pickup_wait_count: jnp.ndarray
    fleet_utilization: jnp.ndarray
    empty_driving_time: jnp.ndarray
    empty_driving_distance: jnp.ndarray
    invalid_actions: jnp.ndarray
    overflow: jnp.ndarray


@struct.dataclass
class Observation:
    raster: jnp.ndarray
    global_features: jnp.ndarray
    action_features: jnp.ndarray


@struct.dataclass
class Timestep:
    observation: Observation
    reward: jnp.ndarray
    discount: jnp.ndarray
    dt_seconds: jnp.ndarray
    done: jnp.ndarray
    truncated: jnp.ndarray
    metrics: Metrics
    action_mask: jnp.ndarray
    current_car_id: jnp.ndarray
    current_node_id: jnp.ndarray
    sim_time_seconds: jnp.ndarray


@struct.dataclass
class EnvState:
    rng: jnp.ndarray
    sim_time_seconds: jnp.ndarray
    next_request_time_seconds: jnp.ndarray
    request_id_counter: jnp.ndarray
    current_car_id: jnp.ndarray
    current_node_id: jnp.ndarray
    decision_pending: jnp.ndarray
    car_active: jnp.ndarray
    car_status: jnp.ndarray
    car_node: jnp.ndarray
    car_from_node: jnp.ndarray
    car_to_node: jnp.ndarray
    car_edge_id: jnp.ndarray
    car_edge_start_time: jnp.ndarray
    car_edge_end_time: jnp.ndarray
    car_assigned_request: jnp.ndarray
    car_target_node: jnp.ndarray
    request_ids: jnp.ndarray
    request_status: jnp.ndarray
    request_pickup_node: jnp.ndarray
    request_dropoff_node: jnp.ndarray
    request_spawn_time: jnp.ndarray
    request_assigned_car_id: jnp.ndarray
    request_pickup_time: jnp.ndarray
    request_dropoff_time: jnp.ndarray
    edge_congestion: jnp.ndarray
    recent_event_codes: jnp.ndarray
    recent_event_car_ids: jnp.ndarray
    recent_event_request_ids: jnp.ndarray
    recent_event_times: jnp.ndarray
    recent_event_cursor: jnp.ndarray
    metrics: Metrics
