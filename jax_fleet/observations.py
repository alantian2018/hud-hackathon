from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_fleet.types import EnvParams, EnvState, Observation


OBSERVATION_MODE_LEGACY = "legacy"
OBSERVATION_MODE_LEARNING_V1 = "learning_v1"

_LOCAL_VIEW_HALF_WIDTH_BLOCKS = 5.0
_ROUTE_FEATURE_MAX_HOPS = 256
_CAR_DECISION_STATUS = 0
_REQUEST_QUEUED_STATUS = 1
_REQUEST_ASSIGNED_STATUS = 2
_REQUEST_ONBOARD_STATUS = 3

# Learning raster channel order:
# 0 available car count
# 1 queued request origin count
# 2 queued request wait-sum, normalized by 30 minutes
# 3 expected demand density over the next 10 minutes
# 4 future available cars arriving in 0-5 minutes
# 5 future available cars arriving in 5-15 minutes
# 6 active request destination count
# 7 current-car focus cell
RASTER_CHANNELS = 8
RASTER_AVAILABLE_CARS = 0
RASTER_QUEUED_ORIGINS = 1
RASTER_QUEUED_WAIT_SUM_MIN = 2
RASTER_EXPECTED_DEMAND_10M = 3
RASTER_FUTURE_SUPPLY_0_5M = 4
RASTER_FUTURE_SUPPLY_5_15M = 5
RASTER_ACTIVE_DESTINATIONS = 6
RASTER_FOCUS_CAR = 7

# Legacy raster channel order:
# 0 available car counts
# 1 busy car counts
# 2 queued request origin counts
# 3 edge congestion raster
# 4 normalized node demand density
# 5 current-car focus cell
LEGACY_RASTER_CHANNELS = 6
LEGACY_AVAILABLE_CAR_CHANNEL = 0
LEGACY_BUSY_CAR_CHANNEL = 1
LEGACY_REQUEST_CHANNEL = 2
LEGACY_CONGESTION_CHANNEL = 3
LEGACY_DENSITY_CHANNEL = 4
LEGACY_FOCUS_CAR_CHANNEL = 5

# Structured feature order:
# 0 elapsed episode fraction
# 1 sin(time of day)
# 2 cos(time of day)
# 3 queued_requests / max_requests
# 4 available_cars / max_cars
# 5 busy_cars / max_cars
# 6 spawn_rate_per_minute / 10
# 7 mean queued wait minutes / 30
# 8 max queued wait minutes / 30
# 9 current car normalized x
# 10 current car normalized y
STRUCTURED_FEATURES = 11
STRUCT_ELAPSED_FRACTION = 0
STRUCT_TIME_SIN = 1
STRUCT_TIME_COS = 2
STRUCT_QUEUED_FRACTION = 3
STRUCT_AVAILABLE_FRACTION = 4
STRUCT_BUSY_FRACTION = 5
STRUCT_SPAWN_RATE = 6
STRUCT_MEAN_QUEUE_WAIT = 7
STRUCT_MAX_QUEUE_WAIT = 8
STRUCT_CURRENT_X = 9
STRUCT_CURRENT_Y = 10
LEGACY_STRUCTURED_FEATURES = 10

# Candidate-edge feature order:
# 0 target dx from current car
# 1 target dy from current car
# 2 target normalized x
# 3 target normalized y
# 4 edge length km
# 5 edge travel time minutes
# 6 edge congestion
# 7 validity flag
# 8 queued reachable count within 5 minutes from target
# 9 queued reachable count within 10 minutes from target
# 10 wait-weighted reachable demand within 10 minutes
# 11 max wait of reachable queued request
# 12 min ETA from target to any queued request
# 13 positive ETA improvement of the target position vs current node
# 14 wait-weighted positive ETA improvement of the target position
# 15 queued requests for which this car is best after taking the edge
# 16 wait-weighted advantage over best other available car
# 17 expected demand near target next 10 minutes
# 18 expected demand near target next 30 minutes
# 19 future supply near target 0-5 minutes
# 20 future supply near target 5-15 minutes
# 21 target supply-demand imbalance
CANDIDATE_EDGE_FEATURES = 22
CE_TARGET_DX = 0
CE_TARGET_DY = 1
CE_TARGET_X = 2
CE_TARGET_Y = 3
CE_LENGTH_KM = 4
CE_TRAVEL_TIME_MIN = 5
CE_CONGESTION = 6
CE_VALID = 7
CE_REACHABLE_5M = 8
CE_REACHABLE_10M = 9
CE_WAIT_WEIGHTED_REACHABLE = 10
CE_MAX_REACHABLE_WAIT = 11
CE_MIN_ETA_TO_QUEUE = 12
CE_ETA_IMPROVEMENT = 13
CE_WAIT_WEIGHTED_ETA_IMPROVEMENT = 14
CE_BEST_AFTER_EDGE_COUNT = 15
CE_WAIT_WEIGHTED_ADVANTAGE = 16
CE_EXPECTED_DEMAND_10M = 17
CE_EXPECTED_DEMAND_30M = 18
CE_FUTURE_SUPPLY_0_5M = 19
CE_FUTURE_SUPPLY_5_15M = 20
CE_SUPPLY_DEMAND_IMBALANCE = 21
LEGACY_CANDIDATE_EDGE_FEATURES = 12

_WAIT_FEATURE_NORMALIZER_MINUTES = 30.0
_ETA_FEATURE_NORMALIZER_MINUTES = 30.0
_ETA_FEATURE_NORMALIZER_SECONDS = _ETA_FEATURE_NORMALIZER_MINUTES * 60.0
_SPAWN_RATE_NORMALIZER_PER_MINUTE = 10.0


def build_observation(state: EnvState, params: EnvParams) -> Observation:
    if params.observation_mode == OBSERVATION_MODE_LEGACY:
        return _build_legacy_observation(state, params)
    return _build_learning_observation(state, params)


def _build_learning_observation(state: EnvState, params: EnvParams) -> Observation:
    graph = params.graph
    current_car = jnp.clip(state.current_car_id, 0, params.max_cars - 1)
    current_node = jnp.clip(state.car_nodes[current_car], 0, graph.num_nodes - 1)
    decision = state.decision_required & (~state.done)

    action_mask = graph.outgoing_mask[current_node] & decision
    edge_ids = jnp.where(action_mask, graph.outgoing_edge_ids[current_node], 0)
    edge_targets = jnp.clip(graph.edge_targets[edge_ids], 0, graph.num_nodes - 1)
    current_lonlat = graph.node_lonlat[current_node]
    target_lonlat = graph.node_lonlat[edge_targets]
    current_xy = _normalized_lonlat(current_lonlat, params)
    target_xy = _normalized_lonlat(target_lonlat, params)
    delta_xy = target_xy - current_xy
    duration = _edge_travel_time_at(edge_ids, state.time_seconds, graph.edge_travel_time_s)
    length = graph.edge_lengths_m[edge_ids]
    hour = (jnp.floor(state.time_seconds / 3600.0).astype(jnp.int32)) % 24
    congestion = graph.edge_congestion[edge_ids, hour]
    base_features = jnp.stack(
        [
            delta_xy[:, 0],
            delta_xy[:, 1],
            target_xy[:, 0],
            target_xy[:, 1],
            length / 1000.0,
            duration / 60.0,
            congestion,
            action_mask.astype(jnp.float32),
        ],
        axis=-1,
    )
    marginal_features = _candidate_marginal_value_features(
        state,
        params,
        current_car,
        current_node,
        edge_targets,
        duration,
    )
    candidate_edges = jnp.concatenate([base_features, marginal_features], axis=-1)
    candidate_edges = jnp.where(action_mask[:, None], candidate_edges, 0.0)
    candidate_edges = jnp.where(jnp.isfinite(candidate_edges), candidate_edges, 0.0)

    return Observation(
        raster=_build_learning_raster(state, params, current_node),
        local_raster=_build_learning_local_raster(state, params, current_node),
        structured=_build_learning_structured(state, params, current_xy),
        candidate_edges=candidate_edges.astype(jnp.float32),
        action_mask=action_mask,
    )


def _build_legacy_observation(state: EnvState, params: EnvParams) -> Observation:
    graph = params.graph
    current_car = jnp.clip(state.current_car_id, 0, params.max_cars - 1)
    current_node = jnp.clip(state.car_nodes[current_car], 0, graph.num_nodes - 1)
    decision = state.decision_required & (~state.done)

    action_mask = graph.outgoing_mask[current_node] & decision
    edge_ids = jnp.where(action_mask, graph.outgoing_edge_ids[current_node], 0)
    edge_targets = jnp.clip(graph.edge_targets[edge_ids], 0, graph.num_nodes - 1)
    current_lonlat = graph.node_lonlat[current_node]
    target_lonlat = graph.node_lonlat[edge_targets]
    current_xy = _normalized_lonlat(current_lonlat, params)
    target_xy = _normalized_lonlat(target_lonlat, params)
    delta_xy = target_xy - current_xy
    duration = _edge_travel_time_at(edge_ids, state.time_seconds, graph.edge_travel_time_s)
    length = graph.edge_lengths_m[edge_ids]
    hour = (jnp.floor(state.time_seconds / 3600.0).astype(jnp.int32)) % 24
    congestion = graph.edge_congestion[edge_ids, hour]
    route_features = _candidate_route_demand_features(
        state,
        params,
        current_node,
        edge_targets,
    )
    candidate_edges = jnp.stack(
        [
            delta_xy[:, 0],
            delta_xy[:, 1],
            target_xy[:, 0],
            target_xy[:, 1],
            length / 1000.0,
            duration / 60.0,
            congestion,
            action_mask.astype(jnp.float32),
        ],
        axis=-1,
    )
    candidate_edges = jnp.concatenate([candidate_edges, route_features], axis=-1)
    candidate_edges = jnp.where(action_mask[:, None], candidate_edges, 0.0)

    queued = state.request_status == _REQUEST_QUEUED_STATUS
    structured = jnp.asarray(
        [
            (state.time_seconds - params.start_time_seconds) / jnp.maximum(params.episode_seconds, 1.0),
            current_car.astype(jnp.float32) / jnp.maximum(params.max_cars - 1, 1),
            current_xy[0],
            current_xy[1],
            queued.sum().astype(jnp.float32) / jnp.maximum(params.max_requests, 1),
            (state.car_status == _CAR_DECISION_STATUS).sum().astype(jnp.float32)
            / jnp.maximum(params.max_cars, 1),
            params.spawn_rate_per_minute,
            state.metrics.completed_requests.astype(jnp.float32),
            state.metrics.dropped_requests.astype(jnp.float32),
            state.metrics.invalid_actions.astype(jnp.float32),
        ],
        dtype=jnp.float32,
    )
    return Observation(
        raster=_build_legacy_raster(state, params, current_node),
        local_raster=_build_legacy_local_raster(state, params, current_node),
        structured=structured,
        candidate_edges=candidate_edges.astype(jnp.float32),
        action_mask=action_mask,
    )


def _build_learning_structured(state: EnvState, params: EnvParams, current_xy):
    queued = state.request_status == _REQUEST_QUEUED_STATUS
    queued_count = queued.sum().astype(jnp.float32)
    wait_minutes = jnp.where(
        queued,
        jnp.maximum(0.0, state.time_seconds - state.request_spawn_times) / 60.0,
        0.0,
    )
    mean_wait = wait_minutes.sum() / jnp.maximum(queued_count, 1.0)
    max_wait = wait_minutes.max()
    elapsed_fraction = jnp.clip(
        (state.time_seconds - params.start_time_seconds) / jnp.maximum(params.episode_seconds, 1.0),
        0.0,
        1.0,
    )
    time_angle = 2.0 * jnp.pi * ((state.time_seconds / 86400.0) % 1.0)
    available_cars = (state.car_status == _CAR_DECISION_STATUS).sum().astype(jnp.float32)
    busy_cars = params.max_cars - available_cars
    return jnp.asarray(
        [
            elapsed_fraction,
            jnp.sin(time_angle),
            jnp.cos(time_angle),
            queued_count / jnp.maximum(params.max_requests, 1),
            available_cars / jnp.maximum(params.max_cars, 1),
            busy_cars / jnp.maximum(params.max_cars, 1),
            jnp.clip(params.spawn_rate_per_minute / _SPAWN_RATE_NORMALIZER_PER_MINUTE, 0.0, 1.0),
            jnp.clip(mean_wait / _WAIT_FEATURE_NORMALIZER_MINUTES, 0.0, 1.0),
            jnp.clip(max_wait / _WAIT_FEATURE_NORMALIZER_MINUTES, 0.0, 1.0),
            current_xy[0],
            current_xy[1],
        ],
        dtype=jnp.float32,
    )


def _build_learning_raster(state: EnvState, params: EnvParams, current_node):
    size = params.raster_size
    graph = params.graph
    raster = jnp.zeros((size, size, RASTER_CHANNELS), dtype=jnp.float32)
    car_rows, car_cols = _node_grid_indices(state.car_nodes, params)
    available_cars = state.car_status == _CAR_DECISION_STATUS
    raster = raster.at[car_rows, car_cols, RASTER_AVAILABLE_CARS].add(available_cars.astype(jnp.float32))

    request_mask = state.request_status == _REQUEST_QUEUED_STATUS
    request_nodes = jnp.clip(state.request_origin_nodes, 0, graph.num_nodes - 1)
    req_rows, req_cols = _node_grid_indices(request_nodes, params)
    wait_minutes = jnp.maximum(0.0, state.time_seconds - state.request_spawn_times) / 60.0
    raster = raster.at[req_rows, req_cols, RASTER_QUEUED_ORIGINS].add(request_mask.astype(jnp.float32))
    raster = raster.at[req_rows, req_cols, RASTER_QUEUED_WAIT_SUM_MIN].add(
        jnp.where(request_mask, wait_minutes / _WAIT_FEATURE_NORMALIZER_MINUTES, 0.0)
    )

    density_rows, density_cols = _node_grid_indices(jnp.arange(graph.num_nodes), params)
    expected_demand = _node_density_normalized_at(state.time_seconds + 10.0 * 60.0, params)
    raster = raster.at[density_rows, density_cols, RASTER_EXPECTED_DEMAND_10M].add(expected_demand)

    future_nodes = jnp.where(available_cars, state.car_nodes, state.car_target_nodes)
    future_rows, future_cols = _node_grid_indices(future_nodes, params)
    ready_delta = jnp.maximum(0.0, state.car_ready_times - state.time_seconds)
    future_0_5 = available_cars | (ready_delta <= 5.0 * 60.0)
    future_5_15 = (~available_cars) & (ready_delta > 5.0 * 60.0) & (ready_delta <= 15.0 * 60.0)
    raster = raster.at[future_rows, future_cols, RASTER_FUTURE_SUPPLY_0_5M].add(
        future_0_5.astype(jnp.float32)
    )
    raster = raster.at[future_rows, future_cols, RASTER_FUTURE_SUPPLY_5_15M].add(
        future_5_15.astype(jnp.float32)
    )

    active = (
        (state.request_status == _REQUEST_QUEUED_STATUS)
        | (state.request_status == _REQUEST_ASSIGNED_STATUS)
        | (state.request_status == _REQUEST_ONBOARD_STATUS)
    )
    dest_nodes = jnp.clip(state.request_dest_nodes, 0, graph.num_nodes - 1)
    dest_rows, dest_cols = _node_grid_indices(dest_nodes, params)
    raster = raster.at[dest_rows, dest_cols, RASTER_ACTIVE_DESTINATIONS].add(active.astype(jnp.float32))

    focus_row, focus_col = _node_grid_indices(jnp.asarray([current_node], dtype=jnp.int32), params)
    raster = raster.at[focus_row[0], focus_col[0], RASTER_FOCUS_CAR].set(1.0)
    return jnp.clip(raster, 0.0, 10.0)


def _build_learning_local_raster(state: EnvState, params: EnvParams, current_node):
    size = params.raster_size
    graph = params.graph
    raster = jnp.zeros((size, size, RASTER_CHANNELS), dtype=jnp.float32)
    center = graph.node_lonlat[jnp.clip(current_node, 0, graph.num_nodes - 1)]
    half_lon, half_lat = _local_half_extents(params, size)

    car_lonlat = graph.node_lonlat[jnp.clip(state.car_nodes, 0, graph.num_nodes - 1)]
    car_rows, car_cols, car_mask = _local_lonlat_grid_indices(car_lonlat, center, half_lon, half_lat, size)
    available_cars = (state.car_status == _CAR_DECISION_STATUS) & car_mask
    raster = raster.at[car_rows, car_cols, RASTER_AVAILABLE_CARS].add(available_cars.astype(jnp.float32))

    request_mask = state.request_status == _REQUEST_QUEUED_STATUS
    request_nodes = jnp.clip(state.request_origin_nodes, 0, graph.num_nodes - 1)
    req_rows, req_cols, req_in_view = _local_lonlat_grid_indices(
        graph.node_lonlat[request_nodes],
        center,
        half_lon,
        half_lat,
        size,
    )
    wait_minutes = jnp.maximum(0.0, state.time_seconds - state.request_spawn_times) / 60.0
    visible_requests = request_mask & req_in_view
    raster = raster.at[req_rows, req_cols, RASTER_QUEUED_ORIGINS].add(visible_requests.astype(jnp.float32))
    raster = raster.at[req_rows, req_cols, RASTER_QUEUED_WAIT_SUM_MIN].add(
        jnp.where(visible_requests, wait_minutes / _WAIT_FEATURE_NORMALIZER_MINUTES, 0.0)
    )

    node_rows, node_cols, node_in_view = _local_lonlat_grid_indices(
        graph.node_lonlat,
        center,
        half_lon,
        half_lat,
        size,
    )
    expected_demand = _node_density_normalized_at(state.time_seconds + 10.0 * 60.0, params)
    raster = raster.at[node_rows, node_cols, RASTER_EXPECTED_DEMAND_10M].add(
        jnp.where(node_in_view, expected_demand, 0.0)
    )

    future_nodes = jnp.where(state.car_status == _CAR_DECISION_STATUS, state.car_nodes, state.car_target_nodes)
    future_lonlat = graph.node_lonlat[jnp.clip(future_nodes, 0, graph.num_nodes - 1)]
    future_rows, future_cols, future_in_view = _local_lonlat_grid_indices(
        future_lonlat,
        center,
        half_lon,
        half_lat,
        size,
    )
    ready_delta = jnp.maximum(0.0, state.car_ready_times - state.time_seconds)
    future_0_5 = ((state.car_status == _CAR_DECISION_STATUS) | (ready_delta <= 5.0 * 60.0)) & future_in_view
    future_5_15 = (
        (state.car_status != _CAR_DECISION_STATUS)
        & (ready_delta > 5.0 * 60.0)
        & (ready_delta <= 15.0 * 60.0)
        & future_in_view
    )
    raster = raster.at[future_rows, future_cols, RASTER_FUTURE_SUPPLY_0_5M].add(
        future_0_5.astype(jnp.float32)
    )
    raster = raster.at[future_rows, future_cols, RASTER_FUTURE_SUPPLY_5_15M].add(
        future_5_15.astype(jnp.float32)
    )

    active = (
        (state.request_status == _REQUEST_QUEUED_STATUS)
        | (state.request_status == _REQUEST_ASSIGNED_STATUS)
        | (state.request_status == _REQUEST_ONBOARD_STATUS)
    )
    dest_nodes = jnp.clip(state.request_dest_nodes, 0, graph.num_nodes - 1)
    dest_rows, dest_cols, dest_in_view = _local_lonlat_grid_indices(
        graph.node_lonlat[dest_nodes],
        center,
        half_lon,
        half_lat,
        size,
    )
    raster = raster.at[dest_rows, dest_cols, RASTER_ACTIVE_DESTINATIONS].add(
        (active & dest_in_view).astype(jnp.float32)
    )

    center_row = jnp.asarray(size // 2, dtype=jnp.int32)
    center_col = jnp.asarray(size // 2, dtype=jnp.int32)
    raster = raster.at[center_row, center_col, RASTER_FOCUS_CAR].set(1.0)
    return jnp.clip(raster, 0.0, 10.0)


def _candidate_marginal_value_features(
    state: EnvState,
    params: EnvParams,
    current_car,
    current_node,
    edge_targets,
    edge_durations_s,
):
    graph = params.graph
    request_origins = jnp.clip(state.request_origin_nodes, 0, graph.num_nodes - 1)
    queued = state.request_status == _REQUEST_QUEUED_STATUS
    queued_float = queued.astype(jnp.float32)
    count_norm = jnp.maximum(params.max_requests, 1)
    wait_minutes = jnp.where(
        queued,
        jnp.maximum(0.0, state.time_seconds - state.request_spawn_times) / 60.0,
        0.0,
    )
    wait_weight = jnp.clip(wait_minutes / _WAIT_FEATURE_NORMALIZER_MINUTES, 0.0, 1.0)

    target_eta_s = graph.routing_travel_time_s[edge_targets[:, None], request_origins[None, :]]
    current_eta_s = graph.routing_travel_time_s[current_node, request_origins]
    our_eta_after_s = edge_durations_s[:, None] + target_eta_s
    target_eta_valid = queued[None, :] & jnp.isfinite(target_eta_s) & jnp.isfinite(our_eta_after_s)

    reachable_5 = target_eta_valid & (target_eta_s <= 5.0 * 60.0)
    reachable_10 = target_eta_valid & (target_eta_s <= 10.0 * 60.0)
    reachable_5_count = reachable_5.astype(jnp.float32).sum(axis=1) / count_norm
    reachable_10_count = reachable_10.astype(jnp.float32).sum(axis=1) / count_norm
    wait_weighted_reachable = (reachable_10.astype(jnp.float32) * wait_weight[None, :]).sum(axis=1) / count_norm
    max_reachable_wait = jnp.where(reachable_10, wait_minutes[None, :], 0.0).max(axis=1)
    max_reachable_wait = jnp.clip(max_reachable_wait / _WAIT_FEATURE_NORMALIZER_MINUTES, 0.0, 1.0)
    min_eta_s = jnp.where(target_eta_valid, target_eta_s, jnp.inf).min(axis=1)
    min_eta = jnp.where(
        jnp.isfinite(min_eta_s),
        jnp.clip((min_eta_s / 60.0) / _ETA_FEATURE_NORMALIZER_MINUTES, 0.0, 1.0),
        0.0,
    )

    # The edge duration is already an explicit feature. This improvement measures
    # the marginal value of the resulting target node; subtracting the edge
    # duration here would collapse shortest-path moves to zero improvement.
    improvement_s = current_eta_s[None, :] - target_eta_s
    improvement_valid = target_eta_valid & jnp.isfinite(current_eta_s)[None, :]
    positive_improvement_min = jnp.where(
        improvement_valid,
        jnp.maximum(improvement_s, 0.0) / 60.0,
        0.0,
    )
    eta_improvement = positive_improvement_min.sum(axis=1) / (
        count_norm * _ETA_FEATURE_NORMALIZER_MINUTES
    )
    wait_weighted_eta_improvement = (positive_improvement_min * wait_weight[None, :]).sum(axis=1) / (
        count_norm * _ETA_FEATURE_NORMALIZER_MINUTES
    )

    best_other_eta_s = _best_other_available_eta_to_requests(state, params, current_car, request_origins, queued)
    has_other = jnp.isfinite(best_other_eta_s)
    advantage_s = jnp.where(
        has_other[None, :],
        best_other_eta_s[None, :] - our_eta_after_s,
        _ETA_FEATURE_NORMALIZER_SECONDS,
    )
    advantage_valid = target_eta_valid & jnp.isfinite(our_eta_after_s)
    positive_advantage_min = jnp.where(
        advantage_valid,
        jnp.maximum(advantage_s, 0.0) / 60.0,
        0.0,
    )
    best_after_edge_count = (advantage_valid & (advantage_s > 0.0)).astype(jnp.float32).sum(axis=1) / count_norm
    wait_weighted_advantage = (positive_advantage_min * wait_weight[None, :]).sum(axis=1) / (
        count_norm * _ETA_FEATURE_NORMALIZER_MINUTES
    )

    expected_demand_10 = _node_density_normalized_at(state.time_seconds + 10.0 * 60.0, params)[edge_targets]
    expected_demand_30 = _node_density_normalized_at(state.time_seconds + 30.0 * 60.0, params)[edge_targets]
    future_supply_0_5, future_supply_5_15 = _future_supply_at_targets(
        state,
        params,
        current_car,
        edge_targets,
    )
    imbalance = jnp.clip(expected_demand_10 - future_supply_0_5, -1.0, 1.0)

    features = jnp.stack(
        [
            reachable_5_count,
            reachable_10_count,
            wait_weighted_reachable,
            max_reachable_wait,
            min_eta,
            eta_improvement,
            wait_weighted_eta_improvement,
            best_after_edge_count,
            wait_weighted_advantage,
            expected_demand_10,
            expected_demand_30,
            future_supply_0_5,
            future_supply_5_15,
            imbalance,
        ],
        axis=-1,
    )
    features = jnp.clip(features, -10.0, 10.0)
    return jnp.where(jnp.isfinite(features), features, 0.0)


def _best_other_available_eta_to_requests(state: EnvState, params: EnvParams, current_car, request_origins, queued):
    graph = params.graph
    car_ids = jnp.arange(params.max_cars, dtype=jnp.int32)
    other_available = (state.car_status == _CAR_DECISION_STATUS) & (car_ids != current_car)
    other_nodes = jnp.clip(state.car_nodes, 0, graph.num_nodes - 1)
    other_eta_s = graph.routing_travel_time_s[other_nodes[:, None], request_origins[None, :]]
    valid = other_available[:, None] & queued[None, :] & jnp.isfinite(other_eta_s)

    radius = int(params.assignment_max_route_edges)
    if radius < max(0, graph.num_nodes - 1):
        horizon = max(1, min(max(1, radius) * 2, _ROUTE_FEATURE_MAX_HOPS))
        other_hops = _route_hops_within(
            graph,
            other_nodes[:, None],
            request_origins[None, :],
            horizon,
        )
        valid = valid & (other_hops <= radius)

    return jnp.where(valid, other_eta_s, jnp.inf).min(axis=0)


def _future_supply_at_targets(state: EnvState, params: EnvParams, current_car, edge_targets):
    car_ids = jnp.arange(params.max_cars, dtype=jnp.int32)
    other_car = car_ids != current_car
    available = state.car_status == _CAR_DECISION_STATUS
    future_nodes = jnp.where(available, state.car_nodes, state.car_target_nodes)
    future_nodes = jnp.clip(future_nodes, 0, params.graph.num_nodes - 1)
    ready_delta = jnp.maximum(0.0, state.car_ready_times - state.time_seconds)
    future_0_5 = other_car & (available | (ready_delta <= 5.0 * 60.0))
    future_5_15 = other_car & (~available) & (ready_delta > 5.0 * 60.0) & (ready_delta <= 15.0 * 60.0)
    at_target = future_nodes[None, :] == edge_targets[:, None]
    supply_0_5 = (at_target & future_0_5[None, :]).astype(jnp.float32).sum(axis=1)
    supply_5_15 = (at_target & future_5_15[None, :]).astype(jnp.float32).sum(axis=1)
    return (
        supply_0_5 / jnp.maximum(params.max_cars, 1),
        supply_5_15 / jnp.maximum(params.max_cars, 1),
    )


def _build_legacy_raster(state: EnvState, params: EnvParams, current_node):
    size = params.raster_size
    raster = jnp.zeros((size, size, LEGACY_RASTER_CHANNELS), dtype=jnp.float32)
    car_rows, car_cols = _node_grid_indices(state.car_nodes, params)
    available_cars = state.car_status == _CAR_DECISION_STATUS
    busy_cars = ~available_cars
    raster = raster.at[car_rows, car_cols, LEGACY_AVAILABLE_CAR_CHANNEL].add(available_cars.astype(jnp.float32))
    raster = raster.at[car_rows, car_cols, LEGACY_BUSY_CAR_CHANNEL].add(busy_cars.astype(jnp.float32))

    request_mask = state.request_status == _REQUEST_QUEUED_STATUS
    request_nodes = jnp.clip(state.request_origin_nodes, 0, params.graph.num_nodes - 1)
    req_rows, req_cols = _node_grid_indices(request_nodes, params)
    raster = raster.at[req_rows, req_cols, LEGACY_REQUEST_CHANNEL].add(request_mask.astype(jnp.float32))

    hour = (jnp.floor(state.time_seconds / 3600.0).astype(jnp.int32)) % 24
    raster = raster.at[:, :, LEGACY_CONGESTION_CHANNEL].set(params.edge_raster_by_hour[hour])
    density_rows, density_cols = _node_grid_indices(jnp.arange(params.graph.num_nodes), params)
    density = _node_density_normalized_at(state.time_seconds, params)
    raster = raster.at[density_rows, density_cols, LEGACY_DENSITY_CHANNEL].add(density)

    focus_row, focus_col = _node_grid_indices(jnp.asarray([current_node], dtype=jnp.int32), params)
    raster = raster.at[focus_row[0], focus_col[0], LEGACY_FOCUS_CAR_CHANNEL].set(1.0)
    return jnp.clip(raster, 0.0, 10.0)


def _build_legacy_local_raster(state: EnvState, params: EnvParams, current_node):
    size = params.raster_size
    graph = params.graph
    raster = jnp.zeros((size, size, LEGACY_RASTER_CHANNELS), dtype=jnp.float32)
    center = graph.node_lonlat[jnp.clip(current_node, 0, graph.num_nodes - 1)]
    half_lon, half_lat = _local_half_extents(params, size)

    car_rows, car_cols, car_mask = _local_lonlat_grid_indices(
        graph.node_lonlat[jnp.clip(state.car_nodes, 0, graph.num_nodes - 1)],
        center,
        half_lon,
        half_lat,
        size,
    )
    available_cars = (state.car_status == _CAR_DECISION_STATUS) & car_mask
    busy_cars = (state.car_status != _CAR_DECISION_STATUS) & car_mask
    raster = raster.at[car_rows, car_cols, LEGACY_AVAILABLE_CAR_CHANNEL].add(available_cars.astype(jnp.float32))
    raster = raster.at[car_rows, car_cols, LEGACY_BUSY_CAR_CHANNEL].add(busy_cars.astype(jnp.float32))

    request_mask = state.request_status == _REQUEST_QUEUED_STATUS
    request_nodes = jnp.clip(state.request_origin_nodes, 0, graph.num_nodes - 1)
    req_rows, req_cols, req_in_view = _local_lonlat_grid_indices(
        graph.node_lonlat[request_nodes],
        center,
        half_lon,
        half_lat,
        size,
    )
    raster = raster.at[req_rows, req_cols, LEGACY_REQUEST_CHANNEL].add(
        (request_mask & req_in_view).astype(jnp.float32)
    )

    hour = (jnp.floor(state.time_seconds / 3600.0).astype(jnp.int32)) % 24
    edge_midpoints = (
        graph.node_lonlat[jnp.clip(graph.edge_sources, 0, graph.num_nodes - 1)]
        + graph.node_lonlat[jnp.clip(graph.edge_targets, 0, graph.num_nodes - 1)]
    ) * 0.5
    edge_rows, edge_cols, edge_in_view = _local_lonlat_grid_indices(
        edge_midpoints,
        center,
        half_lon,
        half_lat,
        size,
    )
    edge_congestion = jnp.where(edge_in_view, graph.edge_congestion[:, hour], 0.0)
    raster = raster.at[edge_rows, edge_cols, LEGACY_CONGESTION_CHANNEL].max(edge_congestion)

    node_rows, node_cols, node_in_view = _local_lonlat_grid_indices(
        graph.node_lonlat,
        center,
        half_lon,
        half_lat,
        size,
    )
    density = _node_density_normalized_at(state.time_seconds, params)
    raster = raster.at[node_rows, node_cols, LEGACY_DENSITY_CHANNEL].add(jnp.where(node_in_view, density, 0.0))

    center_row = jnp.asarray(size // 2, dtype=jnp.int32)
    center_col = jnp.asarray(size // 2, dtype=jnp.int32)
    raster = raster.at[center_row, center_col, LEGACY_FOCUS_CAR_CHANNEL].set(1.0)
    return jnp.clip(raster, 0.0, 10.0)


def _candidate_route_demand_features(
    state: EnvState,
    params: EnvParams,
    current_node,
    edge_targets,
):
    graph = params.graph
    radius = int(params.assignment_max_route_edges)
    global_radius = radius >= max(0, graph.num_nodes - 1)
    horizon = max(1, min(max(1, radius) * 2, _ROUTE_FEATURE_MAX_HOPS))
    request_origins = jnp.clip(state.request_origin_nodes, 0, graph.num_nodes - 1)
    queued = state.request_status == _REQUEST_QUEUED_STATUS
    has_queued = queued.any()
    current_sources = jnp.full((params.max_requests,), current_node, dtype=jnp.int32)
    current_hops = _route_hops_within(graph, current_sources, request_origins, horizon)
    next_sources = jnp.broadcast_to(edge_targets[:, None], (graph.max_degree, params.max_requests))
    next_targets = jnp.broadcast_to(request_origins[None, :], (graph.max_degree, params.max_requests))
    next_hops = _route_hops_within(graph, next_sources, next_targets, horizon)

    current_eta = graph.routing_travel_time_s[current_sources, request_origins]
    next_eta = graph.routing_travel_time_s[next_sources, next_targets]
    current_hops_masked = jnp.where(queued, current_hops, horizon + 1)
    next_hops_masked = jnp.where(queued[None, :], next_hops, horizon + 1)
    current_eta_masked = jnp.where(queued & jnp.isfinite(current_eta), current_eta, jnp.inf)
    next_eta_masked = jnp.where(queued[None, :] & jnp.isfinite(next_eta), next_eta, jnp.inf)

    current_min_hops = current_hops_masked.min()
    next_min_hops = next_hops_masked.min(axis=1)
    current_min_eta = current_eta_masked.min()
    next_min_eta = next_eta_masked.min(axis=1)
    delta_hops = (current_min_hops.astype(jnp.float32) - next_min_hops.astype(jnp.float32)) / float(horizon)
    delta_eta_minutes = (current_min_eta - next_min_eta) / 60.0
    delta_hops = jnp.where(has_queued, delta_hops, 0.0)
    delta_eta_minutes = jnp.where(has_queued & jnp.isfinite(delta_eta_minutes), delta_eta_minutes, 0.0)

    if global_radius:
        within_radius = queued[None, :] & jnp.isfinite(next_eta)
    else:
        within_radius = queued[None, :] & (next_hops <= radius)
    requests_within_radius = within_radius.astype(jnp.float32).sum(axis=1) / jnp.maximum(
        params.max_requests,
        1,
    )
    wait_minutes = jnp.maximum(0.0, state.time_seconds - state.request_spawn_times) / 60.0
    wait_weighted_demand = (within_radius.astype(jnp.float32) * wait_minutes[None, :]).sum(axis=1)
    wait_weighted_demand = wait_weighted_demand / jnp.maximum(params.max_requests, 1)

    return jnp.stack(
        [
            delta_hops,
            delta_eta_minutes,
            requests_within_radius,
            wait_weighted_demand,
        ],
        axis=-1,
    )


def _route_hops_within(graph, source_nodes, target_nodes, max_hops: int):
    current = jnp.clip(source_nodes, 0, graph.num_nodes - 1)
    target = jnp.clip(target_nodes, 0, graph.num_nodes - 1)
    current, target = jnp.broadcast_arrays(current, target)
    reached = current == target
    unresolved = jnp.asarray(max_hops + 1, dtype=jnp.int32)
    hops = jnp.where(reached, jnp.asarray(0, dtype=jnp.int32), unresolved)

    def body(i, carry):
        node, done, hop_count = carry
        edge_id = graph.routing_next_edge[node, target]
        can_step = (~done) & (edge_id >= 0)
        next_node = graph.edge_targets[jnp.clip(edge_id, 0, graph.num_edges - 1)]
        node = jnp.where(can_step, next_node, node)
        now_reached = can_step & (node == target)
        hop_count = jnp.where(now_reached, i + jnp.asarray(1, dtype=jnp.int32), hop_count)
        done = done | now_reached
        return node, done, hop_count

    _, _, hops = jax.lax.fori_loop(0, max_hops, body, (current, reached, hops))
    return hops


def _node_density_at(time_seconds, params: EnvParams):
    hour_float = (time_seconds / 3600.0) % 24.0
    h0 = jnp.floor(hour_float).astype(jnp.int32)
    h1 = (h0 + 1) % 24
    frac = hour_float - jnp.floor(hour_float)
    weights = params.node_density_by_hour[h0] * (1.0 - frac) + params.node_density_by_hour[h1] * frac
    return jnp.maximum(weights, 1.0e-6)


def _node_density_normalized_at(time_seconds, params: EnvParams):
    density = _node_density_at(time_seconds, params)
    return density / jnp.maximum(density.max(), 1.0e-6)


def _edge_travel_time_at(edge_ids, time_seconds, edge_travel_time_s):
    edge_ids = jnp.clip(edge_ids, 0, edge_travel_time_s.shape[0] - 1)
    hour_float = (time_seconds / 3600.0) % 24.0
    h0 = jnp.floor(hour_float).astype(jnp.int32)
    h1 = (h0 + 1) % 24
    frac = hour_float - jnp.floor(hour_float)
    return edge_travel_time_s[edge_ids, h0] * (1.0 - frac) + edge_travel_time_s[edge_ids, h1] * frac


def _node_grid_indices(nodes, params: EnvParams):
    lonlat = params.graph.node_lonlat[jnp.clip(nodes, 0, params.graph.num_nodes - 1)]
    return _lonlat_grid_indices(lonlat, params)


def _lonlat_grid_indices(lonlat, params: EnvParams):
    xy = _normalized_lonlat(lonlat, params)
    cols = jnp.floor(xy[..., 0] * params.raster_size).astype(jnp.int32)
    rows = jnp.floor(xy[..., 1] * params.raster_size).astype(jnp.int32)
    rows = jnp.clip(rows, 0, params.raster_size - 1)
    cols = jnp.clip(cols, 0, params.raster_size - 1)
    return rows, cols


def _normalized_lonlat(lonlat, params: EnvParams):
    min_lon, min_lat, max_lon, max_lat = params.graph.bounds
    span_lon = jnp.maximum(max_lon - min_lon, 1e-6)
    span_lat = jnp.maximum(max_lat - min_lat, 1e-6)
    x = (lonlat[..., 0] - min_lon) / span_lon
    y = (lonlat[..., 1] - min_lat) / span_lat
    return jnp.stack([jnp.clip(x, 0.0, 1.0), jnp.clip(y, 0.0, 1.0)], axis=-1)


def _local_half_extents(params: EnvParams, size: int):
    min_lon, min_lat, max_lon, max_lat = params.graph.bounds
    block_lon = jnp.maximum(max_lon - min_lon, 1.0e-6) / jnp.maximum(size, 1)
    block_lat = jnp.maximum(max_lat - min_lat, 1.0e-6) / jnp.maximum(size, 1)
    return block_lon * _LOCAL_VIEW_HALF_WIDTH_BLOCKS, block_lat * _LOCAL_VIEW_HALF_WIDTH_BLOCKS


def _local_lonlat_grid_indices(lonlat, center, half_lon, half_lat, size: int):
    x = (lonlat[..., 0] - center[0]) / jnp.maximum(half_lon * 2.0, 1.0e-9) + 0.5
    y = (lonlat[..., 1] - center[1]) / jnp.maximum(half_lat * 2.0, 1.0e-9) + 0.5
    in_view = (x >= 0.0) & (x < 1.0) & (y >= 0.0) & (y < 1.0)
    cols = jnp.floor(x * size).astype(jnp.int32)
    rows = jnp.floor(y * size).astype(jnp.int32)
    rows = jnp.clip(rows, 0, size - 1)
    cols = jnp.clip(cols, 0, size - 1)
    return rows, cols, in_view
