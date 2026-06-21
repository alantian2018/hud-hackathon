from __future__ import annotations

import jax.numpy as jnp

from jax_fleet.types import EnvParams, EnvState, Observation


def build_observation(state: EnvState, params: EnvParams) -> Observation:
    graph = params.graph
    current_car = jnp.clip(state.current_car_id, 0, params.max_cars - 1)
    current_node = state.car_nodes[current_car]
    current_node = jnp.clip(current_node, 0, graph.num_nodes - 1)
    decision = state.decision_required & (~state.done)

    action_mask = graph.outgoing_mask[current_node] & decision
    edge_ids = jnp.where(action_mask, graph.outgoing_edge_ids[current_node], 0)
    edge_targets = jnp.clip(graph.edge_targets[edge_ids], 0, graph.num_nodes - 1)
    target_lonlat = graph.node_lonlat[edge_targets]
    duration = _edge_travel_time_at(edge_ids, state.time_seconds, graph.edge_travel_time_s)
    length = graph.edge_lengths_m[edge_ids]
    hour = (jnp.floor(state.time_seconds / 3600.0).astype(jnp.int32)) % 24
    congestion = graph.edge_congestion[edge_ids, hour]
    candidate_edges = jnp.stack(
        [
            target_lonlat[:, 0],
            target_lonlat[:, 1],
            length / 1000.0,
            duration / 60.0,
            congestion,
            action_mask.astype(jnp.float32),
        ],
        axis=-1,
    )
    candidate_edges = jnp.where(action_mask[:, None], candidate_edges, 0.0)

    queued = state.request_status == 1
    day_phase = ((state.time_seconds / 3600.0) % 24.0) / 24.0
    time_sin = jnp.sin(2.0 * jnp.pi * day_phase)
    time_cos = jnp.cos(2.0 * jnp.pi * day_phase)
    structured = jnp.asarray(
        [
            (state.time_seconds - params.start_time_seconds) / jnp.maximum(params.episode_seconds, 1.0),
            time_sin,
            time_cos,
            current_car.astype(jnp.float32) / jnp.maximum(params.max_cars - 1, 1),
            queued.sum().astype(jnp.float32) / jnp.maximum(params.max_requests, 1),
            (state.car_status == 0).sum().astype(jnp.float32) / jnp.maximum(params.max_cars, 1),
            params.spawn_rate_per_minute,
            state.metrics.completed_requests.astype(jnp.float32),
            state.metrics.dropped_requests.astype(jnp.float32),
            state.metrics.invalid_actions.astype(jnp.float32),
        ],
        dtype=jnp.float32,
    )
    return Observation(
        raster=_build_raster(state, params),
        structured=structured,
        candidate_edges=candidate_edges.astype(jnp.float32),
        action_mask=action_mask,
    )


def _build_raster(state: EnvState, params: EnvParams):
    size = params.raster_size
    raster = jnp.zeros((size, size, 3), dtype=jnp.float32)
    car_rows, car_cols = _node_grid_indices(state.car_nodes, params)
    raster = raster.at[car_rows, car_cols, 0].add(1.0)

    request_mask = state.request_status == 1
    request_nodes = jnp.clip(state.request_origin_nodes, 0, params.graph.num_nodes - 1)
    req_rows, req_cols = _node_grid_indices(request_nodes, params)
    raster = raster.at[req_rows, req_cols, 1].add(request_mask.astype(jnp.float32))

    hour = (jnp.floor(state.time_seconds / 3600.0).astype(jnp.int32)) % 24
    raster = raster.at[:, :, 2].set(params.edge_raster_by_hour[hour])
    return jnp.clip(raster, 0.0, 10.0)


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
    min_lon, min_lat, max_lon, max_lat = params.graph.bounds
    span_lon = jnp.maximum(max_lon - min_lon, 1e-6)
    span_lat = jnp.maximum(max_lat - min_lat, 1e-6)
    cols = jnp.floor((lonlat[..., 0] - min_lon) / span_lon * params.raster_size).astype(jnp.int32)
    rows = jnp.floor((lonlat[..., 1] - min_lat) / span_lat * params.raster_size).astype(jnp.int32)
    rows = jnp.clip(rows, 0, params.raster_size - 1)
    cols = jnp.clip(cols, 0, params.raster_size - 1)
    return rows, cols
