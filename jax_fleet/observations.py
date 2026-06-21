from __future__ import annotations

import jax.numpy as jnp

from jax_fleet.types import EnvParams, EnvState, Observation


_LOCAL_VIEW_HALF_WIDTH_BLOCKS = 5.0


def build_observation(state: EnvState, params: EnvParams) -> Observation:
    graph = params.graph
    current_car = jnp.clip(state.current_car_id, 0, params.max_cars - 1)
    current_node = state.car_nodes[current_car]
    current_node = jnp.clip(current_node, 0, graph.num_nodes - 1)
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
    candidate_edges = jnp.where(action_mask[:, None], candidate_edges, 0.0)

    queued = state.request_status == 1
    structured = jnp.asarray(
        [
            (state.time_seconds - params.start_time_seconds) / jnp.maximum(params.episode_seconds, 1.0),
            current_car.astype(jnp.float32) / jnp.maximum(params.max_cars - 1, 1),
            current_xy[0],
            current_xy[1],
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
        raster=_build_raster(state, params, current_node),
        local_raster=_build_local_raster(state, params, current_node),
        structured=structured,
        candidate_edges=candidate_edges.astype(jnp.float32),
        action_mask=action_mask,
    )


def _build_raster(state: EnvState, params: EnvParams, current_node):
    size = params.raster_size
    raster = jnp.zeros((size, size, 5), dtype=jnp.float32)
    car_rows, car_cols = _node_grid_indices(state.car_nodes, params)
    raster = raster.at[car_rows, car_cols, 0].add(1.0)

    request_mask = state.request_status == 1
    request_nodes = jnp.clip(state.request_origin_nodes, 0, params.graph.num_nodes - 1)
    req_rows, req_cols = _node_grid_indices(request_nodes, params)
    raster = raster.at[req_rows, req_cols, 1].add(request_mask.astype(jnp.float32))

    hour = (jnp.floor(state.time_seconds / 3600.0).astype(jnp.int32)) % 24
    raster = raster.at[:, :, 2].set(params.edge_raster_by_hour[hour])
    density_rows, density_cols = _node_grid_indices(jnp.arange(params.graph.num_nodes), params)
    density = _node_density_at(state.time_seconds, params)
    density = density / jnp.maximum(density.max(), 1.0e-6)
    raster = raster.at[density_rows, density_cols, 3].add(density)

    focus_row, focus_col = _node_grid_indices(jnp.asarray([current_node], dtype=jnp.int32), params)
    raster = raster.at[focus_row[0], focus_col[0], 4].set(1.0)
    return jnp.clip(raster, 0.0, 10.0)


def _build_local_raster(state: EnvState, params: EnvParams, current_node):
    size = params.raster_size
    graph = params.graph
    raster = jnp.zeros((size, size, 5), dtype=jnp.float32)
    center = graph.node_lonlat[jnp.clip(current_node, 0, graph.num_nodes - 1)]
    min_lon, min_lat, max_lon, max_lat = graph.bounds
    block_lon = jnp.maximum(max_lon - min_lon, 1.0e-6) / jnp.maximum(size, 1)
    block_lat = jnp.maximum(max_lat - min_lat, 1.0e-6) / jnp.maximum(size, 1)
    half_lon = block_lon * _LOCAL_VIEW_HALF_WIDTH_BLOCKS
    half_lat = block_lat * _LOCAL_VIEW_HALF_WIDTH_BLOCKS

    car_rows, car_cols, car_mask = _local_lonlat_grid_indices(
        graph.node_lonlat[jnp.clip(state.car_nodes, 0, graph.num_nodes - 1)],
        center,
        half_lon,
        half_lat,
        size,
    )
    raster = raster.at[car_rows, car_cols, 0].add(car_mask.astype(jnp.float32))

    request_mask = state.request_status == 1
    request_nodes = jnp.clip(state.request_origin_nodes, 0, graph.num_nodes - 1)
    req_rows, req_cols, req_in_view = _local_lonlat_grid_indices(
        graph.node_lonlat[request_nodes],
        center,
        half_lon,
        half_lat,
        size,
    )
    raster = raster.at[req_rows, req_cols, 1].add((request_mask & req_in_view).astype(jnp.float32))

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
    raster = raster.at[edge_rows, edge_cols, 2].max(edge_congestion)

    node_rows, node_cols, node_in_view = _local_lonlat_grid_indices(
        graph.node_lonlat,
        center,
        half_lon,
        half_lat,
        size,
    )
    density = _node_density_at(state.time_seconds, params)
    density = density / jnp.maximum(density.max(), 1.0e-6)
    raster = raster.at[node_rows, node_cols, 3].add(jnp.where(node_in_view, density, 0.0))

    center_row = jnp.asarray(size // 2, dtype=jnp.int32)
    center_col = jnp.asarray(size // 2, dtype=jnp.int32)
    raster = raster.at[center_row, center_col, 4].set(1.0)
    return jnp.clip(raster, 0.0, 10.0)


def _node_density_at(time_seconds, params: EnvParams):
    hour_float = (time_seconds / 3600.0) % 24.0
    h0 = jnp.floor(hour_float).astype(jnp.int32)
    h1 = (h0 + 1) % 24
    frac = hour_float - jnp.floor(hour_float)
    weights = params.node_density_by_hour[h0] * (1.0 - frac) + params.node_density_by_hour[h1] * frac
    return jnp.maximum(weights, 1.0e-6)


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


def _local_lonlat_grid_indices(lonlat, center, half_lon, half_lat, size: int):
    x = (lonlat[..., 0] - center[0]) / jnp.maximum(half_lon * 2.0, 1.0e-9) + 0.5
    y = (lonlat[..., 1] - center[1]) / jnp.maximum(half_lat * 2.0, 1.0e-9) + 0.5
    in_view = (x >= 0.0) & (x < 1.0) & (y >= 0.0) & (y < 1.0)
    cols = jnp.floor(x * size).astype(jnp.int32)
    rows = jnp.floor(y * size).astype(jnp.int32)
    rows = jnp.clip(rows, 0, size - 1)
    cols = jnp.clip(cols, 0, size - 1)
    return rows, cols, in_view
