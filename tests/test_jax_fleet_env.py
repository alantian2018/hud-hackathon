from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_fleet.env import (
    CAR_DECISION,
    CAR_TO_PICKUP,
    REWARD_MODE_LEGACY_PICKUP_WAIT,
    REQUEST_ASSIGNED,
    REQUEST_DROPPED,
    REQUEST_COMPLETED,
    REQUEST_ONBOARD,
    REQUEST_QUEUED,
    _record_pickup_wait,
    make_env_params,
    nearest_eligible_car_by_eta,
    reset,
    step,
)
from jax_fleet.graph import build_synthetic_graph
from jax_fleet.heuristics import choose_marginal_value_action
from jax_fleet.observations import (
    CANDIDATE_EDGE_FEATURES,
    CE_BEST_AFTER_EDGE_COUNT,
    CE_ETA_IMPROVEMENT,
    CE_REACHABLE_5M,
    CE_VALID,
    CE_WAIT_WEIGHTED_ADVANTAGE,
    LEGACY_CANDIDATE_EDGE_FEATURES,
    LEGACY_DENSITY_CHANNEL,
    LEGACY_FOCUS_CAR_CHANNEL,
    LEGACY_RASTER_CHANNELS,
    LEGACY_STRUCTURED_FEATURES,
    RASTER_CHANNELS,
    RASTER_EXPECTED_DEMAND_10M,
    RASTER_FOCUS_CAR,
    STRUCT_CURRENT_X,
    STRUCT_CURRENT_Y,
    STRUCTURED_FEATURES,
    build_observation,
)
from jax_fleet.scene_export import export_scene


def tiny_graph():
    return build_synthetic_graph(
        node_lonlat=[
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (1.0, 1.0),
        ],
        edges=[
            {"source": 0, "target": 1, "travel_time_s": 5.0, "hourly_multiplier": {17: 3.0}},
            {"source": 1, "target": 0, "travel_time_s": 20.0},
            {"source": 0, "target": 2, "travel_time_s": 7.0},
            {"source": 2, "target": 1, "travel_time_s": 1.0},
            {"source": 1, "target": 3, "travel_time_s": 4.0},
            {"source": 1, "target": 2, "travel_time_s": 1.0},
            {"source": 3, "target": 0, "travel_time_s": 6.0},
            {"source": 2, "target": 3, "travel_time_s": 3.0},
            {"source": 3, "target": 2, "travel_time_s": 3.0},
        ],
    )


def _active_request_count(state) -> int:
    status = np.asarray(state.request_status)
    return int(
        np.isin(status, [REQUEST_QUEUED, REQUEST_ASSIGNED, REQUEST_ONBOARD]).sum()
    )


def test_reset_and_step_are_deterministic_for_fixed_seed() -> None:
    params = make_env_params(tiny_graph(), max_cars=2, max_requests=4, initial_car_nodes=[0, 2])
    key = jax.random.PRNGKey(7)

    state_a, ts_a = reset(key, params)
    state_b, ts_b = reset(key, params)
    next_a, out_a = step(state_a, jnp.int32(0), params)
    next_b, out_b = step(state_b, jnp.int32(0), params)

    np.testing.assert_array_equal(np.asarray(state_a.car_nodes), np.asarray(state_b.car_nodes))
    np.testing.assert_array_equal(np.asarray(next_a.car_nodes), np.asarray(next_b.car_nodes))
    assert float(out_a.reward) == float(out_b.reward)
    assert float(ts_a.dt_seconds) == float(ts_b.dt_seconds) == 0.0


def test_one_step_consumes_one_policy_decision_and_simultaneous_cars_are_ascending() -> None:
    params = make_env_params(tiny_graph(), max_cars=2, max_requests=4, initial_car_nodes=[0, 2])
    state, _ = reset(jax.random.PRNGKey(0), params)

    assert int(state.current_car_id) == 0
    state, ts0 = step(state, jnp.int32(0), params)
    assert int(state.current_car_id) == 1
    assert float(ts0.dt_seconds) == 0.0

    state, ts1 = step(state, jnp.int32(0), params)
    assert int(state.current_car_id) == 1
    assert float(ts1.dt_seconds) == 1.0


def test_no_policy_query_during_auto_pickup_and_dropoff() -> None:
    params = make_env_params(
        tiny_graph(),
        max_cars=1,
        max_requests=4,
        initial_car_nodes=[0],
        preplanned_requests=[{"spawn_time_s": 2.0, "origin": 1, "destination": 3}],
    )
    state, _ = reset(jax.random.PRNGKey(0), params)

    state, ts = step(state, jnp.int32(0), params)

    assert int(state.current_car_id) == 0
    assert float(state.time_seconds) == 9.0
    assert int(state.request_status[0]) == REQUEST_COMPLETED
    assert math.isclose(float(ts.reward), -3.0 / 60.0, rel_tol=1e-6)
    assert float(ts.dt_seconds) == 9.0


def test_pickup_wait_reward_is_emitted_before_dropoff() -> None:
    graph = build_synthetic_graph(
        node_lonlat=[
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
        ],
        edges=[
            {"source": 0, "target": 1, "travel_time_s": 5.0},
            {"source": 1, "target": 2, "travel_time_s": 100.0},
            {"source": 2, "target": 0, "travel_time_s": 1.0},
            {"source": 3, "target": 0, "travel_time_s": 6.0},
        ],
    )
    params = make_env_params(
        graph,
        max_cars=2,
        max_requests=4,
        initial_car_nodes=[0, 3],
        preplanned_requests=[{"spawn_time_s": 2.0, "origin": 1, "destination": 2}],
        episode_seconds=300.0,
    )
    state, _ = reset(jax.random.PRNGKey(0), params)

    state, ts = step(state, jnp.int32(0), params)
    assert int(state.current_car_id) == 1
    assert float(ts.reward) == 0.0

    state, ts = step(state, jnp.int32(0), params)

    assert float(state.time_seconds) == 6.0
    assert int(state.request_status[0]) == REQUEST_ONBOARD
    assert int(state.metrics.completed_requests) == 0
    assert int(state.metrics.recent_pickup_wait_count) == 1
    assert math.isclose(float(ts.reward), -3.0 / 60.0, rel_tol=1e-6)
    assert math.isclose(float(state.metrics.aggregate_reward), -3.0 / 60.0, rel_tol=1e-6)


def test_directed_nearest_car_assignment_uses_eta_not_node_id() -> None:
    params = make_env_params(tiny_graph(), max_cars=2, max_requests=4, initial_car_nodes=[0, 2])
    state, _ = reset(jax.random.PRNGKey(0), params)
    request_id = jnp.int32(0)
    state = state.replace(
        request_status=state.request_status.at[request_id].set(REQUEST_QUEUED),
        request_origin_nodes=state.request_origin_nodes.at[request_id].set(1),
        request_dest_nodes=state.request_dest_nodes.at[request_id].set(3),
        request_spawn_times=state.request_spawn_times.at[request_id].set(0.0),
    )

    chosen = nearest_eligible_car_by_eta(state, request_id, params)

    assert int(chosen) == 1


def test_candidate_edges_include_route_demand_features() -> None:
    params = make_env_params(
        tiny_graph(),
        max_cars=1,
        max_requests=4,
        initial_car_nodes=[0],
        assignment_max_route_edges=2,
        observation_mode="legacy",
    )
    state, _ = reset(jax.random.PRNGKey(0), params)
    state = state.replace(
        time_seconds=jnp.asarray(120.0, dtype=jnp.float32),
        request_status=state.request_status.at[0].set(REQUEST_QUEUED),
        request_origin_nodes=state.request_origin_nodes.at[0].set(1),
        request_dest_nodes=state.request_dest_nodes.at[0].set(3),
        request_spawn_times=state.request_spawn_times.at[0].set(0.0),
    )

    obs = build_observation(state, params)

    assert obs.candidate_edges.shape == (params.graph.max_degree, LEGACY_CANDIDATE_EDGE_FEATURES)
    mean_route_eta_seconds = (23.0 * 5.0 + 15.0) / 24.0
    np.testing.assert_allclose(
        np.asarray(obs.candidate_edges[0, 8:12]),
        [0.25, mean_route_eta_seconds / 60.0, 0.25, 0.5],
        rtol=1e-6,
    )


def test_candidate_edges_expose_marginal_value_signal_for_request() -> None:
    params = make_env_params(
        tiny_graph(),
        max_cars=1,
        max_requests=4,
        initial_car_nodes=[0],
    )
    state, _ = reset(jax.random.PRNGKey(0), params)
    state = state.replace(
        time_seconds=jnp.asarray(120.0, dtype=jnp.float32),
        request_status=state.request_status.at[0].set(REQUEST_QUEUED),
        request_origin_nodes=state.request_origin_nodes.at[0].set(1),
        request_dest_nodes=state.request_dest_nodes.at[0].set(3),
        request_spawn_times=state.request_spawn_times.at[0].set(0.0),
    )

    obs = build_observation(state, params)

    assert obs.candidate_edges.shape == (params.graph.max_degree, CANDIDATE_EDGE_FEATURES)
    assert float(obs.candidate_edges[0, CE_VALID]) == 1.0
    assert float(obs.candidate_edges[0, CE_REACHABLE_5M]) > 0.0
    assert float(obs.candidate_edges[0, CE_ETA_IMPROVEMENT]) > 0.0
    assert float(obs.candidate_edges[0, CE_BEST_AFTER_EDGE_COUNT]) > 0.0
    assert float(obs.candidate_edges[0, CE_WAIT_WEIGHTED_ADVANTAGE]) > 0.0
    assert int(choose_marginal_value_action(obs)) == 0
    invalid_rows = np.asarray(obs.candidate_edges)[~np.asarray(obs.action_mask)]
    if invalid_rows.size:
        np.testing.assert_allclose(invalid_rows, 0.0, atol=0.0)


def test_assignment_requires_car_within_directed_route_edge_range() -> None:
    graph = build_synthetic_graph(
        node_lonlat=[(float(i), 0.0) for i in range(8)],
        edges=[
            *[
                {"source": i, "target": i + 1, "travel_time_s": 1.0}
                for i in range(7)
            ],
            {"source": 7, "target": 0, "travel_time_s": 1.0},
        ],
    )
    far_params = make_env_params(
        graph,
        max_cars=1,
        max_requests=2,
        initial_car_nodes=[0],
        assignment_max_route_edges=2,
    )
    far_state, _ = reset(jax.random.PRNGKey(0), far_params)
    far_state = far_state.replace(
        request_status=far_state.request_status.at[0].set(REQUEST_QUEUED),
        request_origin_nodes=far_state.request_origin_nodes.at[0].set(7),
        request_dest_nodes=far_state.request_dest_nodes.at[0].set(0),
        request_spawn_times=far_state.request_spawn_times.at[0].set(0.0),
    )

    assert int(nearest_eligible_car_by_eta(far_state, jnp.int32(0), far_params)) == -1

    near_params = make_env_params(
        graph,
        max_cars=2,
        max_requests=2,
        initial_car_nodes=[0, 5],
        assignment_max_route_edges=2,
    )
    near_state, _ = reset(jax.random.PRNGKey(0), near_params)
    near_state = near_state.replace(
        request_status=near_state.request_status.at[0].set(REQUEST_QUEUED),
        request_origin_nodes=near_state.request_origin_nodes.at[0].set(7),
        request_dest_nodes=near_state.request_dest_nodes.at[0].set(0),
        request_spawn_times=near_state.request_spawn_times.at[0].set(0.0),
    )

    assert int(nearest_eligible_car_by_eta(near_state, jnp.int32(0), near_params)) == 1


def test_queued_request_waits_until_car_becomes_eligible() -> None:
    params = make_env_params(
        tiny_graph(),
        max_cars=1,
        max_requests=4,
        initial_car_nodes=[0],
        preplanned_requests=[{"spawn_time_s": 1.0, "origin": 2, "destination": 3}],
    )
    state, _ = reset(jax.random.PRNGKey(0), params)

    state, ts = step(state, jnp.int32(0), params)

    assert int(state.request_status[0]) == REQUEST_COMPLETED
    assert float(state.time_seconds) == 9.0
    assert math.isclose(float(ts.reward), -5.0 / 60.0, rel_tol=1e-6)


def test_dense_reward_penalizes_waiting_without_pickup() -> None:
    graph = build_synthetic_graph(
        node_lonlat=[(0.0, 0.0), (1.0, 0.0)],
        edges=[{"source": 0, "target": 0, "travel_time_s": 2.0}],
    )
    params = make_env_params(
        graph,
        max_cars=1,
        max_requests=2,
        initial_car_nodes=[0],
        preplanned_requests=[{"spawn_time_s": 0.0, "origin": 1, "destination": 1}],
    )
    state, _ = reset(jax.random.PRNGKey(0), params)

    state, ts = step(state, jnp.int32(0), params)

    assert int(state.request_status[0]) == REQUEST_QUEUED
    assert float(ts.reward) < 0.0
    assert math.isclose(float(ts.metrics.last_queued_wait_seconds), 2.0, rel_tol=1e-6)
    assert math.isclose(float(ts.metrics.last_dense_wait_penalty), -2.0 / 60.0, rel_tol=1e-6)
    assert math.isclose(float(ts.reward), -2.0 / 60.0, rel_tol=1e-6)


def test_dense_reward_adds_new_drop_penalty() -> None:
    graph = build_synthetic_graph(
        node_lonlat=[(0.0, 0.0), (1.0, 0.0)],
        edges=[{"source": 0, "target": 0, "travel_time_s": 2.0}],
    )
    params = make_env_params(
        graph,
        max_cars=1,
        max_requests=2,
        initial_car_nodes=[0],
        preplanned_requests=[{"spawn_time_s": 0.0, "origin": 1, "destination": 1, "patience_s": 1.0}],
        drop_penalty=10.0,
    )
    state, _ = reset(jax.random.PRNGKey(0), params)

    state, ts = step(state, jnp.int32(0), params)

    assert int(state.request_status[0]) == REQUEST_DROPPED
    assert math.isclose(float(ts.metrics.last_queued_wait_seconds), 1.0, rel_tol=1e-6)
    assert math.isclose(float(ts.metrics.last_drop_penalty_reward), -10.0, rel_tol=1e-6)
    assert math.isclose(float(ts.reward), -10.0 - 1.0 / 60.0, rel_tol=1e-6)


def test_edge_traffic_profile_sets_travel_time_when_edge_is_entered() -> None:
    params = make_env_params(
        tiny_graph(),
        max_cars=1,
        max_requests=2,
        initial_car_nodes=[0],
        start_time_seconds=17 * 3600.0,
    )
    state, _ = reset(jax.random.PRNGKey(0), params)

    state, ts = step(state, jnp.int32(0), params)

    assert float(ts.dt_seconds) == 15.0
    assert float(state.time_seconds) == 17 * 3600.0 + 15.0


def test_time_aware_discount_and_legacy_pickup_reward_mode() -> None:
    params = make_env_params(
        tiny_graph(),
        max_cars=1,
        max_requests=4,
        initial_car_nodes=[0],
        preplanned_requests=[{"spawn_time_s": 2.0, "origin": 1, "destination": 3}],
        gamma=0.99,
        reward_mode=REWARD_MODE_LEGACY_PICKUP_WAIT,
    )
    state, _ = reset(jax.random.PRNGKey(1), params)

    _, ts = step(state, jnp.int32(0), params)

    assert math.isclose(float(ts.discount), 0.99 ** (9.0 / 60.0), rel_tol=1e-6)
    assert math.isclose(float(ts.reward), -3.0 / 60.0, rel_tol=1e-6)


def test_pickup_wait_average_and_aggregate_reward_metrics() -> None:
    params = make_env_params(
        tiny_graph(),
        max_cars=1,
        max_requests=4,
        initial_car_nodes=[0],
        preplanned_requests=[{"spawn_time_s": 2.0, "origin": 1, "destination": 3}],
    )
    state, timestep = reset(jax.random.PRNGKey(3), params)

    state, timestep = step(state, jnp.int32(0), params)
    scene = export_scene(state, timestep, params)

    assert math.isclose(float(state.metrics.aggregate_reward), float(timestep.reward), rel_tol=1e-6)
    assert int(state.metrics.recent_pickup_wait_count) == 1
    assert math.isclose(scene["metrics"]["avg_pickup_wait_last_10_seconds"], 3.0, rel_tol=1e-6)
    assert math.isclose(scene["metrics"]["aggregate_reward"], -3.0 / 60.0, rel_tol=1e-6)


def test_recent_pickup_wait_metric_keeps_only_last_ten_pickups() -> None:
    params = make_env_params(tiny_graph(), max_cars=1, max_requests=2, initial_car_nodes=[0])
    state, _ = reset(jax.random.PRNGKey(4), params)
    state = state.replace(request_spawn_times=state.request_spawn_times.at[0].set(0.0))

    for wait_seconds in range(1, 13):
        state = state.replace(time_seconds=jnp.asarray(float(wait_seconds), dtype=jnp.float32))
        state = _record_pickup_wait(state, jnp.int32(0))

    assert int(state.metrics.recent_pickup_wait_count) == 10
    assert math.isclose(
        float(state.metrics.recent_pickup_wait_seconds.sum()) / float(state.metrics.recent_pickup_wait_count),
        7.5,
        rel_tol=1e-6,
    )


def test_low_default_demand_smoke_stays_finite() -> None:
    params = make_env_params(
        tiny_graph(),
        max_cars=2,
        max_requests=8,
        initial_car_nodes=[0, 2],
        spawn_rate_per_minute=0.03,
        episode_seconds=120.0,
    )
    state, ts = reset(jax.random.PRNGKey(11), params)

    for i in range(20):
        action = jnp.where(ts.observation.action_mask[0], 0, jnp.argmax(ts.observation.action_mask))
        state, ts = step(state, action.astype(jnp.int32), params)
        assert np.isfinite(float(ts.reward))
        assert np.isfinite(float(ts.discount))
        if bool(ts.done):
            break


def test_density_top_up_keeps_half_as_many_people_in_play_as_cars() -> None:
    graph = tiny_graph().replace(
        node_population_density=jnp.asarray([50.0, 8000.0, 150.0, 4000.0], dtype=jnp.float32)
    )
    params = make_env_params(
        graph,
        max_cars=4,
        max_requests=8,
        initial_car_nodes=[0, 1, 2, 3],
        target_active_request_fraction=0.5,
        episode_seconds=80.0,
    )
    state, ts = reset(jax.random.PRNGKey(5), params)

    assert params.target_active_requests == 2
    assert _active_request_count(state) == 2

    for _ in range(12):
        action = jnp.where(ts.observation.action_mask[0], 0, jnp.argmax(ts.observation.action_mask))
        state, ts = step(state, action.astype(jnp.int32), params)
        if bool(ts.done):
            break
        assert _active_request_count(state) == 2


def test_density_top_up_samples_origins_from_weighted_density() -> None:
    graph = tiny_graph().replace(
        node_population_density=jnp.asarray([0.0, 1.0e9, 0.0, 0.0], dtype=jnp.float32)
    )
    params = make_env_params(
        graph,
        max_cars=6,
        max_requests=8,
        initial_car_nodes=[0, 0, 0, 0, 0, 0],
        target_active_request_fraction=0.5,
    )

    state, _ = reset(jax.random.PRNGKey(0), params)
    active = np.isin(
        np.asarray(state.request_status),
        [REQUEST_QUEUED, REQUEST_ASSIGNED, REQUEST_ONBOARD],
    )

    assert int(active.sum()) == 3
    assert set(np.asarray(state.request_origin_nodes)[active].tolist()) == {1}


def test_density_weights_evolve_by_hour_and_pull_midday_toward_center() -> None:
    graph = tiny_graph().replace(
        node_population_density=jnp.asarray([1000.0, 1000.0, 1000.0, 1000.0], dtype=jnp.float32),
        node_grid_rows=jnp.asarray([0, 25, 25, 49], dtype=jnp.int32),
        node_grid_cols=jnp.asarray([0, 25, 25, 49], dtype=jnp.int32),
    )

    params = make_env_params(
        graph,
        max_cars=2,
        max_requests=4,
        initial_car_nodes=[0, 1],
        target_active_request_fraction=0.5,
    )

    weights = np.asarray(params.node_density_by_hour)
    assert weights.shape == (24, graph.num_nodes)
    assert weights[13, 1] > weights[13, 0]
    assert not np.allclose(weights[13], weights[21])


def test_scheduled_requests_overflow_active_capacity_and_count_drops() -> None:
    graph = build_synthetic_graph(
        node_lonlat=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        edges=[
            {"source": 0, "target": 0, "travel_time_s": 2.0},
        ],
    )
    params = make_env_params(
        graph,
        max_cars=1,
        max_requests=1,
        initial_car_nodes=[0],
        preplanned_requests=[
            {"spawn_time_s": 0.0, "origin": 1, "destination": 3},
            {"spawn_time_s": 0.0, "origin": 2, "destination": 3},
        ],
    )

    state, _ = reset(jax.random.PRNGKey(0), params)

    assert int(state.next_scheduled_request_index) == 2
    assert int((state.request_status == REQUEST_QUEUED).sum()) == 1
    assert int(state.metrics.dropped_requests) == 1


def test_queued_scheduled_request_expires_after_patience() -> None:
    graph = build_synthetic_graph(
        node_lonlat=[(0.0, 0.0), (1.0, 0.0)],
        edges=[
            {"source": 0, "target": 0, "travel_time_s": 2.0},
        ],
    )
    params = make_env_params(
        graph,
        max_cars=1,
        max_requests=2,
        initial_car_nodes=[0],
        preplanned_requests=[
            {"spawn_time_s": 0.0, "origin": 1, "destination": 1, "patience_s": 5.0},
        ],
        episode_seconds=12.0,
    )
    state, ts = reset(jax.random.PRNGKey(0), params)

    for _ in range(3):
        state, ts = step(state, jnp.int32(0), params)

    assert int(state.request_status[0]) == REQUEST_DROPPED
    assert int(state.metrics.dropped_requests) == 1
    assert int(state.metrics.queued_requests) == 0


def test_reset_step_jit_and_vmap_shapes() -> None:
    params = make_env_params(tiny_graph(), max_cars=2, max_requests=4, initial_car_nodes=[0, 2])

    jitted_reset = jax.jit(reset, static_argnums=())
    state, ts = jitted_reset(jax.random.PRNGKey(0), params)
    jitted_step = jax.jit(step, static_argnums=())
    state, ts = jitted_step(state, jnp.int32(0), params)

    assert state.car_nodes.shape == (2,)
    assert ts.observation.raster.shape == (50, 50, RASTER_CHANNELS)
    assert ts.observation.local_raster.shape == (50, 50, RASTER_CHANNELS)
    assert ts.observation.structured.shape == (STRUCTURED_FEATURES,)
    assert ts.observation.candidate_edges.shape == (params.graph.max_degree, CANDIDATE_EDGE_FEATURES)
    assert np.isfinite(np.asarray(ts.observation.raster)).all()
    assert np.isfinite(np.asarray(ts.observation.local_raster)).all()
    assert np.isfinite(np.asarray(ts.observation.structured)).all()
    assert np.isfinite(np.asarray(ts.observation.candidate_edges)).all()
    np.testing.assert_allclose(
        np.asarray(ts.observation.structured)[[STRUCT_CURRENT_X, STRUCT_CURRENT_Y]],
        [1.0, 0.0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(ts.observation.candidate_edges[0, :4]),
        [-0.5, 0.0, 0.5, 0.0],
        rtol=1e-6,
    )
    assert float(ts.observation.raster[:, :, RASTER_EXPECTED_DEMAND_10M].max()) > 0.0
    assert float(ts.observation.raster[:, :, RASTER_FOCUS_CAR].sum()) == 1.0
    assert float(ts.observation.local_raster[:, :, RASTER_EXPECTED_DEMAND_10M].max()) > 0.0
    assert float(ts.observation.local_raster[25, 25, RASTER_FOCUS_CAR]) == 1.0
    keys = jax.random.split(jax.random.PRNGKey(5), 3)
    states, timesteps = jax.vmap(lambda k: reset(k, params))(keys)
    assert states.car_nodes.shape == (3, 2)
    assert timesteps.observation.action_mask.shape == (3, params.graph.max_degree)


def test_legacy_raster_splits_available_and_busy_car_channels() -> None:
    params = make_env_params(
        tiny_graph(),
        max_cars=2,
        max_requests=4,
        initial_car_nodes=[0, 0],
        observation_mode="legacy",
    )
    state, _ = reset(jax.random.PRNGKey(0), params)
    state = state.replace(
        car_status=jnp.asarray([CAR_DECISION, CAR_TO_PICKUP], dtype=jnp.int32),
        current_car_id=jnp.asarray(0, dtype=jnp.int32),
    )

    obs = build_observation(state, params)

    assert obs.raster.shape == (50, 50, LEGACY_RASTER_CHANNELS)
    assert obs.local_raster.shape == (50, 50, LEGACY_RASTER_CHANNELS)
    assert obs.structured.shape == (LEGACY_STRUCTURED_FEATURES,)
    row = int(np.asarray(params.graph.node_grid_rows[0]))
    col = int(np.asarray(params.graph.node_grid_cols[0]))
    assert float(obs.raster[row, col, 0]) == 1.0
    assert float(obs.raster[row, col, 1]) == 1.0
    assert float(obs.raster[:, :, 0].sum()) == 1.0
    assert float(obs.raster[:, :, 1].sum()) == 1.0
    assert float(obs.local_raster[25, 25, 0]) == 1.0
    assert float(obs.local_raster[25, 25, 1]) == 1.0
    assert float(obs.raster[:, :, LEGACY_DENSITY_CHANNEL].max()) > 0.0
    assert float(obs.local_raster[25, 25, LEGACY_FOCUS_CAR_CHANNEL]) == 1.0


def test_scene_export_schema_uses_same_state_as_debug_visualizer() -> None:
    params = make_env_params(
        tiny_graph(),
        max_cars=1,
        max_requests=4,
        initial_car_nodes=[0],
        preplanned_requests=[{"spawn_time_s": 2.0, "origin": 1, "destination": 3}],
    )
    state, timestep = reset(jax.random.PRNGKey(0), params)
    state, timestep = step(state, jnp.int32(0), params)

    scene = export_scene(state, timestep, params)

    assert scene["time_seconds"] == 9.0
    assert scene["cars"][0]["node_id"] == 3
    assert scene["requests"][0]["status"] == "completed"
    assert {"cars", "requests", "congestion", "recent_events", "edge_progress", "route_previews"} <= set(scene)
