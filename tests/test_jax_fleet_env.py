from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_fleet.env import (
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


def test_dt_seconds_discount_and_sparse_pickup_reward() -> None:
    params = make_env_params(
        tiny_graph(),
        max_cars=1,
        max_requests=4,
        initial_car_nodes=[0],
        preplanned_requests=[{"spawn_time_s": 2.0, "origin": 1, "destination": 3}],
        gamma=0.99,
        discount_time_unit_seconds=60.0,
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
    assert ts.observation.raster.shape == (50, 50, 5)
    assert ts.observation.local_raster.shape == (50, 50, 5)
    assert float(ts.observation.raster[:, :, 3].max()) > 0.0
    assert float(ts.observation.raster[:, :, 4].sum()) == 1.0
    assert float(ts.observation.local_raster[:, :, 3].max()) > 0.0
    assert float(ts.observation.local_raster[25, 25, 4]) == 1.0
    keys = jax.random.split(jax.random.PRNGKey(5), 3)
    states, timesteps = jax.vmap(lambda k: reset(k, params))(keys)
    assert states.car_nodes.shape == (3, 2)
    assert timesteps.observation.action_mask.shape == (3, params.graph.max_degree)


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
