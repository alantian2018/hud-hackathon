from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_fleet.env import (
    REQUEST_DROPPED,
    REQUEST_COMPLETED,
    REQUEST_QUEUED,
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
    assert ts.observation.raster.shape == (50, 50, 3)
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
