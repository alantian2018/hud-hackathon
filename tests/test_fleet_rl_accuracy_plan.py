from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from fleet_rl.debug_viz import _route_nodes
from fleet_rl.env import EnvParams, FleetEnv
from fleet_rl.export import export_scene
from fleet_rl.graph import build_graph_from_edges, build_synthetic_debug_graph


def test_reset_randomizes_car_start_nodes_by_seed() -> None:
    graph = build_synthetic_debug_graph("grid3")
    env = FleetEnv(graph)
    params = EnvParams.for_graph(
        graph,
        num_cars=6,
        demand_rate_per_second=0.0,
        randomize_start_time=False,
    )

    state_a, _ = env.reset(jax.random.PRNGKey(101), params)
    state_b, _ = env.reset(jax.random.PRNGKey(102), params)
    state_c, _ = env.reset(jax.random.PRNGKey(101), params)

    assert not jnp.array_equal(state_a.car_node[:6], state_b.car_node[:6])
    assert jnp.array_equal(state_a.car_node, state_c.car_node)


def test_edge_traffic_profile_is_sampled_at_edge_entry_time() -> None:
    graph = build_graph_from_edges(
        nodes=[
            {"id": 0, "lon": 0.0, "lat": 0.0},
            {"id": 1, "lon": 1.0, "lat": 0.0},
        ],
        edges=[
            {
                "id": 0,
                "from": 0,
                "to": 1,
                "length_m": 100.0,
                "base_travel_time_s": 10.0,
                "traffic_profile": [1.0, 3.0] + [1.0] * 22,
            },
            {"id": 1, "from": 1, "to": 0, "length_m": 100.0, "base_travel_time_s": 10.0},
        ],
    )
    env = FleetEnv(graph)
    params = EnvParams.for_graph(
        graph,
        num_cars=1,
        demand_rate_per_second=0.0,
        randomize_start_time=False,
    )
    state, _ = env.reset(jax.random.PRNGKey(0), params)
    state = state.replace(sim_time_seconds=jnp.array(3600.0, dtype=jnp.float32))

    state, timestep = env.step(state, jnp.array(0, dtype=jnp.int32), params)

    assert math.isclose(float(timestep.dt_seconds), 30.0, rel_tol=1e-5)


def test_frontend_demand_profile_parameters_drive_request_rate() -> None:
    graph = build_synthetic_debug_graph("grid3")
    env = FleetEnv(graph)
    low_profile = [0.0] * 24
    high_profile = [0.0] * 24
    high_profile[1] = 4.0
    params = EnvParams.for_graph(
        graph,
        num_cars=1,
        demand_rate_per_second=1.0,
        demand_time_profile=high_profile,
        randomize_start_time=False,
    )
    low_params = EnvParams.for_graph(
        graph,
        num_cars=1,
        demand_rate_per_second=1.0,
        demand_time_profile=low_profile,
        randomize_start_time=False,
    )

    assert float(env._request_rate_at(jnp.array(3600.0), params)) > 0.0
    assert float(env._request_rate_at(jnp.array(3600.0), low_params)) == 0.0


def test_wait_quantile_metrics_and_recent_events_are_populated() -> None:
    env, params = FleetEnv(build_synthetic_debug_graph("line")), None
    params = EnvParams.for_graph(
        env.graph,
        num_cars=1,
        demand_rate_per_second=0.0,
        randomize_start_time=False,
    )
    state, _ = env.reset(jax.random.PRNGKey(10), params)
    state = state.replace(
        car_node=state.car_node.at[0].set(0),
        car_from_node=state.car_from_node.at[0].set(0),
        car_to_node=state.car_to_node.at[0].set(0),
    )
    state = env.debug_insert_request(
        state,
        pickup_node=1,
        dropoff_node=2,
        spawn_time=-5.0,
        assign=True,
        params=params,
    )

    state, timestep = env.debug_advance_to_next_decision(state, params)

    assert int(state.metrics.requests_picked_up) == 1
    assert float(state.metrics.avg_pickup_wait_time) == 15.0
    assert float(state.metrics.p50_pickup_wait_time) == 15.0
    assert float(state.metrics.p90_pickup_wait_time) == 15.0
    assert int(jnp.count_nonzero(state.recent_event_codes)) >= 3
    assert export_scene(state, timestep, params)["recent_events"]


def test_scene_export_interpolates_moving_car_coordinates() -> None:
    graph = build_synthetic_debug_graph("line")
    env = FleetEnv(graph)
    params = EnvParams.for_graph(graph, num_cars=1, demand_rate_per_second=0.0, randomize_start_time=False)
    state, timestep = env.reset(jax.random.PRNGKey(0), params)
    state = env._enter_edge(state, jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32), params)
    state = state.replace(sim_time_seconds=jnp.array(5.0, dtype=jnp.float32))
    timestep = timestep.replace(sim_time_seconds=state.sim_time_seconds, dt_seconds=jnp.array(5.0, dtype=jnp.float32))

    scene = export_scene(state, timestep, params)

    assert math.isclose(scene["cars"][0]["lon"], 0.5, rel_tol=1e-5)
    assert math.isclose(scene["cars"][0]["edge_progress"], 0.5, rel_tol=1e-5)


def test_debug_viz_reconstructs_landmark_routes() -> None:
    graph = build_graph_from_edges(
        nodes=[
            {"id": 0, "lon": 0.0, "lat": 0.0},
            {"id": 1, "lon": 1.0, "lat": 0.0},
            {"id": 2, "lon": 2.0, "lat": 0.0},
        ],
        edges=[
            {"id": 0, "from": 0, "to": 1, "length_m": 100.0, "base_travel_time_s": 10.0},
            {"id": 1, "from": 1, "to": 2, "length_m": 100.0, "base_travel_time_s": 10.0},
            {"id": 2, "from": 2, "to": 0, "length_m": 200.0, "base_travel_time_s": 20.0},
        ],
        route_mode="landmark",
        num_landmarks=1,
    )

    assert _route_nodes(graph, 0, 2) == [0, 1, 2]
