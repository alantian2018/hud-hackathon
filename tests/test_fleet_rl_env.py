from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from fleet_rl.env import FleetEnv, EnvParams
from fleet_rl.graph import build_graph_from_edges, build_synthetic_debug_graph, load_ppo_json_graph, python_shortest_path
from fleet_rl.types import POLICY_CONTROLLED, TO_DROPOFF, TO_PICKUP, REQ_ASSIGNED, REQ_QUEUED


def make_env(num_cars: int = 2, demand_rate: float = 0.0, graph_name: str = "line") -> tuple[FleetEnv, EnvParams]:
    graph = build_synthetic_debug_graph(graph_name)
    params = EnvParams.for_graph(
        graph,
        num_cars=num_cars,
        demand_rate_per_second=demand_rate,
        max_active_requests=8,
        max_internal_events=64,
        max_sim_time_seconds=3600.0,
        randomize_start_time=False,
    )
    return FleetEnv(graph), params


def test_directed_graph_asymmetry_and_oracle_match_training_router() -> None:
    graph = build_synthetic_debug_graph("asymmetric")

    forward = graph.travel_time_table[0, 2]
    backward = graph.travel_time_table[2, 0]

    assert forward == 2.0
    assert backward == 20.0
    assert graph.next_hop_table[0, 2] == 1
    assert python_shortest_path(graph, 0, 2).cost == forward
    assert python_shortest_path(graph, 2, 0).cost == backward


def test_landmark_node_to_landmark_next_edge_starts_at_source() -> None:
    graph = build_graph_from_edges(
        nodes=[
            {"id": 0, "lon": 0.0, "lat": 0.0},
            {"id": 1, "lon": 1.0, "lat": 0.0},
            {"id": 2, "lon": 2.0, "lat": 0.0},
            {"id": 3, "lon": 3.0, "lat": 0.0},
        ],
        edges=[
            {"id": 0, "from": 1, "to": 0, "length_m": 100.0, "base_travel_time_s": 10.0},
            {"id": 1, "from": 2, "to": 1, "length_m": 100.0, "base_travel_time_s": 10.0},
            {"id": 2, "from": 3, "to": 2, "length_m": 100.0, "base_travel_time_s": 10.0},
            {"id": 3, "from": 0, "to": 3, "length_m": 500.0, "base_travel_time_s": 50.0},
        ],
        route_mode="landmark",
        num_landmarks=1,
    )

    edge = int(graph.node_to_landmark_next_edge[3, 0])

    assert edge == 2
    assert int(graph.edge_from[edge]) == 3


def test_landmark_route_selector_falls_back_when_table_edge_does_not_start_at_source() -> None:
    graph = build_graph_from_edges(
        nodes=[
            {"id": 0, "lon": 0.0, "lat": 0.0},
            {"id": 1, "lon": 1.0, "lat": 0.0},
            {"id": 2, "lon": 2.0, "lat": 0.0},
            {"id": 3, "lon": 3.0, "lat": 0.0},
        ],
        edges=[
            {"id": 0, "from": 1, "to": 0, "length_m": 100.0, "base_travel_time_s": 10.0},
            {"id": 1, "from": 2, "to": 1, "length_m": 100.0, "base_travel_time_s": 10.0},
            {"id": 2, "from": 3, "to": 2, "length_m": 100.0, "base_travel_time_s": 10.0},
            {"id": 3, "from": 0, "to": 3, "length_m": 500.0, "base_travel_time_s": 50.0},
        ],
        route_mode="landmark",
        num_landmarks=1,
    )
    graph = graph.replace(node_to_landmark_next_edge=graph.node_to_landmark_next_edge.at[3, 0].set(0))
    env = FleetEnv(graph)
    params = EnvParams.for_graph(graph, num_cars=1, demand_rate_per_second=0.0, randomize_start_time=False)
    state, _ = env.reset(jax.random.PRNGKey(123), params)

    edge = int(env._next_route_edge(state, jnp.array(3, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32), params))

    assert edge == 2
    assert int(graph.edge_from[edge]) == 3


def test_reset_and_step_are_deterministic_with_fixed_seed() -> None:
    env, params = make_env(num_cars=2)
    key = jax.random.PRNGKey(0)

    s1, t1 = env.reset(key, params)
    s2, t2 = env.reset(key, params)
    s1b, t1b = env.step(s1, jnp.array(0, dtype=jnp.int32), params)
    s2b, t2b = env.step(s2, jnp.array(0, dtype=jnp.int32), params)

    np.testing.assert_array_equal(s1.car_node, s2.car_node)
    np.testing.assert_array_equal(t1.action_mask, t2.action_mask)
    np.testing.assert_array_equal(s1b.car_node, s2b.car_node)
    assert float(t1b.sim_time_seconds) == float(t2b.sim_time_seconds)


def test_one_step_consumes_one_car_decision_and_simultaneous_decisions_are_ascending_without_time_advance() -> None:
    env, params = make_env(num_cars=3)
    state, timestep = env.reset(jax.random.PRNGKey(3), params)

    assert int(timestep.current_car_id) == 0
    state, timestep = env.step(state, jnp.array(0, dtype=jnp.int32), params)

    assert int(timestep.current_car_id) == 1
    assert float(timestep.dt_seconds) == 0.0
    assert float(timestep.sim_time_seconds) == 0.0

    state, timestep = env.step(state, jnp.array(0, dtype=jnp.int32), params)
    assert int(timestep.current_car_id) == 2
    assert float(timestep.dt_seconds) == 0.0


def test_no_stay_action_and_variable_degree_action_masks() -> None:
    env, params = make_env(num_cars=1, graph_name="variable_degree")
    state, timestep = env.reset(jax.random.PRNGKey(1), params)

    assert timestep.action_mask.shape == (params.max_degree,)
    assert bool(timestep.action_mask[0])
    assert int(timestep.action_mask.sum()) == int(params.graph.out_degree[int(timestep.current_node_id)])
    assert int(timestep.action_mask.sum()) < params.max_degree


def test_no_policy_query_during_to_pickup_or_to_dropoff() -> None:
    env, params = make_env(num_cars=1, demand_rate=0.0)
    state, timestep = env.reset(jax.random.PRNGKey(4), params)
    state = env.debug_insert_request(state, pickup_node=2, dropoff_node=0, spawn_time=0.0, assign=True, params=params)

    assert int(state.car_status[0]) == TO_PICKUP
    state, timestep = env.debug_advance_to_next_decision(state, params)

    assert int(timestep.current_car_id) == 0
    assert int(state.car_status[0]) == POLICY_CONTROLLED
    assert float(timestep.sim_time_seconds) > 0.0
    assert int(state.metrics.requests_picked_up) == 1
    assert int(state.metrics.requests_completed) == 1


def test_nearest_assignment_uses_directed_car_to_pickup_eta() -> None:
    graph = build_synthetic_debug_graph("directed_assignment")
    env = FleetEnv(graph)
    params = EnvParams.for_graph(graph, num_cars=2, demand_rate_per_second=0.0, randomize_start_time=False)
    state, _ = env.reset(jax.random.PRNGKey(8), params)
    state = state.replace(
        car_node=jnp.array([0, 2] + [0] * (params.max_cars - 2), dtype=jnp.int32),
        car_from_node=jnp.array([0, 2] + [0] * (params.max_cars - 2), dtype=jnp.int32),
        car_to_node=jnp.array([0, 2] + [0] * (params.max_cars - 2), dtype=jnp.int32),
    )

    state = env.debug_insert_request(state, pickup_node=1, dropoff_node=0, spawn_time=0.0, assign=True, params=params)

    assert int(state.request_assigned_car_id[0]) == 1
    assert int(state.car_status[1]) == TO_PICKUP


def test_queued_request_behavior_and_assignment_when_car_becomes_eligible() -> None:
    env, params = make_env(num_cars=1, demand_rate=0.0)
    state, _ = env.reset(jax.random.PRNGKey(9), params)
    state = state.replace(car_status=state.car_status.at[0].set(TO_DROPOFF))
    state = env.debug_insert_request(state, pickup_node=1, dropoff_node=2, spawn_time=0.0, assign=True, params=params)

    assert int(state.request_status[0]) == REQ_QUEUED
    assert int(state.metrics.requests_queued) == 1

    state = state.replace(car_status=state.car_status.at[0].set(POLICY_CONTROLLED))
    state = env.debug_assign_queued_to_available_cars(state, params)

    assert int(state.request_status[0]) == REQ_ASSIGNED
    assert int(state.request_assigned_car_id[0]) == 0
    assert int(state.car_status[0]) == TO_PICKUP


def test_edge_based_traffic_sets_entry_travel_time_and_discount_uses_dt() -> None:
    graph = build_synthetic_debug_graph("line")
    graph = graph.replace(edge_base_travel_time_s=graph.edge_base_travel_time_s.at[0].set(10.0))
    env = FleetEnv(graph)
    params = EnvParams.for_graph(
        graph,
        num_cars=1,
        demand_rate_per_second=0.0,
        gamma=0.9,
        discount_time_unit_seconds=10.0,
        randomize_start_time=False,
    )
    state, _ = env.reset(jax.random.PRNGKey(2), params)
    state = state.replace(
        car_node=state.car_node.at[0].set(0),
        car_from_node=state.car_from_node.at[0].set(0),
        car_to_node=state.car_to_node.at[0].set(0),
    )
    state = state.replace(edge_congestion=state.edge_congestion.at[0].set(2.5))

    state, timestep = env.step(state, jnp.array(0, dtype=jnp.int32), params)

    assert math.isclose(float(timestep.dt_seconds), 25.0, rel_tol=1e-5)
    assert math.isclose(float(timestep.discount), 0.9 ** 2.5, rel_tol=1e-5)


def test_sparse_wait_time_reward_emitted_at_pickup() -> None:
    env, params = make_env(num_cars=1, demand_rate=0.0)
    state, _ = env.reset(jax.random.PRNGKey(10), params)
    state = state.replace(car_node=state.car_node.at[0].set(0), car_from_node=state.car_from_node.at[0].set(0), car_to_node=state.car_to_node.at[0].set(0))
    state = env.debug_insert_request(state, pickup_node=1, dropoff_node=2, spawn_time=-5.0, assign=True, params=params)

    state, timestep = env.debug_advance_to_next_decision(state, params)

    assert int(state.metrics.requests_picked_up) == 1
    assert float(timestep.reward) < 0.0
    assert math.isclose(float(timestep.reward), -params.wait_time_scale * 15.0, rel_tol=1e-5)


def test_low_default_demand_smoke_does_not_runaway_queue_under_random_policy() -> None:
    env, params = make_env(num_cars=8, demand_rate=1.0 / 600.0, graph_name="grid3")
    state, timestep = env.reset(jax.random.PRNGKey(33), params)
    key = jax.random.PRNGKey(34)

    def rollout(carry, _):
        state, timestep, key, max_queue = carry
        key, subkey = jax.random.split(key)
        logits = jnp.where(timestep.action_mask, 0.0, -1e9)
        action = jax.random.categorical(subkey, logits).astype(jnp.int32)
        state, timestep = env.step(state, action, params)
        max_queue = jnp.maximum(max_queue, state.metrics.queue_length)
        return (state, timestep, key, max_queue), None

    state, timestep, _key, max_queue = jax.jit(
        lambda s, t, k: jax.lax.scan(rollout, (s, t, k, jnp.array(0, dtype=jnp.int32)), None, length=80)[0]
    )(state, timestep, key)
    assert max_queue <= params.max_active_requests
    assert int(state.metrics.dropped_requests) == 0


def test_jit_reset_step_and_vmap_shapes() -> None:
    env, params = make_env(num_cars=2)

    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)
    state, timestep = reset_jit(jax.random.PRNGKey(0), params)
    state, timestep = step_jit(state, jnp.array(0, dtype=jnp.int32), params)

    assert timestep.observation.raster.shape[-1] == 9
    assert timestep.observation.action_features.shape == (params.max_degree, 9)

    keys = jax.random.split(jax.random.PRNGKey(99), 4)
    states, timesteps = jax.vmap(lambda k: env.reset(k, params))(keys)
    actions = jnp.zeros((4,), dtype=jnp.int32)
    states, timesteps = jax.vmap(lambda s, a: env.step(s, a, params))(states, actions)

    assert states.car_node.shape == (4, params.max_cars)
    assert timesteps.action_mask.shape == (4, params.max_degree)


def test_actual_sf_graph_loads_full_graph_with_compact_landmark_router_and_jit_step() -> None:
    graph = load_ppo_json_graph(
        "dist/data/ppo_nodes.json",
        "dist/data/ppo_edges.json",
        node_limit=None,
        route_mode="landmark",
        num_landmarks=8,
    )

    assert int(graph.num_nodes) > 5_000
    assert int(graph.num_edges) > 10_000
    assert graph.route_mode == "landmark"
    assert graph.travel_time_table.shape == (1, 1)
    assert graph.next_edge_table.shape == (1, 1)
    assert graph.landmark_to_node_time.shape == (8, graph.max_nodes)
    assert int(graph.out_degree[: int(graph.num_nodes)].min()) > 0

    env = FleetEnv(graph)
    params = EnvParams.for_graph(
        graph,
        num_cars=4,
        demand_rate_per_second=0.0,
        max_active_requests=8,
        max_internal_events=64,
        randomize_start_time=False,
    )
    state, timestep = jax.jit(env.reset)(jax.random.PRNGKey(101), params)
    state, timestep = jax.jit(env.step)(state, jnp.array(0, dtype=jnp.int32), params)

    assert timestep.action_mask.shape == (graph.max_degree,)
    assert int(timestep.current_car_id) >= 0
