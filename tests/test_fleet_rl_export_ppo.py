from __future__ import annotations

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from fleet_rl.env import EnvParams, FleetEnv
from fleet_rl.export import export_scene
from fleet_rl.graph import build_synthetic_debug_graph, load_ppo_json_graph
from fleet_rl.ppo import (
    PPOConfig,
    collect_rollout_from,
    initialize_env_batch,
    make_train_state,
    mean_recent_pickup_wait,
    mean_recent_request_wait_or_age,
    ppo_smoke_update,
)
from fleet_rl.types import Metrics


def test_scene_export_is_json_compatible_and_contains_frontend_concepts() -> None:
    graph = build_synthetic_debug_graph("line")
    env = FleetEnv(graph)
    params = EnvParams.for_graph(graph, num_cars=2, demand_rate_per_second=0.0, randomize_start_time=False)
    state, timestep = env.reset(jax.random.PRNGKey(0), params)
    state, timestep = env.step(state, jnp.array(0, dtype=jnp.int32), params)

    scene = export_scene(state, timestep, params)

    assert set(scene) >= {"sim_time_seconds", "cars", "active_requests", "queued_requests", "edges", "recent_events"}
    assert scene["cars"][0]["id"] == 0
    assert {"status", "current_node", "from_node", "to_node", "edge_id", "edge_progress", "assigned_request_id"} <= set(scene["cars"][0])
    assert isinstance(scene["edges"][0]["congestion"], float)


def test_ppo_smoke_update_runs_on_synthetic_graph() -> None:
    graph = build_synthetic_debug_graph("grid3")
    env = FleetEnv(graph)
    env_params = EnvParams.for_graph(
        graph,
        num_cars=4,
        demand_rate_per_second=1.0 / 500.0,
        max_active_requests=16,
        max_internal_events=64,
        randomize_start_time=False,
    )
    state, timestep = env.reset(jax.random.PRNGKey(1), env_params)
    config = PPOConfig(num_envs=2, num_steps=8, update_epochs=1, minibatch_size=16)
    train_state = make_train_state(jax.random.PRNGKey(2), timestep.observation, env_params.max_degree, config)

    train_state, metrics = ppo_smoke_update(
        train_state,
        env,
        env_params,
        config,
        jax.random.PRNGKey(3),
    )

    assert metrics["loss"].shape == ()
    assert jnp.isfinite(metrics["loss"])
    assert metrics["rollout_reward"].shape == ()


def test_ppo_smoke_update_runs_on_actual_sf_graph() -> None:
    graph = load_ppo_json_graph(
        "dist/data/ppo_nodes.json",
        "dist/data/ppo_edges.json",
        node_limit=None,
        route_mode="landmark",
        num_landmarks=8,
    )
    env = FleetEnv(graph)
    env_params = EnvParams.for_graph(
        graph,
        num_cars=4,
        demand_rate_per_second=1.0 / 900.0,
        max_active_requests=16,
        max_internal_events=64,
        randomize_start_time=False,
    )
    _state, timestep = env.reset(jax.random.PRNGKey(11), env_params)
    config = PPOConfig(num_envs=1, num_steps=2, update_epochs=1, minibatch_size=2)
    train_state = make_train_state(jax.random.PRNGKey(12), timestep.observation, env_params.max_degree, config)

    train_state, metrics = ppo_smoke_update(
        train_state,
        env,
        env_params,
        config,
        jax.random.PRNGKey(13),
    )

    assert metrics["loss"].shape == ()
    assert jnp.isfinite(metrics["loss"])


def test_rollout_collection_can_continue_without_resetting_between_updates() -> None:
    graph = build_synthetic_debug_graph("grid3")
    env = FleetEnv(graph)
    env_params = EnvParams.for_graph(
        graph,
        num_cars=4,
        demand_rate_per_second=1.0 / 30.0,
        max_active_requests=16,
        max_internal_events=64,
        randomize_start_time=False,
    )
    _state, timestep = env.reset(jax.random.PRNGKey(21), env_params)
    config = PPOConfig(num_envs=1, num_steps=16, update_epochs=1, minibatch_size=16)
    train_state = make_train_state(jax.random.PRNGKey(22), timestep.observation, env_params.max_degree, config)
    states, timesteps, key = initialize_env_batch(env, env_params, config, jax.random.PRNGKey(23))

    _transitions, _last_value, states, timesteps, key = collect_rollout_from(
        train_state, env, env_params, config, states, timesteps, key
    )
    first_segment_time = timesteps.sim_time_seconds
    _transitions, _last_value, states, timesteps, key = collect_rollout_from(
        train_state, env, env_params, config, states, timesteps, key
    )

    assert jnp.all(timesteps.sim_time_seconds >= first_segment_time)
    assert float(timesteps.sim_time_seconds[0]) > 0.0


def test_mean_recent_pickup_wait_uses_last_10_global_pickups() -> None:
    samples = jnp.zeros((2, 16), dtype=jnp.float32)
    samples = samples.at[0, :12].set(jnp.arange(1, 13, dtype=jnp.float32))
    samples = samples.at[1, :3].set(jnp.array([100.0, 200.0, 300.0], dtype=jnp.float32))
    times = jnp.zeros((2, 16), dtype=jnp.float32)
    times = times.at[0, :12].set(jnp.arange(1, 13, dtype=jnp.float32))
    times = times.at[1, :3].set(jnp.array([13.0, 14.0, 15.0], dtype=jnp.float32))
    metrics = Metrics(
        requests_spawned=jnp.zeros((2,), dtype=jnp.int32),
        requests_queued=jnp.zeros((2,), dtype=jnp.int32),
        requests_assigned=jnp.zeros((2,), dtype=jnp.int32),
        requests_picked_up=jnp.zeros((2,), dtype=jnp.int32),
        requests_completed=jnp.zeros((2,), dtype=jnp.int32),
        dropped_requests=jnp.zeros((2,), dtype=jnp.int32),
        queue_length=jnp.zeros((2,), dtype=jnp.int32),
        total_pickup_wait_time=jnp.zeros((2,), dtype=jnp.float32),
        avg_pickup_wait_time=jnp.zeros((2,), dtype=jnp.float32),
        p50_pickup_wait_time=jnp.zeros((2,), dtype=jnp.float32),
        p90_pickup_wait_time=jnp.zeros((2,), dtype=jnp.float32),
        p95_pickup_wait_time=jnp.zeros((2,), dtype=jnp.float32),
        pickup_wait_samples=samples,
        pickup_wait_sample_times=times,
        pickup_wait_count=jnp.array([12, 3], dtype=jnp.int32),
        fleet_utilization=jnp.zeros((2,), dtype=jnp.float32),
        empty_driving_time=jnp.zeros((2,), dtype=jnp.float32),
        empty_driving_distance=jnp.zeros((2,), dtype=jnp.float32),
        invalid_actions=jnp.zeros((2,), dtype=jnp.int32),
        overflow=jnp.zeros((2,), dtype=bool),
    )

    # Latest 10 pickup timestamps are env0 times 6..12 plus env1 times 13..15.
    expected = (sum(range(6, 13)) + 100 + 200 + 300) / 10
    assert math.isclose(float(mean_recent_pickup_wait(metrics, window=10)), expected, rel_tol=1e-6)


def test_mean_recent_request_wait_or_age_uses_last_10_spawned_requests() -> None:
    states = SimpleNamespace(
        sim_time_seconds=jnp.array([100.0, 20.0], dtype=jnp.float32),
        request_ids=jnp.array(
            [
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, -1],
            ],
            dtype=jnp.int32,
        ),
        request_spawn_time=jnp.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        request_pickup_time=jnp.array(
            [
                [11.0, 22.0, 33.0, 44.0, 55.0, 66.0],
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            ],
            dtype=jnp.float32,
        ),
    )

    # Latest 10 spawns are 2..11. Picked requests use final wait; open requests use current age.
    expected = (20 + 30 + 40 + 50 + 60 + 13 + 12 + 11 + 10 + 9) / 10
    assert math.isclose(float(mean_recent_request_wait_or_age(states, window=10)), expected, rel_tol=1e-6)
