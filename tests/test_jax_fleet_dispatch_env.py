from __future__ import annotations

import math

import jax
import numpy as np

from jax_fleet.dispatch_env import ManualDispatchEnv
from jax_fleet.env import (
    CAR_DECISION,
    REQUEST_COMPLETED,
    REQUEST_QUEUED,
    make_env_params,
    reset,
)
from jax_fleet.graph import build_synthetic_graph


def dispatch_graph():
    return build_synthetic_graph(
        node_lonlat=[
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (1.0, 1.0),
        ],
        edges=[
            {"source": 0, "target": 1, "travel_time_s": 5.0},
            {"source": 1, "target": 2, "travel_time_s": 4.0},
            {"source": 2, "target": 3, "travel_time_s": 3.0},
            {"source": 3, "target": 0, "travel_time_s": 6.0},
            {"source": 0, "target": 3, "travel_time_s": 10.0},
        ],
    )


def test_manual_dispatch_mode_keeps_queued_requests_unassigned_on_reset() -> None:
    params = make_env_params(
        dispatch_graph(),
        max_cars=1,
        max_requests=2,
        initial_car_nodes=[0],
        preplanned_requests=[{"spawn_time_s": 0.0, "origin": 1, "destination": 2}],
        manual_dispatch=True,
    )

    state, _ = reset(jax.random.PRNGKey(0), params)

    assert int(state.request_status[0]) == REQUEST_QUEUED
    assert int(state.request_assigned_car_ids[0]) == -1
    assert int(state.car_status[0]) == CAR_DECISION
    assert bool(state.decision_required)


def test_manual_dispatch_env_assigns_request_and_auto_routes_to_dropoff() -> None:
    graph = dispatch_graph()
    params = make_env_params(
        graph,
        max_cars=1,
        max_requests=2,
        initial_car_nodes=[0],
        preplanned_requests=[{"spawn_time_s": 0.0, "origin": 1, "destination": 2}],
        manual_dispatch=True,
        episode_seconds=120.0,
    )
    env = ManualDispatchEnv(graph=graph, params=params, seed=0)
    env.reset()

    result = env.submit_dispatch_plan(
        [{"car_id": 0, "action": "assign_request", "request_id": 0}]
    )

    assert result.action_results[0]["valid"]
    assert int(env.state.request_status[0]) == REQUEST_COMPLETED
    assert int(env.state.metrics.completed_requests) == 1
    assert int(env.state.car_status[0]) == CAR_DECISION
    assert int(env.state.car_nodes[0]) == 2
    assert math.isclose(float(result.timestep.reward), -5.0 / 60.0, rel_tol=1e-6)
    assert math.isclose(float(env.state.metrics.aggregate_reward), -5.0 / 60.0, rel_tol=1e-6)


def test_manual_dispatch_env_repositions_to_macro_target_node() -> None:
    graph = dispatch_graph()
    params = make_env_params(
        graph,
        max_cars=1,
        max_requests=1,
        initial_car_nodes=[0],
        manual_dispatch=True,
        episode_seconds=120.0,
    )
    env = ManualDispatchEnv(graph=graph, params=params, seed=1)
    env.reset()

    result = env.submit_dispatch_plan(
        [{"car_id": 0, "action": "reposition", "target_compact_node_id": 2}]
    )

    assert result.action_results[0]["valid"]
    assert int(env.state.car_status[0]) == CAR_DECISION
    assert int(env.state.car_nodes[0]) == 2
    assert float(result.timestep.dt_seconds) == 9.0


def test_manual_dispatch_env_wait_action_advances_to_next_dispatch_window() -> None:
    graph = dispatch_graph()
    params = make_env_params(
        graph,
        max_cars=1,
        max_requests=1,
        initial_car_nodes=[0],
        manual_dispatch=True,
        episode_seconds=120.0,
    )
    env = ManualDispatchEnv(graph=graph, params=params, seed=2, default_wait_seconds=15.0)
    env.reset()

    result = env.submit_dispatch_plan([{"car_id": 0, "action": "wait", "duration_seconds": 12.0}])

    assert result.action_results[0]["valid"]
    assert float(result.timestep.dt_seconds) == 12.0
    assert int(env.state.car_status[0]) == CAR_DECISION
    assert np.asarray(env.state.decision_required).item() is True
