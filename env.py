from __future__ import annotations

import asyncio
from pathlib import Path
import socket
from typing import Any

from hud import Environment
from hud.capabilities import Capability

from jax_fleet.dispatch_env import ManualDispatchEnv
from jax_fleet.dispatch_tools import build_tools, metric_snapshot, score_manual_dispatch
from jax_fleet.env import make_env_params
from jax_fleet.graph import build_synthetic_graph


MCP_HOST = "127.0.0.1"

env = Environment(name="manual-passenger-dispatch", version="0.1.0")

_dispatch_env: ManualDispatchEnv | None = None
_tools_task: asyncio.Task | None = None
_mcp_port: int | None = None


def _get_dispatch_env() -> ManualDispatchEnv:
    if _dispatch_env is None:
        raise RuntimeError("No active dispatch simulation. Start a HUD task first.")
    return _dispatch_env


tools = build_tools(_get_dispatch_env)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((MCP_HOST, 0))
        return int(sock.getsockname()[1])


def build_dispatch_graph():
    return build_synthetic_graph(
        node_lonlat=[
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.8, 1.0),
            (1.5, -0.8),
            (3.0, -0.2),
        ],
        edges=[
            {"source": 0, "target": 1, "travel_time_s": 5.0},
            {"source": 0, "target": 3, "travel_time_s": 8.0},
            {"source": 1, "target": 0, "travel_time_s": 5.0},
            {"source": 1, "target": 2, "travel_time_s": 4.0},
            {"source": 1, "target": 4, "travel_time_s": 6.0},
            {"source": 2, "target": 1, "travel_time_s": 5.0},
            {"source": 2, "target": 5, "travel_time_s": 5.0},
            {"source": 3, "target": 1, "travel_time_s": 3.0},
            {"source": 3, "target": 4, "travel_time_s": 4.0},
            {"source": 4, "target": 0, "travel_time_s": 6.0},
            {"source": 4, "target": 2, "travel_time_s": 3.0},
            {"source": 5, "target": 0, "travel_time_s": 10.0},
            {"source": 5, "target": 2, "travel_time_s": 5.0},
        ],
    )


def build_preplanned_requests(scenario: str) -> list[dict[str, Any]]:
    base = [
        {"spawn_time_s": 0.0, "origin": 1, "destination": 5, "patience_s": 120.0},
        {"spawn_time_s": 0.0, "origin": 3, "destination": 2, "patience_s": 120.0},
        {"spawn_time_s": 20.0, "origin": 4, "destination": 0, "patience_s": 140.0},
        {"spawn_time_s": 45.0, "origin": 2, "destination": 3, "patience_s": 150.0},
        {"spawn_time_s": 80.0, "origin": 5, "destination": 1, "patience_s": 170.0},
    ]
    if scenario == "surge":
        return [
            *base,
            {"spawn_time_s": 10.0, "origin": 1, "destination": 4, "patience_s": 95.0},
            {"spawn_time_s": 55.0, "origin": 3, "destination": 5, "patience_s": 140.0},
        ]
    return base


def create_manual_dispatch_env(
    *,
    seed: int,
    scenario: str,
    max_cars: int,
    max_requests: int,
    episode_seconds: float,
    spawn_rate_per_minute: float = 0.0,
) -> ManualDispatchEnv:
    graph = build_dispatch_graph()
    initial_car_nodes = [0, 4, 5, 3][:max_cars]
    params = make_env_params(
        graph,
        max_cars=max_cars,
        max_requests=max_requests,
        initial_car_nodes=initial_car_nodes,
        preplanned_requests=build_preplanned_requests(scenario),
        manual_dispatch=True,
        episode_seconds=episode_seconds,
        spawn_rate_per_minute=spawn_rate_per_minute,
        wait_time_scale=1.0 / 60.0,
    )
    dispatch_env = ManualDispatchEnv(
        graph=graph,
        params=params,
        seed=seed,
        default_wait_seconds=20.0,
    )
    dispatch_env.reset()
    return dispatch_env


@env.initialize
async def _start_tools() -> None:
    global _mcp_port, _tools_task
    if _tools_task is None:
        _mcp_port = _free_port()
        _tools_task = asyncio.create_task(
            tools.run_async(transport="http", host=MCP_HOST, port=_mcp_port)
        )
        await asyncio.sleep(1.0)
    env.add_capability(Capability.mcp(name="tools", url=f"http://{MCP_HOST}:{_mcp_port}/mcp"))


@env.shutdown
async def _stop_tools() -> None:
    global _tools_task
    if _tools_task is not None:
        _tools_task.cancel()
        _tools_task = None


@env.template(id="manual_dispatch_episode")
async def manual_dispatch_episode(
    seed: int = 1,
    scenario: str = "balanced",
    max_cars: int = 3,
    max_requests: int = 8,
    episode_seconds: float = 240.0,
    max_dispatch_rounds: int = 12,
    spawn_rate_per_minute: float = 0.0,
    persistent_state_path: str | None = None,
    reset_persistent_state: bool = False,
):
    global _dispatch_env
    state_event: dict[str, Any] = {"mode": "fresh"}
    if persistent_state_path and reset_persistent_state:
        Path(persistent_state_path).unlink(missing_ok=True)

    _dispatch_env = create_manual_dispatch_env(
        seed=seed,
        scenario=scenario,
        max_cars=max_cars,
        max_requests=max_requests,
        episode_seconds=episode_seconds,
        spawn_rate_per_minute=spawn_rate_per_minute,
    )
    if persistent_state_path and Path(persistent_state_path).exists():
        state_event = {"mode": "resumed", **_dispatch_env.load_snapshot(persistent_state_path)}

    score_baseline = metric_snapshot(_dispatch_env)

    answer = yield f"""
You are the passenger dispatch controller for a mobility fleet.

Scenario: {scenario}
State mode: {state_event["mode"]}
Current simulator time: {_dispatch_env.get_dispatch_state()["time_seconds"]:.1f} seconds
Goal: minimize passenger pickup wait time while completing as many requests as possible.

You control macro dispatch actions, not turn-by-turn driving. Repeat for up to {max_dispatch_rounds} dispatch rounds or until done:
1. Call get_dispatch_state.
2. Call get_eta_matrix to compare idle cars against queued requests.
3. Assign idle cars to queued requests when there is a sensible match.
4. Reposition idle cars only when no good request is waiting or to prepare for likely demand.
5. Use submit_dispatch_plan with actions like:
   {{"car_id": 0, "action": "assign_request", "request_id": 1}}
   {{"car_id": 2, "action": "reposition", "grid_cell": [0, 1]}}
   {{"car_id": 1, "action": "wait", "duration_seconds": 20}}

Prefer assigning older requests, shorter pickup ETAs, and keeping at least one car near central demand.
Return a concise final summary after the episode is done or after your dispatch rounds.
The score is computed from simulator metrics, not from wording.
"""

    _ = answer
    score, _details = score_manual_dispatch(_dispatch_env, baseline=score_baseline)
    if persistent_state_path:
        _dispatch_env.save_snapshot(persistent_state_path)
    yield score


if __name__ == "__main__":
    sim = create_manual_dispatch_env(
        seed=1,
        scenario="balanced",
        max_cars=3,
        max_requests=8,
        episode_seconds=240.0,
    )
    while not sim.done:
        state = sim.get_dispatch_state()
        if not state["idle_cars"]:
            break
        eta_entries = sim.get_eta_matrix()["entries"]
        actions = []
        used_requests: set[int] = set()
        for car in state["idle_cars"]:
            car_id = car["car_id"]
            options = [
                item
                for item in eta_entries
                if item["car_id"] == car_id
                and item["request_id"] not in used_requests
                and item["reachable"]
            ]
            if options:
                best = min(options, key=lambda item: item["pickup_eta_seconds"])
                used_requests.add(best["request_id"])
                actions.append(
                    {"car_id": car_id, "action": "assign_request", "request_id": best["request_id"]}
                )
            else:
                actions.append({"car_id": car_id, "action": "wait", "duration_seconds": 20})
        sim.submit_dispatch_plan(actions)
    score, details = score_manual_dispatch(sim)
    print({"score": score, **details})
