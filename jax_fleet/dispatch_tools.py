from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jax_fleet.dispatch_env import ManualDispatchEnv


def build_tools(get_env: Callable[[], ManualDispatchEnv]):
    from fastmcp import FastMCP

    server = FastMCP(name="manual-dispatch-tools")

    @server.tool
    def get_dispatch_state() -> dict[str, Any]:
        """Return current dispatch state: idle cars, queued requests, and business metrics."""
        return get_env().get_dispatch_state()

    @server.tool
    def get_idle_cars() -> list[dict[str, Any]]:
        """Return cars currently available for manual dispatch decisions."""
        return get_env().get_idle_cars(ready_only=True)

    @server.tool
    def get_open_requests() -> list[dict[str, Any]]:
        """Return queued passenger requests waiting for assignment."""
        return get_env().get_open_requests()

    @server.tool
    def get_eta_matrix(
        car_ids: list[int] | None = None,
        request_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Return pickup ETAs from selected idle cars to selected queued requests."""
        return get_env().get_eta_matrix(car_ids=car_ids, request_ids=request_ids)

    @server.tool
    def get_reposition_targets(count: int = 8) -> list[dict[str, Any]]:
        """Return high-density grid cells that are useful repositioning targets."""
        return get_env().get_reposition_targets(count=count)

    @server.tool
    def simulate_dispatch_plan(actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Preview macro dispatch actions without mutating the simulator."""
        return get_env().simulate_dispatch_plan(actions)

    @server.tool
    def submit_dispatch_plan(actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply macro dispatch actions and advance until the next dispatch decision."""
        result = get_env().submit_dispatch_plan(actions)
        return {
            "reward": float(result.timestep.reward),
            "dt_seconds": float(result.timestep.dt_seconds),
            "done": bool(result.timestep.done),
            "action_results": result.action_results,
            "scene": result.scene,
        }

    @server.tool
    def get_scene() -> dict[str, Any]:
        """Return a JSON-compatible scene for debugging or visualization."""
        return get_env().scene(include_static=False, include_route_previews=True)

    @server.tool
    def get_episode_score() -> dict[str, Any]:
        """Return normalized score and current episode metrics."""
        env = get_env()
        score, details = score_manual_dispatch(env)
        return {"score": score, **details}

    return server


def metric_snapshot(env: ManualDispatchEnv) -> dict[str, float]:
    scene = env.scene(include_static=False, include_route_previews=False)
    metrics = scene["metrics"]
    return {
        "completed_requests": float(metrics.get("completed_requests", 0)),
        "dropped_requests": float(metrics.get("dropped_requests", 0)),
        "active_requests": float(metrics.get("active_requests", 0)),
        "pickup_wait_seconds": float(metrics.get("pickup_wait_seconds", 0.0)),
        "aggregate_reward": float(metrics.get("aggregate_reward", 0.0)),
    }


def score_manual_dispatch(
    env: ManualDispatchEnv,
    *,
    baseline: dict[str, float] | None = None,
) -> tuple[float, dict[str, Any]]:
    metrics = metric_snapshot(env)
    base = baseline or {}
    completed = float(metrics.get("completed_requests", 0))
    dropped = float(metrics.get("dropped_requests", 0))
    active = float(metrics.get("active_requests", 0))
    pickup_wait = float(metrics.get("pickup_wait_seconds", 0.0))
    if baseline is not None:
        completed = max(0.0, completed - float(base.get("completed_requests", 0.0)))
        dropped = max(0.0, dropped - float(base.get("dropped_requests", 0.0)))
        pickup_wait = max(0.0, pickup_wait - float(base.get("pickup_wait_seconds", 0.0)))
    avg_wait = pickup_wait / max(1.0, completed)
    total_known = max(1.0, completed + dropped + active)
    service_rate = completed / total_known
    wait_score = max(0.0, 1.0 - min(avg_wait, 240.0) / 240.0)
    drop_penalty = min(0.35, dropped * 0.08)
    score = max(0.0, min(1.0, 0.15 + service_rate * 0.55 + wait_score * 0.30 - drop_penalty))
    return round(score, 4), {
        "completed_requests": int(completed),
        "dropped_requests": int(dropped),
        "active_requests": int(active),
        "pickup_wait_seconds": round(pickup_wait, 3),
        "average_pickup_wait_seconds": round(avg_wait, 3),
        "service_rate": round(service_rate, 4),
        "aggregate_reward": float(metrics.get("aggregate_reward", 0.0)),
        "segment_aggregate_reward": round(
            float(metrics.get("aggregate_reward", 0.0))
            - float(base.get("aggregate_reward", 0.0)),
            6,
        )
        if baseline is not None
        else float(metrics.get("aggregate_reward", 0.0)),
    }
