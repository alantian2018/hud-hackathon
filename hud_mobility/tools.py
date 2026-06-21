from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from .planners import build_value_aware_plan, critique_plan, global_batch_matching, reposition_targets
from .schemas import ActionPlan
from .world import EVENT_PRESETS, MobilityWorld


EPISODES: dict[str, MobilityWorld] = {}


def create_episode(
    *,
    seed: int = 7,
    fleet_size: int = 20,
    start_minute: int = 8 * 60,
    step_minutes: int = 5,
    horizon_steps: int = 12,
    event_id: str | None = None,
    demand_scale: float = 1.0,
    rows: int = 14,
    cols: int = 14,
) -> str:
    episode_id = f"mobility-{uuid4().hex[:12]}"
    EPISODES[episode_id] = MobilityWorld(
        (rows, cols),
        seed=seed,
        fleet_size=fleet_size,
        start_minute=start_minute,
        step_minutes=step_minutes,
        horizon_steps=horizon_steps,
        event_id=event_id,
        demand_scale=demand_scale,
    )
    return episode_id


def get_episode(episode_id: str) -> MobilityWorld:
    if episode_id not in EPISODES:
        raise KeyError(f"unknown episode_id: {episode_id}")
    return EPISODES[episode_id]


def close_episode(episode_id: str) -> None:
    EPISODES.pop(episode_id, None)


def observe_state(episode_id: str) -> dict[str, Any]:
    """Return the current fleet, request, traffic, demand, and metric state."""
    return get_episode(episode_id).observe()


def forecast_hotspots(episode_id: str, lookahead_steps: int = 3, k: int = 8) -> dict[str, Any]:
    """Forecast high-demand grid cells for upcoming dispatch steps."""
    world = get_episode(episode_id)
    return {
        "episode_id": episode_id,
        "timestep": world.timestep,
        "hotspots": world.forecast_hotspots(lookahead_steps=lookahead_steps, k=k),
    }


def propose_matching(episode_id: str) -> dict[str, Any]:
    """Return value/urgency-aware global assignment candidates."""
    world = get_episode(episode_id)
    pairs = global_batch_matching(world)
    return {
        "episode_id": episode_id,
        "assignments": [
            {"car_id": pair.car_id, "person_id": pair.person_id, "score": round(pair.score, 4)}
            for pair in pairs
        ],
        "pair_scores": [pair.to_dict() for pair in pairs],
    }


def propose_repositioning(episode_id: str, assigned_car_ids_json: list[str] | str | None = None) -> dict[str, Any]:
    """Return proactive idle-car repositioning targets for future demand."""
    if isinstance(assigned_car_ids_json, str):
        raw = json.loads(assigned_car_ids_json or "[]")
    else:
        raw = assigned_car_ids_json or []
    assigned = {str(item) for item in raw}
    world = get_episode(episode_id)
    targets = reposition_targets(world, assigned_car_ids=assigned)
    return {
        "episode_id": episode_id,
        "repositions": [target.to_dict() for target in targets],
    }


def propose_full_plan(episode_id: str) -> dict[str, Any]:
    """Return a complete non-baseline plan that the orchestrator may edit."""
    world = get_episode(episode_id)
    plan = build_value_aware_plan(world)
    return {"episode_id": episode_id, "plan": plan.to_dict(), "critique": critique_plan(world, plan)}


def critique_action_plan(episode_id: str, plan_json: dict[str, Any] | str | None = None) -> dict[str, Any]:
    """Validate and summarize likely risk in a candidate action plan object or JSON string."""
    world = get_episode(episode_id)
    plan = ActionPlan.from_any(plan_json)
    return {"episode_id": episode_id, "critique": critique_plan(world, plan)}


def step_world(episode_id: str, plan_json: dict[str, Any] | str | None = None) -> dict[str, Any]:
    """Apply one action plan object or JSON string and advance the simulator by one step."""
    world = get_episode(episode_id)
    plan = ActionPlan.from_any(plan_json)
    return _apply_plan_and_report(episode_id, world, plan)


def execute_recommended_step(episode_id: str) -> dict[str, Any]:
    """Advance one step using the non-greedy specialist planner's current best plan."""
    world = get_episode(episode_id)
    plan = build_value_aware_plan(world)
    result = _apply_plan_and_report(episode_id, world, plan)
    result["recommended_plan"] = plan.to_dict()
    return result


def run_recommended_episode(episode_id: str) -> dict[str, Any]:
    """Run the full remaining horizon with the non-greedy specialist planner."""
    world = get_episode(episode_id)
    steps = 0
    while not world.done:
        world.step(build_value_aware_plan(world))
        steps += 1
    return {"episode_id": episode_id, "steps_executed": steps, **world.reward()}


def _apply_plan_and_report(episode_id: str, world: MobilityWorld, plan: ActionPlan) -> dict[str, Any]:
    observation = world.step(plan)
    plan_counts = {
        "assignments": len(plan.assignments),
        "repositions": len(plan.repositions),
        "holds": len(plan.holds),
    }
    return {
        "episode_id": episode_id,
        "applied_plan_counts": plan_counts,
        "reward_so_far": observation["reward_so_far"],
        "done": observation["done"],
        "remaining_steps": observation["remaining_steps"],
        "observation": observation,
    }


def submit_episode(episode_id: str) -> dict[str, Any]:
    """Return the final absolute reward and metrics for the episode."""
    world = get_episode(episode_id)
    return {"episode_id": episode_id, **world.reward()}


def list_scenarios() -> dict[str, Any]:
    """List supported event scenarios."""
    return {"events": sorted(EVENT_PRESETS)}


def build_mcp_server():
    from fastmcp import FastMCP

    server = FastMCP(name="mobility-orchestrator-tools")
    server.tool(observe_state)
    server.tool(forecast_hotspots)
    server.tool(propose_matching)
    server.tool(propose_repositioning)
    server.tool(propose_full_plan)
    server.tool(critique_action_plan)
    server.tool(step_world)
    server.tool(execute_recommended_step)
    server.tool(run_recommended_episode)
    server.tool(submit_episode)
    server.tool(list_scenarios)
    return server
