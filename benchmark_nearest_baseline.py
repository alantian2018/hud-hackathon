from __future__ import annotations

import argparse
import json
import statistics
from typing import Callable

from hud_mobility.planners import build_value_aware_plan
from hud_mobility.schemas import ActionPlan, AssignmentAction
from hud_mobility.tasks import TASK_SPECS
from hud_mobility.world import MobilityWorld


Policy = Callable[[MobilityWorld], ActionPlan]


def nearest_car_baseline(world: MobilityWorld) -> ActionPlan:
    """Evaluation-only nearest-car policy, kept outside the HUD reward path."""
    world.prepare_current_step()
    idle = list(world.idle_cars())
    assignments: list[AssignmentAction] = []
    for request in world.waiting_requests():
        best = None
        for idx, car in enumerate(idle):
            route = world.router.route(car.position, request.person.origin, world.last_traffic_heatmap)
            if best is None or route.cost < best[0]:
                best = (route.cost, idx, car.id)
        if best is None:
            continue
        _cost, idx, car_id = best
        idle.pop(idx)
        assignments.append(AssignmentAction(car_id=car_id, person_id=request.person.id))
    return ActionPlan(assignments=assignments, rationale="Evaluation-only nearest-car baseline.")


def run_episode(policy: Policy, spec: dict, fleet_size: int) -> dict:
    world = MobilityWorld(
        (14, 14),
        fleet_size=fleet_size,
        seed=int(spec["seed"]),
        start_minute=int(spec["start_minute"]),
        horizon_steps=int(spec["horizon_steps"]),
        event_id=spec["event_id"],
        demand_scale=float(spec["demand_scale"]),
    )
    while not world.done:
        world.step(policy(world))
    return world.reward()


def summarize(results: list[dict]) -> dict:
    rewards = [float(item["reward"]) for item in results]
    return {
        "mean_reward": round(statistics.mean(rewards), 6),
        "reward_std": round(statistics.pstdev(rewards), 6) if len(rewards) > 1 else 0.0,
        "min_reward": round(min(rewards), 6),
        "max_reward": round(max(rewards), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the orchestrator planner against a nearest-car baseline.")
    parser.add_argument("--fleet-size", type=int, default=20)
    parser.add_argument("--jsonl", action="store_true", help="Print per-task JSON lines before the summary.")
    args = parser.parse_args()

    rows = []
    baseline_results = []
    orchestrator_results = []
    for idx, spec in enumerate(TASK_SPECS):
        baseline = run_episode(nearest_car_baseline, spec, args.fleet_size)
        orchestrator = run_episode(build_value_aware_plan, spec, args.fleet_size)
        baseline_results.append(baseline)
        orchestrator_results.append(orchestrator)
        row = {
            "task_index": idx,
            "seed": spec["seed"],
            "event_id": spec["event_id"],
            "nearest_baseline_reward": baseline["reward"],
            "orchestrator_planner_reward": orchestrator["reward"],
            "delta": round(float(orchestrator["reward"]) - float(baseline["reward"]), 6),
        }
        rows.append(row)
        if args.jsonl:
            print(json.dumps(row, sort_keys=True))

    baseline_summary = summarize(baseline_results)
    orchestrator_summary = summarize(orchestrator_results)
    summary = {
        "tasks": len(TASK_SPECS),
        "nearest_baseline": baseline_summary,
        "orchestrator_planner": orchestrator_summary,
        "mean_delta": round(orchestrator_summary["mean_reward"] - baseline_summary["mean_reward"], 6),
        "wins": sum(1 for row in rows if row["delta"] > 0.0),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
