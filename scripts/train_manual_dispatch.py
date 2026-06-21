from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from hud import Job, LocalRuntime, Taskset, TrainingClient
from hud.agents import create_agent

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks import tasks as dispatch_tasks


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a HUD/Tinker LLM policy on the manual passenger dispatch task."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HUD trainable model id/name/model_name. Default: %(default)s",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=None,
        help="Task slug to train on. Repeat to include more tasks.",
    )
    parser.add_argument("--steps", type=int, default=1, help="Number of optimizer updates.")
    parser.add_argument(
        "--group-size",
        type=int,
        default=2,
        help="Rollouts per task per update. GRPO advantages are normalized within this group.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--num-substeps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--agent-max-steps",
        type=int,
        default=40,
        help="Maximum LLM/tool turns per rollout.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Concurrent HUD rollouts. Keep 1 for small local demos.",
    )
    parser.add_argument(
        "--rollout-timeout",
        type=float,
        default=180.0,
        help="Wall-clock timeout in seconds for each local rollout.",
    )
    parser.add_argument(
        "--job-name",
        default="manual-dispatch-qwen-training",
        help="HUD job name for the training session.",
    )
    parser.add_argument(
        "--persistent-state-path",
        default="runs/hud/{slug}_state.pkl",
        help="Snapshot path for continuous simulator state. Use {slug} for per-task files.",
    )
    parser.add_argument(
        "--reset-persistent-state",
        action="store_true",
        help="Delete existing persistent state before this training session starts.",
    )
    parser.add_argument(
        "--no-persistent-state",
        action="store_true",
        help="Use standard HUD independent rollouts instead of continuing simulator state.",
    )
    parser.add_argument(
        "--persistent-episode-seconds",
        type=float,
        default=3600.0,
        help="Episode horizon used when persistent state is enabled.",
    )
    parser.add_argument(
        "--persistent-spawn-rate-per-minute",
        type=float,
        default=1.0,
        help="Random request rate used when persistent state is enabled.",
    )
    return parser.parse_args()


def build_taskset(task_ids: list[str]) -> Taskset:
    all_tasks = Taskset("manual-passenger-dispatch", dispatch_tasks)
    unknown = sorted(set(task_ids) - set(all_tasks.tasks))
    if unknown:
        available = ", ".join(all_tasks.tasks)
        raise SystemExit(f"Unknown task id(s): {', '.join(unknown)}. Available: {available}")
    return all_tasks.filter(task_ids)


def state_path_for(pattern: str, slug: str) -> str:
    return pattern.format(slug=slug)


def with_persistent_task_args(taskset: Taskset, args: argparse.Namespace) -> Taskset:
    if args.no_persistent_state:
        return taskset

    updated = []
    for slug, task in taskset.items():
        task_args = dict(task.args)
        task_args.update(
            {
                "persistent_state_path": state_path_for(args.persistent_state_path, slug),
                "episode_seconds": max(
                    float(task_args.get("episode_seconds", 0.0)),
                    float(args.persistent_episode_seconds),
                ),
                "spawn_rate_per_minute": float(args.persistent_spawn_rate_per_minute),
                "reset_persistent_state": False,
            }
        )
        cloned = task.model_copy(update={"args": task_args})
        cloned.slug = task.slug
        cloned.columns = task.columns
        updated.append(cloned)
    return Taskset(taskset.name, updated, origin=taskset.origin)


def reset_persistent_files(taskset: Taskset, pattern: str) -> list[str]:
    removed = []
    for slug in taskset.tasks:
        path = Path(state_path_for(pattern, slug))
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def summarize_rollouts(batch: list[Any]) -> dict[str, Any]:
    rewards = [float(run.reward) for run in batch]
    return {
        "rollouts": len(batch),
        "mean_reward": mean(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "errors": sum(1 for run in batch if getattr(run.trace, "is_error", False)),
        "trace_ids": [run.trace_id for run in batch if run.trace_id],
    }


async def train() -> None:
    args = parse_args()
    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")
    if args.group_size < 1:
        raise SystemExit("--group-size must be >= 1")
    if args.max_concurrent < 1:
        raise SystemExit("--max-concurrent must be >= 1")
    if not args.no_persistent_state and args.max_concurrent != 1:
        raise SystemExit("Persistent simulator state requires --max-concurrent 1")

    task_ids = args.task_id or ["manual-dispatch-balanced"]
    base_taskset = build_taskset(task_ids)
    reset_paths = (
        reset_persistent_files(base_taskset, args.persistent_state_path)
        if args.reset_persistent_state and not args.no_persistent_state
        else []
    )
    taskset = with_persistent_task_args(base_taskset, args)
    agent = create_agent(
        args.model,
        max_steps=args.agent_max_steps,
        completion_kwargs={
            "temperature": args.temperature,
            "extra_body": {"return_token_ids": True},
        },
    )
    trainer = TrainingClient(args.model)
    runtime = LocalRuntime("env.py")
    session = await Job.start(args.job_name, group=args.group_size)

    print(
        json.dumps(
            {
                "event": "start",
                "job_id": session.id,
                "model": args.model,
                "tasks": list(taskset.tasks),
                "steps": args.steps,
                "group_size": args.group_size,
                "persistent_state": not args.no_persistent_state,
                "persistent_state_path": None
                if args.no_persistent_state
                else args.persistent_state_path,
                "reset_persistent_paths": reset_paths,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.steps == 0:
        return

    for update in range(1, args.steps + 1):
        start_index = len(session.runs)
        await taskset.run(
            agent,
            runtime=runtime,
            job=session,
            group=args.group_size,
            max_concurrent=args.max_concurrent,
            rollout_timeout=args.rollout_timeout,
        )
        batch = session.runs[start_index:]
        rollout_summary = summarize_rollouts(batch)
        print(
            json.dumps({"event": "rollouts", "update": update, **rollout_summary}, sort_keys=True),
            flush=True,
        )

        result = await trainer.step(
            batch,
            learning_rate=args.learning_rate,
            group_size=args.group_size,
            reward_scale=args.reward_scale,
            num_substeps=args.num_substeps,
            weight_decay=args.weight_decay,
        )
        print(
            json.dumps(
                {
                    "event": "train_step",
                    "update": update,
                    "checkpoint_id": result.checkpoint_id,
                    "model": result.model,
                    "sampler_path": result.sampler_path,
                    "state_path": result.state_path,
                    "step": result.step,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(train())
