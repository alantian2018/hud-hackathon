from __future__ import annotations

import argparse
import asyncio
import os


async def train(args: argparse.Namespace) -> None:
    if not os.environ.get("HUD_API_KEY"):
        raise SystemExit("HUD_API_KEY must be set in the environment before training.")

    from hud import Taskset, TrainingClient
    from hud.agents import create_agent
    from hud.eval import Job, LocalRuntime

    from hud_mobility.tasks import taskset as full_taskset

    taskset = full_taskset
    if args.limit_tasks is not None:
        taskset = Taskset(
            f"{full_taskset.name}-first-{args.limit_tasks}",
            list(full_taskset)[: args.limit_tasks],
        )

    agent = create_agent(
        args.model,
        completion_kwargs={"extra_body": {"return_token_ids": True}},
    )
    trainer = TrainingClient(args.model)
    runtime = LocalRuntime("hud_mobility/env.py", ready_timeout=120.0)
    session = await Job.start(args.job_name, group=args.group)

    for step in range(args.steps):
        start = len(session.runs)
        await taskset.run(
            agent,
            runtime=runtime,
            job=session,
            group=args.group,
            max_concurrent=args.max_concurrent,
        )
        batch = session.runs[start:]
        reward = sum(run.reward for run in batch) / max(1, len(batch))
        print(f"step={step} rollouts={len(batch)} mean_reward={reward:.4f}")
        result = await trainer.step(
            batch,
            learning_rate=args.learning_rate,
            group_size=args.group,
        )
        print(f"checkpoint={result.checkpoint_id} model={result.model}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a HUD LLM orchestrator on the mobility taskset.")
    parser.add_argument("--model", required=True, help="Trainable HUD gateway model slug, e.g. mobility-rl.")
    parser.add_argument("--job-name", default="mobility-orchestrator-rl")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    asyncio.run(train(args))


if __name__ == "__main__":
    main()
