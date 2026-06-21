from __future__ import annotations

import argparse
import json
from pathlib import Path

from jax_fleet.devices import jax_device_summary
from jax_fleet.graph import load_public_data_graph
from jax_fleet.ppo.train import TrainingConfig, benchmark_env_steps, train

DEFAULT_DATA_DIR = "dist/data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and prepare the JAX fleet backend.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gpu = subparsers.add_parser("check-gpu", help="Print JAX device information and optionally require a GPU.")
    gpu.add_argument("--require-gpu", action="store_true", help="Exit nonzero unless JAX sees a GPU device.")

    routing = subparsers.add_parser("prepare-routing", help="Build or validate the SF routing cache.")
    routing.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    routing.add_argument("--cache-dir", default="cache/jax_fleet")
    routing.add_argument("--chunk-size", default=512, type=int)

    train_parser = subparsers.add_parser("train", help="Run PPO training.")
    train_parser.add_argument("--graph", choices=["synthetic", "sf"], default="sf")
    train_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    train_parser.add_argument("--routing-cache-dir", default="cache/jax_fleet")
    train_parser.add_argument("--routing-chunk-size", default=512, type=int)
    train_parser.add_argument("--seed", default=0, type=int)
    train_parser.add_argument("--num-envs", default=4, type=int)
    train_parser.add_argument("--num-steps", default=16, type=int)
    train_parser.add_argument("--num-updates", default=1, type=int)
    train_parser.add_argument("--max-cars", default=40, type=int)
    train_parser.add_argument("--max-requests", default=32, type=int)
    train_parser.add_argument("--assignment-max-route-edges", default=10000, type=int)
    train_parser.add_argument("--episode-seconds", default=3600.0, type=float)
    train_parser.add_argument("--spawn-rate-per-minute", default=0.0, type=float)
    train_parser.add_argument("--spawn-source", choices=["uniform", "density", "js-visual"], default=None)
    train_parser.add_argument("--reward-mode", choices=["dense_wait", "legacy_pickup_wait"], default="dense_wait")
    train_parser.add_argument("--observation-mode", choices=["learning_v1", "legacy"], default="learning_v1")
    train_parser.add_argument("--drop-penalty", default=10.0, type=float)
    train_parser.add_argument("--pickup-bonus", default=0.0, type=float)
    train_parser.add_argument("--time-discount-reference-seconds", default=60.0, type=float)
    train_parser.add_argument("--learning-rate", default=3e-4, type=float)
    train_parser.add_argument("--update-epochs", default=4, type=int)
    train_parser.add_argument("--num-minibatches", default=4, type=int)
    train_parser.add_argument("--checkpoint-dir", default="runs/jax_fleet/checkpoints")
    train_parser.add_argument("--checkpoint-every", default=1, type=int)
    train_parser.add_argument("--metrics-path", default="runs/jax_fleet/metrics.jsonl")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--require-gpu", action="store_true", help="Fail fast if JAX cannot see a GPU.")
    train_parser.add_argument("--track", action="store_true", help="Log metrics to Weights & Biases.")
    train_parser.add_argument("--wandb-project-name", default="jax_fleet")
    train_parser.add_argument("--wandb-entity", default=None)
    train_parser.add_argument("--wandb-run-name", default=None)
    train_parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
        help="Optional W&B mode passed to wandb.init.",
    )
    train_parser.add_argument(
        "--wandb-video-every",
        default=0,
        type=int,
        help="Upload a rendered policy diagnostic video every N updates. 0 disables video logging.",
    )
    train_parser.add_argument("--wandb-video-max-steps", default=50000, type=int)
    train_parser.add_argument("--wandb-video-max-pickups", default=20, type=int)
    train_parser.add_argument("--wandb-video-max-frames", default=240, type=int)
    train_parser.add_argument("--wandb-video-width", default=960, type=int)
    train_parser.add_argument("--wandb-video-height", default=600, type=int)
    train_parser.add_argument("--wandb-video-fps", default=12, type=int)

    benchmark = subparsers.add_parser("benchmark-env", help="Measure compiled env steps/sec with rendering disabled.")
    benchmark.add_argument("--graph", choices=["synthetic", "sf"], default="sf")
    benchmark.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    benchmark.add_argument("--routing-cache-dir", default="cache/jax_fleet")
    benchmark.add_argument("--routing-chunk-size", default=512, type=int)
    benchmark.add_argument("--seed", default=0, type=int)
    benchmark.add_argument("--num-envs", default=4, type=int)
    benchmark.add_argument("--steps", default=256, type=int)
    benchmark.add_argument("--max-cars", default=40, type=int)
    benchmark.add_argument("--max-requests", default=32, type=int)
    benchmark.add_argument("--assignment-max-route-edges", default=10000, type=int)
    benchmark.add_argument("--episode-seconds", default=3600.0, type=float)
    benchmark.add_argument("--spawn-rate-per-minute", default=0.0, type=float)
    benchmark.add_argument("--spawn-source", choices=["uniform", "density", "js-visual"], default=None)
    benchmark.add_argument("--reward-mode", choices=["dense_wait", "legacy_pickup_wait"], default="dense_wait")
    benchmark.add_argument("--observation-mode", choices=["learning_v1", "legacy"], default="learning_v1")
    benchmark.add_argument("--drop-penalty", default=10.0, type=float)
    benchmark.add_argument("--pickup-bonus", default=0.0, type=float)
    benchmark.add_argument("--time-discount-reference-seconds", default=60.0, type=float)
    benchmark.add_argument("--require-gpu", action="store_true", help="Fail fast if JAX cannot see a GPU.")
    return parser


def _gpu_requirement_failed() -> dict[str, object] | None:
    payload = jax_device_summary()
    if payload["gpu_available"]:
        return None
    return {
        **payload,
        "ok": False,
        "error": (
            "JAX did not find a GPU device. Install this project with the `gpu` extra "
            "and verify the NVIDIA driver."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check-gpu":
        payload = jax_device_summary()
        payload["ok"] = bool(payload["gpu_available"] or not args.require_gpu)
        if args.require_gpu and not payload["gpu_available"]:
            payload["error"] = (
                "JAX did not find a GPU device. Install this project with the `gpu` extra "
                "and verify the NVIDIA driver."
            )
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["ok"] else 1

    if args.command == "prepare-routing":
        def progress(done: int, total: int) -> None:
            print(json.dumps({"event": "routing_progress", "done": done, "total": total}), flush=True)

        graph = load_public_data_graph(
            args.data_dir,
            include_routing=True,
            cache_dir=args.cache_dir,
            routing_chunk_size=args.chunk_size,
            routing_progress=progress,
        )
        payload = {
            "graph": "sf_largest_scc",
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "max_degree": graph.max_degree,
            "cache_dir": str(Path(args.cache_dir)),
        }
        print(json.dumps(payload, sort_keys=True))
        return 0

    if args.command == "train":
        if args.require_gpu:
            failed = _gpu_requirement_failed()
            if failed is not None:
                print(json.dumps(failed, sort_keys=True))
                return 1

        config = TrainingConfig(
            graph_name=args.graph,
            data_dir=Path(args.data_dir),
            routing_cache_dir=Path(args.routing_cache_dir),
            routing_chunk_size=args.routing_chunk_size,
            seed=args.seed,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            num_updates=args.num_updates,
            max_cars=args.max_cars,
            max_requests=args.max_requests,
            assignment_max_route_edges=args.assignment_max_route_edges,
            episode_seconds=args.episode_seconds,
            spawn_rate_per_minute=args.spawn_rate_per_minute,
            spawn_source=args.spawn_source,
            reward_mode=args.reward_mode,
            observation_mode=args.observation_mode,
            drop_penalty=args.drop_penalty,
            pickup_bonus=args.pickup_bonus,
            time_discount_reference_seconds=args.time_discount_reference_seconds,
            learning_rate=args.learning_rate,
            update_epochs=args.update_epochs,
            num_minibatches=args.num_minibatches,
            checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
            checkpoint_every=args.checkpoint_every,
            metrics_path=Path(args.metrics_path) if args.metrics_path else None,
            resume=args.resume,
            track=args.track,
            wandb_project_name=args.wandb_project_name,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_mode=args.wandb_mode,
            wandb_video_every=args.wandb_video_every,
            wandb_video_max_steps=args.wandb_video_max_steps,
            wandb_video_max_pickups=args.wandb_video_max_pickups,
            wandb_video_max_frames=args.wandb_video_max_frames,
            wandb_video_width=args.wandb_video_width,
            wandb_video_height=args.wandb_video_height,
            wandb_video_fps=args.wandb_video_fps,
            require_gpu=args.require_gpu,
        )
        print(json.dumps(train(config), sort_keys=True))
        return 0

    if args.command == "benchmark-env":
        if args.require_gpu:
            failed = _gpu_requirement_failed()
            if failed is not None:
                print(json.dumps(failed, sort_keys=True))
                return 1

        config = TrainingConfig(
            graph_name=args.graph,
            data_dir=Path(args.data_dir),
            routing_cache_dir=Path(args.routing_cache_dir),
            routing_chunk_size=args.routing_chunk_size,
            seed=args.seed,
            num_envs=args.num_envs,
            max_cars=args.max_cars,
            max_requests=args.max_requests,
            assignment_max_route_edges=args.assignment_max_route_edges,
            episode_seconds=args.episode_seconds,
            spawn_rate_per_minute=args.spawn_rate_per_minute,
            spawn_source=args.spawn_source,
            reward_mode=args.reward_mode,
            observation_mode=args.observation_mode,
            drop_penalty=args.drop_penalty,
            pickup_bonus=args.pickup_bonus,
            time_discount_reference_seconds=args.time_discount_reference_seconds,
            checkpoint_dir=None,
            metrics_path=None,
            require_gpu=args.require_gpu,
        )
        print(json.dumps(benchmark_env_steps(config, steps=args.steps), sort_keys=True))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
