from __future__ import annotations

import argparse
from pathlib import Path

import jax

from .env import EnvParams, FleetEnv
from .graph import build_synthetic_debug_graph, load_ppo_json_graph
from .ppo import (
    PPOConfig,
    collect_rollout_from,
    initialize_env_batch,
    make_train_state,
    mean_recent_pickup_wait,
    mean_recent_request_wait_or_age,
    ppo_update,
    save_checkpoint,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on the event-driven fleet repositioning environment.")
    parser.add_argument(
        "--graph",
        default=None,
        choices=["line", "asymmetric", "directed_assignment", "variable_degree", "grid3"],
        help="Synthetic debug graph. If omitted, train on the actual exported street graph.",
    )
    parser.add_argument("--nodes", default="dist/data/ppo_nodes.json", help="ppo_nodes.json path for the actual graph.")
    parser.add_argument("--edges", default="dist/data/ppo_edges.json", help="ppo_edges.json path for the actual graph.")
    parser.add_argument("--node-limit", default=0, type=int, help="Limit actual graph nodes for debugging; 0 means full graph.")
    parser.add_argument("--route-landmarks", default=64, type=int, help="Number of landmark routes for compact full-graph routing.")
    parser.add_argument("--num-cars", default=16, type=int)
    parser.add_argument("--num-envs", default=16, type=int)
    parser.add_argument("--num-steps", default=128, type=int)
    parser.add_argument("--updates", default=100, type=int)
    parser.add_argument("--demand-rate", default=1.0 / 600.0, type=float)
    parser.add_argument("--learning-rate", default=2.5e-4, type=float)
    parser.add_argument("--checkpoint-dir", default="checkpoints/fleet_ppo")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--log-every", default=1, type=int)
    parser.add_argument("--save-every", default=10, type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.graph:
        graph = build_synthetic_debug_graph(args.graph)
    else:
        graph = load_ppo_json_graph(
            args.nodes,
            args.edges,
            node_limit=None if args.node_limit <= 0 else args.node_limit,
            route_mode="landmark",
            num_landmarks=args.route_landmarks,
        )

    env = FleetEnv(graph)
    env_params = EnvParams.for_graph(
        graph,
        num_cars=args.num_cars,
        demand_rate_per_second=args.demand_rate,
        randomize_start_time=True,
    )
    reset_state, reset_timestep = env.reset(jax.random.PRNGKey(args.seed), env_params)
    del reset_state

    config = PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        total_updates=args.updates,
        minibatch_size=args.num_envs * args.num_steps,
    )
    key = jax.random.PRNGKey(args.seed + 1)
    key, init_key = jax.random.split(key)
    train_state = make_train_state(init_key, reset_timestep.observation, env_params.max_degree, config)
    key, env_key = jax.random.split(key)
    env_states, env_timesteps, key = initialize_env_batch(env, env_params, config, env_key)

    def update_once(state, env_states, env_timesteps, update_key):
        transitions, last_value, env_states, env_timesteps, next_key = collect_rollout_from(
            state,
            env,
            env_params,
            config,
            env_states,
            env_timesteps,
            update_key,
        )
        state, metrics = ppo_update(state, transitions, last_value, config, next_key)
        metrics = {
            **metrics,
            "wait10": mean_recent_request_wait_or_age(env_states, window=10),
            "pickup_wait10": mean_recent_pickup_wait(env_states.metrics, window=10),
        }
        return state, env_states, env_timesteps, metrics

    update_once = jax.jit(update_once)
    checkpoint_dir = Path(args.checkpoint_dir)

    for update in range(1, args.updates + 1):
        key, update_key = jax.random.split(key)
        train_state, env_states, env_timesteps, metrics = update_once(
            train_state,
            env_states,
            env_timesteps,
            update_key,
        )
        metrics = jax.device_get(metrics)
        if update % args.log_every == 0:
            print(
                "update={update} loss={loss:.4f} policy={policy:.4f} value={value:.4f} "
                "entropy={entropy:.4f} reward={reward:.4f} wait10={wait10:.2f}s "
                "pickup_wait10={pickup_wait10:.2f}s".format(
                    update=update,
                    loss=float(metrics["loss"]),
                    policy=float(metrics["policy_loss"]),
                    value=float(metrics["value_loss"]),
                    entropy=float(metrics["entropy"]),
                    reward=float(metrics["rollout_reward"]),
                    wait10=float(metrics["wait10"]),
                    pickup_wait10=float(metrics["pickup_wait10"]),
                )
            )
        if args.save_every > 0 and update % args.save_every == 0:
            save_checkpoint(checkpoint_dir / f"update_{update:06d}.msgpack", train_state)

    save_checkpoint(checkpoint_dir / "latest.msgpack", train_state)


if __name__ == "__main__":
    main()
