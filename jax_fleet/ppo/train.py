from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import partial
import json
from pathlib import Path
from typing import Any

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from jax_fleet.env import reset, step
from jax_fleet.graph import build_synthetic_graph, load_public_data_graph
from jax_fleet.ppo.model import ActorCritic
from jax_fleet.spawns import make_spawned_env_params


class TrainState(train_state.TrainState):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    graph_name: str = "synthetic"
    data_dir: Path | str = Path("public/data")
    routing_cache_dir: Path | str = Path("cache/jax_fleet")
    routing_chunk_size: int = 512
    seed: int = 0
    num_envs: int = 4
    num_steps: int = 16
    num_updates: int = 1
    max_cars: int = 1
    max_requests: int = 16
    assignment_max_route_edges: int = 15
    episode_seconds: float = 3600.0
    spawn_rate_per_minute: float = 0.0
    spawn_source: str | None = None
    learning_rate: float = 3e-4
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    checkpoint_dir: Path | str | None = Path("runs/jax_fleet/checkpoints")
    checkpoint_every: int = 1
    metrics_path: Path | str | None = Path("runs/jax_fleet/metrics.jsonl")
    resume: bool = False

    def replace(self, **changes):
        return replace(self, **changes)

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("data_dir", "routing_cache_dir", "checkpoint_dir", "metrics_path"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        return payload


def default_smoke_graph():
    return build_synthetic_graph(
        node_lonlat=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        edges=[
            {"source": 0, "target": 1, "travel_time_s": 2.0},
            {"source": 1, "target": 2, "travel_time_s": 2.0},
            {"source": 2, "target": 0, "travel_time_s": 2.0},
        ],
    )


def train_smoke(
    *,
    graph=None,
    seed: int = 0,
    num_envs: int = 4,
    num_steps: int = 16,
    num_updates: int = 1,
    learning_rate: float = 3e-4,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    config = TrainingConfig(
        seed=seed,
        num_envs=num_envs,
        num_steps=num_steps,
        num_updates=num_updates,
        learning_rate=learning_rate,
        max_cars=1,
        max_requests=4,
        episode_seconds=240.0,
        metrics_path=metrics_path,
        checkpoint_dir=None,
    )
    return train(config, graph=graph or default_smoke_graph())


def train(config: TrainingConfig, *, graph=None) -> dict[str, Any]:
    graph = graph or _load_graph_from_config(config)
    spawn_source = config.spawn_source or ("density" if config.graph_name == "sf" else "uniform")
    initial_nodes = (
        None
        if spawn_source in {"js-visual", "density"}
        else _default_initial_car_nodes(graph.num_nodes, config.max_cars, config.seed)
    )
    env_params = make_spawned_env_params(
        graph,
        graph_name=config.graph_name,
        data_dir=config.data_dir,
        spawn_source=spawn_source,
        seed=config.seed,
        max_cars=config.max_cars,
        max_requests=config.max_requests,
        initial_car_nodes=initial_nodes,
        assignment_max_route_edges=config.assignment_max_route_edges,
        episode_seconds=config.episode_seconds,
        spawn_rate_per_minute=config.spawn_rate_per_minute,
    )
    rng = jax.random.PRNGKey(config.seed)
    rng, reset_rng, init_rng = jax.random.split(rng, 3)
    reset_keys = jax.random.split(reset_rng, config.num_envs)
    states, timesteps = jax.vmap(lambda key: reset(key, env_params))(reset_keys)

    model = ActorCritic(max_degree=graph.max_degree)
    variables = model.init(init_rng, timesteps.observation)
    tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))
    learner = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)

    start_update = 0
    checkpoint_dir = _resolve_optional_path(config.checkpoint_dir)
    metrics_path = _resolve_optional_path(config.metrics_path)
    checkpointer = ocp.PyTreeCheckpointer() if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "training_config.json").write_text(
            json.dumps(config.to_jsonable(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if config.resume and checkpoint_dir is not None:
        latest = latest_checkpoint_path(checkpoint_dir)
        if latest is not None:
            restore_target = {
                "params": learner.params,
                "opt_state": learner.opt_state,
                "train_step": learner.step,
                "rng": rng,
                "update": start_update,
            }
            restored = checkpointer.restore(latest, item=restore_target)
            learner = learner.replace(
                params=restored["params"],
                opt_state=restored["opt_state"],
                step=restored["train_step"],
            )
            rng = restored["rng"]
            start_update = int(restored["update"])

    metrics: dict[str, Any] = {
        "updates": start_update,
        "last_loss": 0.0,
        "last_mean_reward": 0.0,
        "latest_checkpoint": str(latest_checkpoint_path(checkpoint_dir)) if checkpoint_dir else None,
    }

    for update in range(start_update + 1, config.num_updates + 1):
        rollout = _collect_rollout(
            learner=learner,
            states=states,
            timesteps=timesteps,
            env_params=env_params,
            rng=rng,
            num_steps=config.num_steps,
        )
        states = rollout["states"]
        timesteps = rollout["timesteps"]
        rng = rollout["rng"]
        bootstrap_value = learner.apply_fn({"params": learner.params}, timesteps.observation)[1]
        advantages, returns = compute_gae(
            rewards=rollout["rewards"],
            values=rollout["values"],
            bootstrap_value=bootstrap_value,
            discounts=rollout["discounts"],
            dones=rollout["dones"],
            gae_lambda=config.gae_lambda,
        )

        batch_obs = _flatten_time_env_tree(rollout["observations"])
        batch = {
            "actions": rollout["actions"].reshape((-1,)),
            "old_log_probs": rollout["log_probs"].reshape((-1,)),
            "advantages": advantages.reshape((-1,)),
            "returns": returns.reshape((-1,)),
            "values": rollout["values"].reshape((-1,)),
        }
        rng, update_rng = jax.random.split(rng)
        learner, loss_metrics = _ppo_update(learner, batch_obs, batch, config, update_rng)
        metrics = {
            "update": update,
            "updates": update,
            "last_loss": float(loss_metrics["loss"]),
            "last_policy_loss": float(loss_metrics["policy_loss"]),
            "last_value_loss": float(loss_metrics["value_loss"]),
            "last_entropy": float(loss_metrics["entropy"]),
            "last_approx_kl": float(loss_metrics["approx_kl"]),
            "last_clipfrac": float(loss_metrics["clipfrac"]),
            "last_mean_reward": float(jnp.asarray(rollout["rewards"]).mean()),
            "latest_checkpoint": None,
        }
        if checkpoint_dir is not None and config.checkpoint_every > 0 and update % config.checkpoint_every == 0:
            save_path = checkpoint_path(checkpoint_dir, update)
            checkpointer.save(
                save_path,
                {
                    "params": learner.params,
                    "opt_state": learner.opt_state,
                    "train_step": learner.step,
                    "rng": rng,
                    "update": update,
                },
                force=True,
            )
            metrics["latest_checkpoint"] = str(save_path)
        _append_metrics(metrics_path, metrics)

    if checkpoint_dir is not None and metrics.get("latest_checkpoint") is None:
        latest = latest_checkpoint_path(checkpoint_dir)
        metrics["latest_checkpoint"] = str(latest) if latest is not None else None
    return metrics


def benchmark_env_steps(
    config: TrainingConfig,
    *,
    graph=None,
    steps: int = 256,
) -> dict[str, Any]:
    import time

    graph = graph or _load_graph_from_config(config)
    spawn_source = config.spawn_source or ("density" if config.graph_name == "sf" else "uniform")
    initial_nodes = (
        None
        if spawn_source in {"js-visual", "density"}
        else _default_initial_car_nodes(graph.num_nodes, config.max_cars, config.seed)
    )
    env_params = make_spawned_env_params(
        graph,
        graph_name=config.graph_name,
        data_dir=config.data_dir,
        spawn_source=spawn_source,
        seed=config.seed,
        max_cars=config.max_cars,
        max_requests=config.max_requests,
        initial_car_nodes=initial_nodes,
        assignment_max_route_edges=config.assignment_max_route_edges,
        episode_seconds=config.episode_seconds,
        spawn_rate_per_minute=config.spawn_rate_per_minute,
    )
    reset_fn = jax.jit(lambda keys: jax.vmap(lambda key: reset(key, env_params))(keys))
    step_fn = jax.jit(lambda states, actions: jax.vmap(lambda s, a: step(s, a, env_params))(states, actions))

    @partial(jax.jit, static_argnames=("scan_steps",))
    def scan_steps(states, actions, *, scan_steps: int):
        def body(loop_states, _):
            next_states, next_timesteps = jax.vmap(lambda s, a: step(s, a, env_params))(loop_states, actions)
            return next_states, next_timesteps.reward

        return jax.lax.scan(body, states, xs=None, length=scan_steps)

    keys = jax.random.split(jax.random.PRNGKey(config.seed), config.num_envs)
    states, timesteps = reset_fn(keys)
    actions = jnp.zeros((config.num_envs,), dtype=jnp.int32)
    next_states, next_timesteps = step_fn(states, actions)
    next_timesteps.reward.block_until_ready()
    states, timesteps = next_states, next_timesteps
    states, rewards = scan_steps(states, actions, scan_steps=max(1, int(steps)))
    rewards.block_until_ready()

    start = time.perf_counter()
    states, rewards = scan_steps(states, actions, scan_steps=int(steps))
    rewards.block_until_ready()
    elapsed = max(1e-9, time.perf_counter() - start)
    total_steps = int(steps) * int(config.num_envs)
    return {
        "graph": config.graph_name,
        "num_envs": int(config.num_envs),
        "steps": int(steps),
        "total_env_steps": total_steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": total_steps / elapsed,
    }


def compute_gae(
    *,
    rewards,
    values,
    bootstrap_value,
    discounts,
    dones,
    gae_lambda: float,
):
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    values = jnp.asarray(values, dtype=jnp.float32)
    bootstrap_value = jnp.asarray(bootstrap_value, dtype=jnp.float32)
    discounts = jnp.asarray(discounts, dtype=jnp.float32)
    dones = jnp.asarray(dones, dtype=jnp.bool_)
    not_done = 1.0 - dones.astype(jnp.float32)

    def body(carry, inputs):
        next_value, next_advantage = carry
        reward, value, discount, alive = inputs
        transition_discount = discount * alive
        delta = reward + transition_discount * next_value - value
        advantage = delta + transition_discount * gae_lambda * next_advantage
        return (value, advantage), advantage

    _, reversed_advantages = jax.lax.scan(
        body,
        (bootstrap_value, jnp.zeros_like(bootstrap_value, dtype=jnp.float32)),
        (
            jnp.flip(rewards, axis=0),
            jnp.flip(values, axis=0),
            jnp.flip(discounts, axis=0),
            jnp.flip(not_done, axis=0),
        ),
    )
    advantages = jnp.flip(reversed_advantages, axis=0)
    returns = advantages + values
    return advantages, returns


def checkpoint_path(checkpoint_dir: str | Path, update: int) -> Path:
    return Path(checkpoint_dir) / f"update_{int(update):06d}"


def latest_checkpoint_path(checkpoint_dir: str | Path | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    root = Path(checkpoint_dir)
    if not root.exists():
        return None
    candidates = []
    for path in root.glob("update_*"):
        try:
            update = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((update, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _load_graph_from_config(config: TrainingConfig):
    if config.graph_name == "synthetic":
        return default_smoke_graph()
    if config.graph_name == "sf":
        return load_public_data_graph(
            config.data_dir,
            include_routing=True,
            cache_dir=config.routing_cache_dir,
            routing_chunk_size=config.routing_chunk_size,
        )
    raise ValueError(f"unknown graph_name: {config.graph_name}")


def _default_initial_car_nodes(num_nodes: int, max_cars: int, seed: int) -> list[int]:
    rng = jax.random.PRNGKey(seed + 991)
    nodes = jax.random.randint(rng, (max_cars,), 0, num_nodes, dtype=jnp.int32)
    return [int(node) for node in nodes]


def _collect_rollout(
    *,
    learner: TrainState,
    states,
    timesteps,
    env_params,
    rng,
    num_steps: int,
) -> dict[str, Any]:
    return _collect_rollout_scan(learner, states, timesteps, env_params, rng, num_steps)


@partial(jax.jit, static_argnames=("num_steps",))
def _collect_rollout_scan(
    learner: TrainState,
    states,
    timesteps,
    env_params,
    rng,
    num_steps: int,
) -> dict[str, Any]:
    def body(carry, _):
        loop_states, loop_timesteps, loop_rng = carry
        observation = loop_timesteps.observation
        logits, value = learner.apply_fn({"params": learner.params}, observation)
        loop_rng, action_rng, reset_rng = jax.random.split(loop_rng, 3)
        action = jax.random.categorical(action_rng, logits).astype(jnp.int32)
        all_log_probs = jax.nn.log_softmax(logits)
        selected_log_prob = jnp.take_along_axis(all_log_probs, action[:, None], axis=1).squeeze(1)

        next_states, next_timesteps = jax.vmap(lambda s, a: step(s, a, env_params))(loop_states, action)
        reset_keys = jax.random.split(reset_rng, next_timesteps.done.shape[0])
        reset_states, reset_timesteps = jax.vmap(lambda key: reset(key, env_params))(reset_keys)
        loop_states = jax.tree_util.tree_map(
            lambda reset_leaf, next_leaf: _select_done(next_timesteps.done, reset_leaf, next_leaf),
            reset_states,
            next_states,
        )
        loop_timesteps = jax.tree_util.tree_map(
            lambda reset_leaf, next_leaf: _select_done(next_timesteps.done, reset_leaf, next_leaf),
            reset_timesteps,
            next_timesteps,
        )
        transition = {
            "observations": observation,
            "actions": action,
            "log_probs": selected_log_prob,
            "rewards": next_timesteps.reward,
            "values": value,
            "discounts": next_timesteps.discount,
            "dones": next_timesteps.done,
        }
        return (loop_states, loop_timesteps, loop_rng), transition

    (states, timesteps, rng), rollout = jax.lax.scan(
        body,
        (states, timesteps, rng),
        xs=None,
        length=num_steps,
    )
    return {
        "observations": rollout["observations"],
        "actions": rollout["actions"],
        "log_probs": rollout["log_probs"],
        "rewards": rollout["rewards"],
        "values": rollout["values"],
        "discounts": rollout["discounts"],
        "dones": rollout["dones"],
        "states": states,
        "timesteps": timesteps,
        "rng": rng,
    }


@partial(jax.jit, static_argnames=("config",))
def _ppo_update(
    learner: TrainState,
    observation_batch,
    batch: dict[str, Any],
    config: TrainingConfig,
    rng,
):
    advantages = batch["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    batch = {**batch, "advantages": advantages}

    batch_size = batch["actions"].shape[0]
    num_minibatches = max(1, min(int(config.num_minibatches), int(batch_size)))
    minibatch_size = (int(batch_size) + num_minibatches - 1) // num_minibatches
    padded_size = num_minibatches * minibatch_size
    pad_amount = padded_size - int(batch_size)
    sample_mask = jnp.concatenate(
        [
            jnp.ones((batch_size,), dtype=jnp.float32),
            jnp.zeros((pad_amount,), dtype=jnp.float32),
        ],
        axis=0,
    )
    padded_obs = _pad_batch_tree(observation_batch, pad_amount)
    padded_batch = _pad_batch_tree(batch, pad_amount)

    def update_epoch(carry, epoch_rng):
        epoch_learner = carry
        permutation = jax.random.permutation(epoch_rng, padded_size)
        shuffled_obs = jax.tree_util.tree_map(
            lambda leaf: leaf[permutation].reshape(
                (num_minibatches, minibatch_size) + leaf.shape[1:]
            ),
            padded_obs,
        )
        shuffled_batch = jax.tree_util.tree_map(
            lambda leaf: leaf[permutation].reshape(
                (num_minibatches, minibatch_size) + leaf.shape[1:]
            ),
            padded_batch,
        )
        shuffled_mask = sample_mask[permutation].reshape((num_minibatches, minibatch_size))

        def update_minibatch(inner_learner, minibatch):
            minibatch_obs, minibatch_batch, minibatch_mask = minibatch
            next_learner, metrics = _ppo_minibatch_update(
                inner_learner,
                minibatch_obs,
                minibatch_batch,
                minibatch_mask,
                config,
            )
            return next_learner, metrics

        return jax.lax.scan(
            update_minibatch,
            epoch_learner,
            (shuffled_obs, shuffled_batch, shuffled_mask),
        )

    epoch_rngs = jax.random.split(rng, max(1, int(config.update_epochs)))
    learner, metrics = jax.lax.scan(update_epoch, learner, epoch_rngs)
    metrics = jax.tree_util.tree_map(lambda leaf: leaf.mean(), metrics)
    return learner, metrics


def _ppo_minibatch_update(
    learner: TrainState,
    observation_batch,
    batch: dict[str, Any],
    sample_mask,
    config: TrainingConfig,
):
    normalizer = jnp.maximum(sample_mask.sum(), 1.0)

    def weighted_mean(values):
        return jnp.sum(values * sample_mask) / normalizer

    def loss_fn(params):
        logits, new_values = learner.apply_fn({"params": params}, observation_batch)
        log_probs = jax.nn.log_softmax(logits)
        new_log_probs = jnp.take_along_axis(log_probs, batch["actions"][:, None], axis=1).squeeze(1)
        ratio = jnp.exp(new_log_probs - batch["old_log_probs"])
        clipped_ratio = jnp.clip(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
        policy_loss = -weighted_mean(
            jnp.minimum(ratio * batch["advantages"], clipped_ratio * batch["advantages"])
        )
        value_pred_clipped = batch["values"] + jnp.clip(
            new_values - batch["values"],
            -config.clip_coef,
            config.clip_coef,
        )
        value_loss = (new_values - batch["returns"]) ** 2
        value_loss_clipped = (value_pred_clipped - batch["returns"]) ** 2
        value_loss = 0.5 * weighted_mean(jnp.maximum(value_loss, value_loss_clipped))
        probs = jax.nn.softmax(logits)
        entropy = weighted_mean(-jnp.sum(probs * log_probs, axis=-1))
        old_approx_kl = weighted_mean(-jnp.log(ratio))
        approx_kl = weighted_mean((ratio - 1.0) - jnp.log(ratio))
        clipfrac = weighted_mean((jnp.abs(ratio - 1.0) > config.clip_coef).astype(jnp.float32))
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        return loss, {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "old_approx_kl": old_approx_kl,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
        }

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(learner.params)
    learner = learner.apply_gradients(grads=grads)
    return learner, metrics


def _pad_batch_tree(tree, pad_amount: int):
    if int(pad_amount) <= 0:
        return tree
    return jax.tree_util.tree_map(
        lambda leaf: jnp.pad(
            leaf,
            [(0, int(pad_amount))] + [(0, 0)] * (leaf.ndim - 1),
        ),
        tree,
    )


def _flatten_time_env_tree(tree):
    return jax.tree_util.tree_map(lambda leaf: leaf.reshape((leaf.shape[0] * leaf.shape[1],) + leaf.shape[2:]), tree)


def _select_done(done, reset_leaf, next_leaf):
    mask = done
    while mask.ndim < next_leaf.ndim:
        mask = mask[..., None]
    return jnp.where(mask, reset_leaf, next_leaf)


def _append_metrics(metrics_path: str | Path | None, metrics: dict[str, Any]) -> None:
    if metrics_path is None:
        return
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")


def _resolve_optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()
