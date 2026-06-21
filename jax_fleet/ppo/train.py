from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import partial
import json
from pathlib import Path
import time
from typing import Any

from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from jax_fleet.env import reset, step
from jax_fleet.graph import build_synthetic_graph, load_public_data_graph
from jax_fleet.ppo.model import ActorCritic
from jax_fleet.rich_renderer import RichRenderer
from jax_fleet.scene_export import export_scene
from jax_fleet.spawns import make_spawned_env_params


class TrainState(train_state.TrainState):
    pass


class ReturnNormalizer:
    """Running mean/std of returns.

    The value head predicts unit-scale (normalized) values; the rest of the loop
    (GAE, bootstrap, metrics) works in raw return units. Denormalize the head's
    output with ``value * std + mean`` and normalize value targets with
    ``(target - mean) / std``. Disabled normalizers behave as ``mean=0, std=1``.
    """

    def __init__(self, *, enabled: bool, decay: float):
        self.enabled = bool(enabled)
        self.decay = float(decay)
        self.mean = 0.0
        self.std = 1.0
        self._initialized = False

    def as_arrays(self):
        if not self.enabled:
            return jnp.asarray(0.0, jnp.float32), jnp.asarray(1.0, jnp.float32)
        return jnp.asarray(self.mean, jnp.float32), jnp.asarray(self.std, jnp.float32)

    def update(self, returns) -> None:
        if not self.enabled:
            return
        batch_mean = float(jnp.mean(returns))
        batch_std = max(float(jnp.std(returns)), 1.0e-6)
        if not self._initialized:
            self.mean = batch_mean
            self.std = batch_std
            self._initialized = True
        else:
            decay = self.decay
            self.mean = decay * self.mean + (1.0 - decay) * batch_mean
            self.std = max(decay * self.std + (1.0 - decay) * batch_std, 1.0e-6)


@dataclass(frozen=True)
class TrainingConfig:
    graph_name: str = "sf"
    data_dir: Path | str = Path("dist/data")
    routing_cache_dir: Path | str = Path("cache/jax_fleet")
    routing_chunk_size: int = 512
    seed: int = 0
    num_envs: int = 8
    num_steps: int = 128
    num_updates: int = 500
    max_cars: int = 40
    max_requests: int = 32
    assignment_max_route_edges: int = 10000
    # The fleet task is continuing: idle cars reposition forever. With an
    # infinite horizon the env never reaches `done`, so there is no autoreset
    # mid-rollout and the per-transition discount is never zeroed -- each
    # num_steps window is a truncation of one long trajectory, bootstrapped by
    # the value head. Set a finite value to recover episodic resets.
    episode_seconds: float = float("inf")
    spawn_rate_per_minute: float = 0.0
    spawn_source: str | None = None
    reward_mode: str = "dense_wait"
    observation_mode: str = "learning_v1"
    drop_penalty: float = 10.0
    pickup_bonus: float = 0.0
    time_discount_reference_seconds: float = 60.0
    learning_rate: float = 3e-4
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    normalize_returns: bool = True
    return_norm_decay: float = 0.99
    clip_vloss: bool = False
    checkpoint_dir: Path | str | None = Path("runs/jax_fleet/checkpoints")
    checkpoint_every: int = 1
    metrics_path: Path | str | None = Path("runs/jax_fleet/metrics.jsonl")
    resume: bool = False
    track: bool = False
    wandb_project_name: str = "jax_fleet"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str | None = None
    wandb_video_every: int = 0
    wandb_video_max_steps: int = 50_000
    wandb_video_max_pickups: int = 20
    wandb_video_max_frames: int = 240
    wandb_video_width: int = 960
    wandb_video_height: int = 600
    wandb_video_fps: int = 12
    require_gpu: bool = False

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
        graph_name="synthetic",
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
    if config.require_gpu:
        from jax_fleet.devices import require_gpu_available

        require_gpu_available()

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
        reward_mode=config.reward_mode,
        observation_mode=config.observation_mode,
        drop_penalty=config.drop_penalty,
        pickup_bonus=config.pickup_bonus,
        time_discount_reference_seconds=config.time_discount_reference_seconds,
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
    # Running statistics that map the value head's unit-scale output to the raw
    # return scale. Dense-wait returns are large (order -1000s); predicting them
    # directly makes the value loss enormous and starves the policy under the
    # shared gradient-norm clip. Normalizing keeps value targets ~unit-scale.
    return_stats = ReturnNormalizer(enabled=config.normalize_returns, decay=config.return_norm_decay)
    wandb_run = _init_wandb_run(config)
    started_at = time.perf_counter()
    try:
        for update in range(start_update + 1, config.num_updates + 1):
            value_mean, value_std = return_stats.as_arrays()
            rollout = _collect_rollout(
                learner=learner,
                states=states,
                timesteps=timesteps,
                env_params=env_params,
                rng=rng,
                num_steps=config.num_steps,
                value_mean=value_mean,
                value_std=value_std,
            )
            states = rollout["states"]
            timesteps = rollout["timesteps"]
            rng = rollout["rng"]
            bootstrap_value = (
                learner.apply_fn({"params": learner.params}, timesteps.observation)[1] * value_std
                + value_mean
            )
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
            learner, loss_metrics = _ppo_update(
                learner, batch_obs, batch, config, update_rng, value_mean, value_std
            )
            # Refresh the normalizer for the next update using this rollout's returns.
            return_stats.update(returns)
            global_step = int(update * config.num_envs * config.num_steps)
            elapsed = max(1.0e-9, time.perf_counter() - started_at)
            metrics = _training_metrics(
                update=update,
                global_step=global_step,
                elapsed_seconds=elapsed,
                rollout=rollout,
                returns=returns,
                loss_metrics=loss_metrics,
                learner=learner,
                config=config,
                start_update=start_update,
                latest_checkpoint=None,
            )
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
            _log_wandb_metrics(wandb_run, metrics, step=global_step)
            _maybe_log_wandb_video(
                wandb_run,
                learner=learner,
                env_params=env_params,
                config=config,
                update=update,
                global_step=global_step,
            )
    finally:
        _finish_wandb_run(wandb_run)

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

    if config.require_gpu:
        from jax_fleet.devices import require_gpu_available

        require_gpu_available()

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
        reward_mode=config.reward_mode,
        observation_mode=config.observation_mode,
        drop_penalty=config.drop_penalty,
        pickup_bonus=config.pickup_bonus,
        time_discount_reference_seconds=config.time_discount_reference_seconds,
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


def _training_metrics(
    *,
    update: int,
    global_step: int,
    elapsed_seconds: float,
    rollout: dict[str, Any],
    returns,
    loss_metrics,
    learner: TrainState,
    config: TrainingConfig,
    start_update: int,
    latest_checkpoint: str | None,
) -> dict[str, Any]:
    rewards = jnp.asarray(rollout["rewards"], dtype=jnp.float32)
    returns = jnp.asarray(returns, dtype=jnp.float32)
    values = jnp.asarray(rollout["values"], dtype=jnp.float32)
    discounts = jnp.asarray(rollout["discounts"], dtype=jnp.float32)
    dones = jnp.asarray(rollout["dones"], dtype=jnp.float32)
    dt_seconds = jnp.asarray(rollout["dt_seconds"], dtype=jnp.float32)
    env_metrics = rollout["env_metrics"]
    last_env_metrics = jax.tree_util.tree_map(lambda leaf: leaf[-1], env_metrics)
    rollout_last_queued_wait_seconds = jnp.asarray(
        env_metrics.last_queued_wait_seconds,
        dtype=jnp.float32,
    )
    rollout_last_dense_wait_penalty = jnp.asarray(
        env_metrics.last_dense_wait_penalty,
        dtype=jnp.float32,
    )
    rollout_queued_requests = jnp.asarray(env_metrics.queued_requests, dtype=jnp.float32)
    completed_steps = max(1, (int(update) - int(start_update)) * int(config.num_envs) * int(config.num_steps))
    sps = int(completed_steps / max(elapsed_seconds, 1.0e-9))
    explained_variance = _explained_variance(returns.reshape((-1,)), values.reshape((-1,)))

    metrics = {
        "update": int(update),
        "updates": int(update),
        "global_step": int(global_step),
        "latest_checkpoint": latest_checkpoint,
        "last_loss": float(loss_metrics["loss"]),
        "last_policy_loss": float(loss_metrics["policy_loss"]),
        "last_value_loss": float(loss_metrics["value_loss"]),
        "last_entropy": float(loss_metrics["entropy"]),
        "last_approx_kl": float(loss_metrics["approx_kl"]),
        "last_clipfrac": float(loss_metrics["clipfrac"]),
        "last_mean_reward": float(rewards.mean()),
        "charts/global_step": int(global_step),
        "charts/update": int(update),
        "charts/learning_rate": float(config.learning_rate),
        "charts/SPS": sps,
        "charts/train_step": int(jnp.asarray(learner.step)),
        "losses/loss": float(loss_metrics["loss"]),
        "losses/policy_loss": float(loss_metrics["policy_loss"]),
        "losses/value_loss": float(loss_metrics["value_loss"]),
        "losses/entropy": float(loss_metrics["entropy"]),
        "losses/old_approx_kl": float(loss_metrics["old_approx_kl"]),
        "losses/approx_kl": float(loss_metrics["approx_kl"]),
        "losses/clipfrac": float(loss_metrics["clipfrac"]),
        "losses/explained_variance": float(explained_variance),
        "rollout/mean_reward": float(rewards.mean()),
        "rollout/mean_return": float(returns.mean()),
        "rollout/mean_discount": float(discounts.mean()),
        "rollout/mean_dt_seconds": float(dt_seconds.mean()),
        "rollout/done_fraction": float(dones.mean()),
        "rollout/mean_last_queued_wait_seconds": float(rollout_last_queued_wait_seconds.mean()),
        "rollout/max_last_queued_wait_seconds": float(rollout_last_queued_wait_seconds.max()),
        "rollout/mean_last_dense_wait_penalty": float(rollout_last_dense_wait_penalty.mean()),
        "rollout/mean_queued_requests": float(rollout_queued_requests.mean()),
        "env/completed_requests": float(jnp.asarray(last_env_metrics.completed_requests).mean()),
        "env/dropped_requests": float(jnp.asarray(last_env_metrics.dropped_requests).mean()),
        "env/picked_up_requests": float(jnp.asarray(last_env_metrics.picked_up_requests).mean()),
        "env/queued_requests": float(jnp.asarray(last_env_metrics.queued_requests).mean()),
        "env/invalid_actions": float(jnp.asarray(last_env_metrics.invalid_actions).mean()),
        "env/pickup_wait_seconds": float(jnp.asarray(last_env_metrics.pickup_wait_seconds).mean()),
        "env/aggregate_reward": float(jnp.asarray(last_env_metrics.aggregate_reward).mean()),
        "env/dense_wait_penalty": float(jnp.asarray(last_env_metrics.dense_wait_penalty).mean()),
        "env/drop_penalty_reward": float(jnp.asarray(last_env_metrics.drop_penalty_reward).mean()),
        "env/pickup_bonus_reward": float(jnp.asarray(last_env_metrics.pickup_bonus_reward).mean()),
        "env/queued_wait_seconds": float(jnp.asarray(last_env_metrics.queued_wait_seconds).mean()),
        "env/last_queued_wait_seconds": float(jnp.asarray(last_env_metrics.last_queued_wait_seconds).mean()),
        "env/reward_mode": float(jnp.asarray(last_env_metrics.reward_mode).mean()),
    }
    return metrics


def _explained_variance(y_true, y_pred):
    y_true = jnp.asarray(y_true, dtype=jnp.float32)
    y_pred = jnp.asarray(y_pred, dtype=jnp.float32)
    var_y = jnp.var(y_true)
    return jnp.where(var_y > 1.0e-8, 1.0 - jnp.var(y_true - y_pred) / var_y, jnp.nan)


def _init_wandb_run(config: TrainingConfig):
    if not config.track:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking requested with --track, but wandb is not installed. "
            "Install it with `pip install wandb` or `pip install .[tracking]`."
        ) from exc

    kwargs: dict[str, Any] = {
        "project": config.wandb_project_name,
        "config": config.to_jsonable(),
    }
    if config.wandb_entity:
        kwargs["entity"] = config.wandb_entity
    if config.wandb_run_name:
        kwargs["name"] = config.wandb_run_name
    if config.wandb_mode:
        kwargs["mode"] = config.wandb_mode
    return wandb.init(**kwargs)


def _log_wandb_metrics(wandb_run, metrics: dict[str, Any], *, step: int) -> None:
    if wandb_run is None:
        return
    wandb_run.log(metrics, step=step)


def _maybe_log_wandb_video(
    wandb_run,
    *,
    learner: TrainState,
    env_params,
    config: TrainingConfig,
    update: int,
    global_step: int,
) -> None:
    if wandb_run is None or int(config.wandb_video_every) <= 0:
        return
    if int(update) % int(config.wandb_video_every) != 0:
        return
    try:
        payload = _render_policy_rollout_video(
            learner=learner,
            env_params=env_params,
            config=config,
            seed=int(config.seed) + int(update) * 1009,
        )
    except Exception as exc:
        wandb_run.log(
            {
                "diagnostics/policy_video_failed": 1.0,
                "diagnostics/policy_video_error": str(exc),
            },
            step=global_step,
        )
        return
    if payload is None:
        return
    video, video_metrics = payload
    try:
        import wandb
    except ImportError:
        return
    wandb_run.log(
        {
            "diagnostics/policy_video": wandb.Video(
                video,
                fps=max(1, int(config.wandb_video_fps)),
                format="mp4",
            ),
            **video_metrics,
        },
        step=global_step,
    )


def _render_policy_rollout_video(
    *,
    learner: TrainState,
    env_params,
    config: TrainingConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]] | None:
    max_steps = max(1, int(config.wandb_video_max_steps))
    max_pickups = max(0, int(config.wandb_video_max_pickups))
    max_frames = max(1, int(config.wandb_video_max_frames))
    capture_every = max(1, int(np.ceil(max_steps / max_frames)))
    renderer = RichRenderer(
        width=max(1, int(config.wandb_video_width)),
        height=max(1, int(config.wandb_video_height)),
    )
    cpu_device = jax.devices("cpu")[0]
    env_params = jax.device_put(env_params, cpu_device)
    model_params = jax.device_put(learner.params, cpu_device)

    @partial(jax.jit, device=cpu_device)
    def policy_step(loop_state, loop_timestep, model_params):
        logits, _ = learner.apply_fn({"params": model_params}, loop_timestep.observation)
        action = jnp.argmax(logits).astype(jnp.int32)
        return step(loop_state, action, env_params)

    state, timestep = reset(jax.device_put(jax.random.PRNGKey(seed), cpu_device), env_params)
    frames: list[np.ndarray] = []

    def capture(include_static: bool) -> None:
        scene = export_scene(
            state,
            timestep,
            env_params,
            include_static=include_static,
            include_route_previews=True,
        )
        frames.append(renderer.render(scene))

    capture(include_static=True)
    steps = 0
    while steps < max_steps and not bool(np.asarray(timestep.done)):
        if max_pickups > 0 and int(np.asarray(state.metrics.picked_up_requests)) >= max_pickups:
            break
        state, timestep = policy_step(state, timestep, model_params)
        steps += 1
        should_capture = (steps % capture_every == 0) and (len(frames) < max_frames)
        if should_capture:
            capture(include_static=renderer.needs_static_scene(float(np.asarray(state.time_seconds))))

    if len(frames) == 0:
        return None
    if len(frames) == 1:
        capture(include_static=renderer.needs_static_scene(float(np.asarray(state.time_seconds))))
    video = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
    video_metrics = {
        "diagnostics/video_steps": float(steps),
        "diagnostics/video_pickups": float(np.asarray(state.metrics.picked_up_requests)),
        "diagnostics/video_completed_requests": float(np.asarray(state.metrics.completed_requests)),
        "diagnostics/video_dropped_requests": float(np.asarray(state.metrics.dropped_requests)),
        "diagnostics/video_reward": float(np.asarray(state.metrics.aggregate_reward)),
    }
    return video.astype(np.uint8, copy=False), video_metrics


def _finish_wandb_run(wandb_run) -> None:
    if wandb_run is not None:
        wandb_run.finish()


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
    value_mean=0.0,
    value_std=1.0,
) -> dict[str, Any]:
    return _collect_rollout_scan(
        learner,
        states,
        timesteps,
        env_params,
        rng,
        jnp.asarray(value_mean, jnp.float32),
        jnp.asarray(value_std, jnp.float32),
        num_steps,
    )


@partial(jax.jit, static_argnames=("num_steps",))
def _collect_rollout_scan(
    learner: TrainState,
    states,
    timesteps,
    env_params,
    rng,
    value_mean,
    value_std,
    num_steps: int,
) -> dict[str, Any]:
    def body(carry, _):
        loop_states, loop_timesteps, loop_rng = carry
        observation = loop_timesteps.observation
        logits, value_norm = learner.apply_fn({"params": learner.params}, observation)
        # The head outputs a unit-scale value; store it in raw return units so GAE
        # and the bootstrap stay consistent with the env reward scale.
        value = value_norm * value_std + value_mean
        loop_rng, action_rng, reset_rng = jax.random.split(loop_rng, 3)
        action = jax.random.categorical(action_rng, logits).astype(jnp.int32)
        all_log_probs = jax.nn.log_softmax(logits)
        selected_log_prob = jnp.take_along_axis(all_log_probs, action[:, None], axis=1).squeeze(1)

        next_states, next_timesteps = jax.vmap(lambda s, a: step(s, a, env_params))(loop_states, action)
        # Episodes rarely end inside a rollout, so only pay for the (expensive)
        # full vmapped reset when at least one env is actually done.
        any_done = jnp.any(next_timesteps.done)

        def _do_reset(_):
            reset_keys = jax.random.split(reset_rng, next_timesteps.done.shape[0])
            return jax.vmap(lambda key: reset(key, env_params))(reset_keys)

        reset_states, reset_timesteps = jax.lax.cond(
            any_done,
            _do_reset,
            lambda _: (next_states, next_timesteps),
            operand=None,
        )
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
            "dt_seconds": next_timesteps.dt_seconds,
            "env_metrics": next_timesteps.metrics,
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
        "dt_seconds": rollout["dt_seconds"],
        "env_metrics": rollout["env_metrics"],
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
    value_mean=0.0,
    value_std=1.0,
):
    value_mean = jnp.asarray(value_mean, jnp.float32)
    value_std = jnp.asarray(value_std, jnp.float32)
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
                value_mean,
                value_std,
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
    value_mean=0.0,
    value_std=1.0,
):
    value_mean = jnp.asarray(value_mean, jnp.float32)
    value_std = jnp.asarray(value_std, jnp.float32)
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
        # The value head predicts normalized targets; compare in that space so the
        # squared error (and clip range) are unit-scale regardless of return size.
        returns_norm = (batch["returns"] - value_mean) / value_std
        old_values_norm = (batch["values"] - value_mean) / value_std
        value_error = (new_values - returns_norm) ** 2
        if config.clip_vloss:
            value_pred_clipped = old_values_norm + jnp.clip(
                new_values - old_values_norm,
                -config.clip_coef,
                config.clip_coef,
            )
            value_error_clipped = (value_pred_clipped - returns_norm) ** 2
            value_error = jnp.maximum(value_error, value_error_clipped)
        value_loss = 0.5 * weighted_mean(value_error)
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
