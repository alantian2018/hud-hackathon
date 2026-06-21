from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flax import linen as nn
from flax import serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from .env import EnvParams, FleetEnv
from .types import Observation


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 8
    num_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 2.5e-4
    max_grad_norm: float = 0.5
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    gae_lambda: float = 0.95
    total_updates: int = 1000


class ActorCritic(nn.Module):
    max_degree: int

    @nn.compact
    def __call__(self, obs: Observation, action_mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = obs.raster.astype(jnp.float32)
        x = nn.Conv(16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(64)(x)
        x = nn.relu(x)

        g = nn.Dense(64)(obs.global_features.astype(jnp.float32))
        g = nn.relu(g)
        context = jnp.concatenate([x, g], axis=-1)
        context = nn.Dense(64)(context)
        context = nn.relu(context)

        a = nn.Dense(64)(obs.action_features.astype(jnp.float32))
        a = nn.relu(a)
        a = nn.Dense(64)(a)
        a = nn.relu(a)
        fused = nn.relu(a + context[:, None, :])
        logits = nn.Dense(1)(fused).squeeze(-1)
        logits = jnp.where(action_mask, logits, -1.0e9)

        value = nn.Dense(64)(context)
        value = nn.relu(value)
        value = nn.Dense(1)(value).squeeze(-1)
        return logits, value


class TrainState(train_state.TrainState):
    pass


def linear_schedule(config: PPOConfig):
    return optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=0.0,
        transition_steps=max(1, config.total_updates),
    )


def make_train_state(
    rng: jnp.ndarray,
    sample_observation: Observation,
    max_degree: int,
    config: PPOConfig,
) -> TrainState:
    model = ActorCritic(max_degree=max_degree)
    batched_obs = Observation(
        raster=sample_observation.raster[None, ...],
        global_features=sample_observation.global_features[None, ...],
        action_features=sample_observation.action_features[None, ...],
    )
    action_mask = jnp.ones((1, max_degree), dtype=bool)
    variables = model.init(rng, batched_obs, action_mask)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=linear_schedule(config)),
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)


def masked_logprob(logits: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)


def categorical_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(probs * log_probs, axis=-1)


def _flatten_obs(obs: Observation) -> Observation:
    leading = obs.raster.shape[:2]
    batch = leading[0] * leading[1]
    return Observation(
        raster=obs.raster.reshape((batch,) + obs.raster.shape[2:]),
        global_features=obs.global_features.reshape((batch,) + obs.global_features.shape[2:]),
        action_features=obs.action_features.reshape((batch,) + obs.action_features.shape[2:]),
    )


def _index_obs(obs: Observation, indices: jnp.ndarray) -> Observation:
    return Observation(
        raster=obs.raster[indices],
        global_features=obs.global_features[indices],
        action_features=obs.action_features[indices],
    )


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    discounts: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    def step(carry, transition):
        next_adv, next_value = carry
        reward, value, discount, done = transition
        effective_discount = discount * (1.0 - done.astype(jnp.float32))
        delta = reward + effective_discount * next_value - value
        adv = delta + effective_discount * gae_lambda * next_adv
        return (adv, value), adv

    (_last_adv, _last_val), advantages_rev = jax.lax.scan(
        step,
        (jnp.zeros_like(last_value), last_value),
        (rewards[::-1], values[::-1], discounts[::-1], dones[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def collect_rollout(
    train_state: TrainState,
    env: FleetEnv,
    env_params: EnvParams,
    config: PPOConfig,
    rng: jnp.ndarray,
):
    states, timesteps, rng = initialize_env_batch(env, env_params, config, rng)
    transitions, last_value, states, timesteps, rng = collect_rollout_from(
        train_state,
        env,
        env_params,
        config,
        states,
        timesteps,
        rng,
    )
    del states, timesteps
    return transitions, last_value, rng


def initialize_env_batch(
    env: FleetEnv,
    env_params: EnvParams,
    config: PPOConfig,
    rng: jnp.ndarray,
):
    reset_keys = jax.random.split(rng, config.num_envs + 1)
    states, timesteps = jax.vmap(lambda k: env.reset(k, env_params))(reset_keys[1:])
    return states, timesteps, reset_keys[0]


def collect_rollout_from(
    train_state: TrainState,
    env: FleetEnv,
    env_params: EnvParams,
    config: PPOConfig,
    states,
    timesteps,
    rng: jnp.ndarray,
):

    def rollout_step(carry, _):
        states, timesteps, key = carry
        logits, values = train_state.apply_fn({"params": train_state.params}, timesteps.observation, timesteps.action_mask)
        key, action_key = jax.random.split(key)
        actions = jax.random.categorical(action_key, logits).astype(jnp.int32)
        logprobs = masked_logprob(logits, actions)
        next_states, next_timesteps = jax.vmap(lambda s, a: env.step(s, a, env_params))(states, actions)
        transition = {
            "obs": timesteps.observation,
            "action_mask": timesteps.action_mask,
            "actions": actions,
            "logprobs": logprobs,
            "values": values,
            "rewards": next_timesteps.reward,
            "discounts": next_timesteps.discount,
            "dones": next_timesteps.done,
        }
        return (next_states, next_timesteps, key), transition

    (states, timesteps, rng), transitions = jax.lax.scan(
        rollout_step,
        (states, timesteps, rng),
        None,
        length=config.num_steps,
    )
    _logits, last_value = train_state.apply_fn({"params": train_state.params}, timesteps.observation, timesteps.action_mask)
    return transitions, last_value, states, timesteps, rng


def mean_recent_pickup_wait(metrics, window: int = 10) -> jnp.ndarray:
    samples = metrics.pickup_wait_samples
    times = metrics.pickup_wait_sample_times
    counts = metrics.pickup_wait_count.astype(jnp.int32)
    capacity = samples.shape[-1]
    sample_indices = jnp.arange(capacity, dtype=jnp.int32)
    valid = sample_indices < jnp.minimum(counts[..., None], capacity)
    flat_samples = jnp.reshape(samples, (-1,))
    flat_times = jnp.reshape(jnp.where(valid, times, -jnp.inf), (-1,))
    take_count = min(window, flat_samples.shape[0])
    top_indices = jnp.argsort(flat_times)[-take_count:]
    top_valid = flat_times[top_indices] > -jnp.inf
    total = jnp.sum(jnp.where(top_valid, flat_samples[top_indices], 0.0))
    denom = jnp.sum(top_valid).astype(jnp.float32)
    return jnp.where(denom > 0.0, total / denom, 0.0)


def mean_recent_request_wait_or_age(states, window: int = 10) -> jnp.ndarray:
    spawn_times = states.request_spawn_time
    pickup_times = states.request_pickup_time
    sim_time = states.sim_time_seconds
    while sim_time.ndim < spawn_times.ndim:
        sim_time = sim_time[..., None]

    final_or_current_wait = jnp.where(
        pickup_times >= 0.0,
        pickup_times - spawn_times,
        sim_time - spawn_times,
    )
    final_or_current_wait = jnp.maximum(0.0, final_or_current_wait)
    valid = states.request_ids >= 0
    flat_waits = jnp.reshape(final_or_current_wait, (-1,))
    flat_spawns = jnp.reshape(jnp.where(valid, spawn_times, -jnp.inf), (-1,))
    take_count = min(window, flat_waits.shape[0])
    top_indices = jnp.argsort(flat_spawns)[-take_count:]
    top_valid = flat_spawns[top_indices] > -jnp.inf
    total = jnp.sum(jnp.where(top_valid, flat_waits[top_indices], 0.0))
    denom = jnp.sum(top_valid).astype(jnp.float32)
    return jnp.where(denom > 0.0, total / denom, 0.0)


def ppo_loss(
    params,
    apply_fn,
    obs: Observation,
    action_mask: jnp.ndarray,
    actions: jnp.ndarray,
    old_logprobs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    config: PPOConfig,
):
    logits, values = apply_fn({"params": params}, obs, action_mask)
    new_logprobs = masked_logprob(logits, actions)
    entropy = categorical_entropy(logits).mean()
    log_ratio = new_logprobs - old_logprobs
    ratio = jnp.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
    policy_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
    value_loss = 0.5 * ((returns - values) ** 2).mean()
    loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    return loss, {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
    }


def ppo_update(
    train_state: TrainState,
    transitions,
    last_value: jnp.ndarray,
    config: PPOConfig,
    rng: jnp.ndarray,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    advantages, returns = compute_gae(
        transitions["rewards"],
        transitions["values"],
        transitions["discounts"],
        transitions["dones"],
        last_value,
        config.gae_lambda,
    )
    flat_obs = _flatten_obs(transitions["obs"])
    batch = config.num_steps * config.num_envs
    flat_masks = transitions["action_mask"].reshape((batch, transitions["action_mask"].shape[-1]))
    flat_actions = transitions["actions"].reshape((batch,))
    flat_logprobs = transitions["logprobs"].reshape((batch,))
    flat_advantages = advantages.reshape((batch,))
    flat_returns = returns.reshape((batch,))
    flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

    minibatch_size = min(config.minibatch_size, batch)
    num_minibatches = max(1, batch // minibatch_size)

    def epoch_step(carry, _):
        state, key, metrics = carry
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, batch)
        permutation = permutation[: num_minibatches * minibatch_size].reshape((num_minibatches, minibatch_size))

        def minibatch_step(inner_carry, indices):
            inner_state, last_metrics = inner_carry
            obs_mb = _index_obs(flat_obs, indices)
            masks_mb = flat_masks[indices]
            actions_mb = flat_actions[indices]
            logprobs_mb = flat_logprobs[indices]
            advantages_mb = flat_advantages[indices]
            returns_mb = flat_returns[indices]

            (loss, new_metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                inner_state.params,
                inner_state.apply_fn,
                obs_mb,
                masks_mb,
                actions_mb,
                logprobs_mb,
                advantages_mb,
                returns_mb,
                config,
            )
            inner_state = inner_state.apply_gradients(grads=grads)
            return (inner_state, new_metrics), None

        (state, metrics), _ = jax.lax.scan(minibatch_step, (state, metrics), permutation)
        return (state, key, metrics), None

    initial_metrics = {
        "loss": jnp.array(0.0),
        "policy_loss": jnp.array(0.0),
        "value_loss": jnp.array(0.0),
        "entropy": jnp.array(0.0),
        "approx_kl": jnp.array(0.0),
    }
    (train_state, rng, metrics), _ = jax.lax.scan(
        epoch_step,
        (train_state, rng, initial_metrics),
        None,
        length=config.update_epochs,
    )
    metrics = {
        **metrics,
        "rollout_reward": transitions["rewards"].mean(),
        "advantage_mean": advantages.mean(),
    }
    return train_state, metrics


def ppo_smoke_update(
    train_state: TrainState,
    env: FleetEnv,
    env_params: EnvParams,
    config: PPOConfig,
    rng: jnp.ndarray,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    def update_once(state, key):
        transitions, last_value, next_key = collect_rollout(state, env, env_params, config, key)
        state, metrics = ppo_update(state, transitions, last_value, config, next_key)
        return state, metrics

    return jax.jit(update_once)(train_state, rng)


def save_checkpoint(path: str | Path, state: TrainState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(state))


def load_checkpoint(path: str | Path, state: TrainState) -> TrainState:
    return serialization.from_bytes(state, Path(path).read_bytes())
