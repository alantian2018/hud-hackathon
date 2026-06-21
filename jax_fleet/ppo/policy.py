from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from jax_fleet.ppo.model import ActorCritic


class PolicyTrainState(train_state.TrainState):
    pass


@dataclass(frozen=True)
class CheckpointPolicy:
    path: Path
    update: int | None
    params: Any
    apply_fn: Any
    action_fn: Any

    def action(self, observation) -> int:
        return int(np.asarray(self.action_fn(self.params, observation)))


def load_checkpoint_policy(
    checkpoint_path: str | Path,
    observation,
    *,
    max_degree: int,
    learning_rate: float = 3e-4,
    max_grad_norm: float = 0.5,
    use_jit: bool = True,
) -> CheckpointPolicy:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    model = ActorCritic(max_degree=max_degree)
    variables = model.init(jax.random.PRNGKey(0), observation)
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))
    learner = PolicyTrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    restore_target = {
        "params": learner.params,
        "opt_state": learner.opt_state,
        "train_step": learner.step,
        "rng": jax.random.PRNGKey(0),
        "update": 0,
    }
    restored = ocp.PyTreeCheckpointer().restore(checkpoint_path, item=restore_target)
    params = restored["params"]
    update = _maybe_int(restored.get("update"))

    def choose_action(policy_params, policy_observation):
        logits, _ = model.apply({"params": policy_params}, policy_observation)
        logits = jnp.asarray(logits)
        if logits.ndim == 2:
            logits = logits[0]
        return jnp.argmax(logits).astype(jnp.int32)

    action_fn = jax.jit(choose_action) if use_jit else choose_action
    return CheckpointPolicy(
        path=checkpoint_path,
        update=update,
        params=params,
        apply_fn=model.apply,
        action_fn=action_fn,
    )


def resolve_policy_checkpoint_path(
    checkpoint: str | Path | None,
    *,
    checkpoint_dir: str | Path,
) -> Path:
    if checkpoint is None or str(checkpoint).lower() == "latest":
        latest = latest_checkpoint_path(checkpoint_dir)
        if latest is None:
            raise FileNotFoundError(f"no update_* checkpoints found under {checkpoint_dir}")
        return latest

    path = Path(checkpoint).expanduser()
    if path.exists() and path.is_dir() and not _looks_like_checkpoint(path):
        latest = latest_checkpoint_path(path)
        if latest is None:
            raise FileNotFoundError(f"no update_* checkpoints found under {path}")
        return latest
    return path.resolve()


def latest_checkpoint_path(checkpoint_dir: str | Path | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    root = Path(checkpoint_dir).expanduser()
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
    return max(candidates, key=lambda item: item[0])[1].resolve()


def _looks_like_checkpoint(path: Path) -> bool:
    return (path / "_CHECKPOINT_METADATA").exists() or (path / "_METADATA").exists()


def _maybe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(np.asarray(value))
    except (TypeError, ValueError):
        return None
