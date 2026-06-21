from __future__ import annotations

import jax.numpy as jnp

from jax_fleet.observations import (
    CE_ETA_IMPROVEMENT,
    CE_REACHABLE_10M,
    CE_WAIT_WEIGHTED_ADVANTAGE,
    CE_WAIT_WEIGHTED_ETA_IMPROVEMENT,
)
from jax_fleet.types import Observation


def marginal_value_edge_scores(observation: Observation):
    """Scores valid candidate edges using hand-built marginal-value features."""
    candidates = observation.candidate_edges
    score = (
        candidates[..., CE_ETA_IMPROVEMENT]
        + candidates[..., CE_WAIT_WEIGHTED_ETA_IMPROVEMENT]
        + candidates[..., CE_WAIT_WEIGHTED_ADVANTAGE]
        + 0.25 * candidates[..., CE_REACHABLE_10M]
    )
    return jnp.where(observation.action_mask, score, -jnp.inf)


def choose_marginal_value_action(observation: Observation):
    """Returns the valid edge slot with the largest heuristic marginal value."""
    return jnp.argmax(marginal_value_edge_scores(observation), axis=-1).astype(jnp.int32)
