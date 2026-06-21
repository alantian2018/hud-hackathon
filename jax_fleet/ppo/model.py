from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp
import numpy as np

from jax_fleet.types import Observation


# PPO-standard orthogonal initialization. Hidden/conv layers feeding ReLU use a
# sqrt(2) gain, tanh layers use 1.0, the policy logit head uses a tiny 0.01 gain
# so the initial policy is near-uniform, and the value head uses 1.0.
_RELU_INIT = nn.initializers.orthogonal(np.sqrt(2.0))
_TANH_INIT = nn.initializers.orthogonal(1.0)
_POLICY_HEAD_INIT = nn.initializers.orthogonal(0.01)
_VALUE_HEAD_INIT = nn.initializers.orthogonal(1.0)


class ActorCritic(nn.Module):
    max_degree: int
    hidden_size: int = 64

    @nn.compact
    def __call__(self, observation: Observation):
        raster = observation.raster
        local_raster = observation.local_raster
        structured = observation.structured
        candidates = observation.candidate_edges
        mask = observation.action_mask

        single_observation = raster.ndim == 3
        if raster.ndim == 3:
            raster = raster[None, ...]
            local_raster = local_raster[None, ...]
            structured = structured[None, ...]
            candidates = candidates[None, ...]
            mask = mask[None, ...]

        x = nn.Conv(16, (5, 5), strides=(2, 2), padding="SAME", kernel_init=_RELU_INIT, name="global_conv_1")(raster)
        x = nn.relu(x)
        x = nn.Conv(24, (3, 3), strides=(2, 2), padding="SAME", kernel_init=_RELU_INIT, name="global_conv_2")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden_size, kernel_init=_RELU_INIT, name="global_dense")(x)
        x = nn.relu(x)

        local = nn.Conv(16, (5, 5), strides=(2, 2), padding="SAME", kernel_init=_RELU_INIT, name="local_conv_1")(local_raster)
        local = nn.relu(local)
        local = nn.Conv(24, (3, 3), strides=(2, 2), padding="SAME", kernel_init=_RELU_INIT, name="local_conv_2")(local)
        local = nn.relu(local)
        local = local.reshape((local.shape[0], -1))
        local = nn.Dense(self.hidden_size, kernel_init=_RELU_INIT, name="local_dense")(local)
        local = nn.relu(local)

        s = nn.Dense(self.hidden_size, kernel_init=_RELU_INIT, name="structured_dense")(structured)
        s = nn.relu(s)
        global_features = nn.Dense(self.hidden_size, kernel_init=_TANH_INIT, name="fused_dense")(
            jnp.concatenate([x, local, s], axis=-1)
        )
        global_features = nn.tanh(global_features)

        edge_embedding = nn.Dense(self.hidden_size, kernel_init=_TANH_INIT, name="edge_shared_dense_1")(candidates)
        edge_embedding = nn.tanh(edge_embedding)
        edge_embedding = nn.Dense(self.hidden_size, kernel_init=_TANH_INIT, name="edge_shared_dense_2")(edge_embedding)
        edge_embedding = nn.tanh(edge_embedding)
        global_for_edges = jnp.broadcast_to(
            global_features[:, None, :],
            (global_features.shape[0], self.max_degree, global_features.shape[-1]),
        )
        action_input = jnp.concatenate([global_for_edges, edge_embedding], axis=-1)
        action_hidden = nn.Dense(self.hidden_size, kernel_init=_TANH_INIT, name="shared_action_scorer_dense")(action_input)
        action_hidden = nn.tanh(action_hidden)
        logits = nn.Dense(1, kernel_init=_POLICY_HEAD_INIT, name="shared_action_scorer_out")(action_hidden).squeeze(-1)
        logits = jnp.where(mask, logits, -1.0e9)

        valid_edges = mask.astype(jnp.float32)[..., None]
        edge_count = jnp.maximum(valid_edges.sum(axis=1), 1.0)
        pooled_edges = (edge_embedding * valid_edges).sum(axis=1) / edge_count
        value_input = jnp.concatenate([global_features, pooled_edges], axis=-1)
        value = nn.Dense(self.hidden_size, kernel_init=_TANH_INIT, name="value_dense_1")(value_input)
        value = nn.tanh(value)
        value = nn.Dense(1, kernel_init=_VALUE_HEAD_INIT, name="value_out")(value).squeeze(-1)
        if single_observation:
            return logits[0], value[0]
        return logits, value
