from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp

from jax_fleet.types import Observation


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

        if raster.ndim == 3:
            raster = raster[None, ...]
            local_raster = local_raster[None, ...]
            structured = structured[None, ...]
            candidates = candidates[None, ...]
            mask = mask[None, ...]

        x = nn.Conv(16, (5, 5), strides=(2, 2), padding="SAME", name="global_conv_1")(raster)
        x = nn.relu(x)
        x = nn.Conv(24, (3, 3), strides=(2, 2), padding="SAME", name="global_conv_2")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden_size, name="global_dense")(x)
        x = nn.relu(x)

        local = nn.Conv(16, (5, 5), strides=(2, 2), padding="SAME", name="local_conv_1")(local_raster)
        local = nn.relu(local)
        local = nn.Conv(24, (3, 3), strides=(2, 2), padding="SAME", name="local_conv_2")(local)
        local = nn.relu(local)
        local = local.reshape((local.shape[0], -1))
        local = nn.Dense(self.hidden_size, name="local_dense")(local)
        local = nn.relu(local)

        s = nn.Dense(self.hidden_size, name="structured_dense")(structured)
        s = nn.relu(s)
        global_features = nn.Dense(self.hidden_size, name="fused_dense")(
            jnp.concatenate([x, local, s], axis=-1)
        )
        global_features = nn.tanh(global_features)

        action_features = nn.Dense(self.hidden_size)(candidates)
        action_features = nn.tanh(action_features)
        fused = nn.tanh(action_features + global_features[:, None, :])
        logits = nn.Dense(1)(fused).squeeze(-1)
        logits = jnp.where(mask, logits, -1.0e9)

        value = nn.Dense(self.hidden_size)(global_features)
        value = nn.tanh(value)
        value = nn.Dense(1)(value).squeeze(-1)
        return logits, value
