"""Real-Time Chunking: keeps a newly sampled action chunk consistent with the one the robot
is already executing, so the policy can be queried while the robot keeps moving.

See docs/real_time_chunking.md. Time convention here is openpi's: t=1 is noise, t=0 is data.
"""

import dataclasses
import enum
import math

import jax.numpy as jnp

from openpi.shared import array_typing as at


class RTCMode(enum.Enum):
    # Soft inpainting via velocity-field guidance. Works with any checkpoint.
    GUIDED = "guided"
    # Hard inpainting: pin the prefix and drive it at timestep 0. Needs a checkpoint
    # trained with `Pi0Config.rtc_training_max_delay > 0`.
    TRAINED = "trained"


class PrefixAttentionSchedule(enum.Enum):
    LINEAR = "linear"
    EXP = "exp"
    # Hard masks: the two degenerate cases of the same curve.
    ONES = "ones"
    ZEROS = "zeros"


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    """Static inference-time settings, baked into the jitted `sample_actions`.

    The per-request quantities (delay, horizon, previous chunk) are passed as arrays
    instead, so they never trigger a recompile.
    """

    mode: RTCMode = RTCMode.GUIDED
    prefix_attention_schedule: PrefixAttentionSchedule = PrefixAttentionSchedule.EXP
    max_guidance_weight: float = 10.0
    # TRAINED only: re-noise the pinned prefix to the current timestep instead of inserting
    # it clean, keeping the input on-distribution for a checkpoint that was not trained
    # with RTC conditioning. The model still does not know the prefix is fixed.
    trained_prefix_noise: bool = False


@at.typecheck
def get_prefix_weights(
    start: at.Int[at.Array, ""] | int,
    end: at.Int[at.Array, ""] | int,
    total: int,
    schedule: PrefixAttentionSchedule,
) -> at.Float[at.Array, " {total}"]:
    """Soft prefix-attention weights: 1 below `start` (already executed), decaying to 0 at
    `end` (free to react to the new observation).

    `start`/`end` may be traced, so a changing measured delay does not recompile.
    """
    start = jnp.clip(jnp.asarray(start), 0, total)
    end = jnp.clip(jnp.asarray(end), 0, total)
    start = jnp.minimum(start, end)
    idx = jnp.arange(total)

    if schedule is PrefixAttentionSchedule.ONES:
        return jnp.where(idx < end, 1.0, 0.0)
    if schedule is PrefixAttentionSchedule.ZEROS:
        return jnp.where(idx < start, 1.0, 0.0)

    # Matches torch.linspace(1, 0, n + 2)[1:-1] over the transition, n = end - start.
    ramp = 1.0 - (idx - start + 1) / (end - start + 1)
    weights = jnp.clip(jnp.where(idx < start, 1.0, ramp), 0.0, 1.0)
    weights = jnp.where(idx >= end, 0.0, weights)
    if schedule is PrefixAttentionSchedule.EXP:
        # Convex, and fixes both endpoints (0 -> 0, 1 -> 1).
        weights = weights * jnp.expm1(weights) / (math.e - 1.0)
    return weights


@at.typecheck
def guidance_weight(
    time: at.Float[at.Array, ""] | float,
    max_guidance_weight: float,
) -> at.Float[at.Array, ""]:
    """Clipped guidance weight `c_t / r_t^2` from the RTC paper, which reduces to
    `(t^2 + (1-t)^2) / (t (1-t))`.

    Symmetric under t <-> 1-t, so it is identical in openpi's flipped time convention. It
    bottoms out at 2 and diverges at both ends, hence the clip.
    """
    t = jnp.clip(jnp.asarray(time, dtype=jnp.float32), 1e-6, 1.0 - 1e-6)
    return jnp.minimum((t**2 + (1.0 - t) ** 2) / (t * (1.0 - t)), max_guidance_weight)
