from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import rtc as _rtc
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

logger = logging.getLogger(__name__)

BasePolicy: TypeAlias = _base_policy.BasePolicy

# RTC fields the client adds to the observation dict; stripped before the input transforms.
RTC_PREV_CHUNK_ID = "rtc/prev_chunk_id"
RTC_PREFIX_START = "rtc/prefix_start"
RTC_INFERENCE_DELAY = "rtc/inference_delay"
RTC_EXECUTION_HORIZON = "rtc/execution_horizon"
# Keys the server adds to the response.
RTC_CHUNK_ID = "rtc/chunk_id"
RTC_ACTION_HORIZON = "rtc/action_horizon"


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        rtc_config: _rtc.RTCConfig | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            rtc_config: Enables Real-Time Chunking: the policy keeps the raw (model-space)
                chunk it last returned and constrains the next one to agree with it over the
                overlap the client reports. JAX models only.
        """
        self._model = model
        self._input_transforms = list(transforms)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        self._rtc_config = rtc_config
        # Last chunk returned, in raw model space (guidance has to happen there), so the
        # client never round-trips actions back, plus the delta-action anchor it was encoded
        # against (None for policies whose actions are already absolute).
        self._rtc_prev_chunk: np.ndarray | None = None
        self._rtc_prev_anchor: np.ndarray | None = None
        self._rtc_chunk_id = 0
        self._rtc_normalize = next((t for t in self._input_transforms if isinstance(t, _transforms.Normalize)), None)
        self._rtc_logged_anchor = False

        if self._is_pytorch_model:
            if rtc_config is not None:
                raise NotImplementedError("RTC is only implemented for the JAX models (openpi.models.pi0).")
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # `rtc_config` selects the mode and schedule, so it changes the traced graph and
            # has to be static; the per-request delay/horizon are passed as arrays.
            self._sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("rtc_config",))
            self._rng = rng or jax.random.key(0)
            if rtc_config is not None:
                max_delay = getattr(model, "rtc_training_max_delay", 0)
                if rtc_config.mode is _rtc.RTCMode.TRAINED and max_delay == 0 and not rtc_config.trained_prefix_noise:
                    logger.warning(
                        "RTC mode=TRAINED on a checkpoint with rtc_training_max_delay=0: the pinned prefix is "
                        "off-distribution. Retrain with Pi0Config(rtc_training_max_delay=...), or set "
                        "RTCConfig(trained_prefix_noise=True)."
                    )
                logger.info("RTC enabled: %s", rtc_config)

    def _take_rtc_request(self, inputs: dict) -> dict[str, int] | None:
        """Pop the RTC fields out of the raw observation dict, returning None when the client
        is not driving RTC this step (first chunk after a reset, or an RTC-unaware client)."""
        request = {k: inputs.pop(k) for k in list(inputs) if k.startswith("rtc/")}
        if not request or self._rtc_config is None:
            return None
        prev_chunk_id = int(request.get(RTC_PREV_CHUNK_ID, 0))
        if prev_chunk_id == 0:  # nothing executing yet: sample unconstrained
            return None
        if prev_chunk_id != self._rtc_chunk_id or self._rtc_prev_chunk is None:
            raise RuntimeError(
                f"RTC chunk id mismatch: client is executing chunk {prev_chunk_id} but the server last "
                f"returned {self._rtc_chunk_id}. Only one RTC client per server is supported."
            )
        return {
            "prefix_start": int(request.get(RTC_PREFIX_START, 0)),
            "inference_delay": int(request.get(RTC_INFERENCE_DELAY, 0)),
            "execution_horizon": int(request.get(RTC_EXECUTION_HORIZON, self._model.action_horizon)),
        }

    def _rtc_anchor(self, inputs: dict) -> np.ndarray | None:
        """The delta-action anchor for this observation, in normalized action units.

        Walks the input transforms so each sees the data in its own frame, and stops at the
        first one that declares an anchor. Returns None when the actions are absolute, which
        is when no re-anchoring is needed.
        """
        data = inputs
        for transform in self._input_transforms:
            if isinstance(transform, _transforms.DeltaActionAnchor):
                anchor = transform.delta_action_anchor(data)
                if anchor is not None:
                    if not self._rtc_logged_anchor:
                        self._rtc_logged_anchor = True
                        logger.info("RTC: delta actions detected, re-anchoring against %s", type(transform).__name__)
                    if self._rtc_normalize is not None:
                        anchor = self._rtc_normalize({"actions": np.asarray(anchor)})["actions"]
                    return np.asarray(anchor)
            data = transform(data)
        return None

    def _rtc_sample_kwargs(self, request: dict[str, int] | None, anchor: np.ndarray | None) -> dict[str, Any]:
        """Build the RTC arguments to `sample_actions` from a client request."""
        if self._rtc_config is None:
            return {}
        kwargs: dict[str, Any] = {"rtc_config": self._rtc_config}
        if request is None:
            return kwargs
        if request["prefix_start"] >= len(self._rtc_prev_chunk):
            # No overlap left; guiding towards zero padding would be worse than not guiding.
            logger.warning(
                "RTC: previous chunk fully consumed (prefix_start=%d >= %d); sampling unconstrained.",
                request["prefix_start"],
                len(self._rtc_prev_chunk),
            )
            return kwargs
        # Left-align: the client has executed `prefix_start` steps of the previous chunk, so
        # its step `prefix_start + k` and the new chunk's step `k` are the same control step.
        # The zero-padded tail carries weight 0, so it never contributes.
        prev = self._rtc_prev_chunk
        start = np.clip(request["prefix_start"], 0, len(prev))
        valid = len(prev) - start
        aligned = np.zeros_like(prev)
        aligned[:valid] = prev[start:]
        if self._rtc_prev_anchor is not None and anchor is not None:
            # Delta actions are encoded against the observation they were predicted from, so
            # the previous chunk sits in an older frame -- offset by whatever the robot did in
            # between. Normalization is affine, so the difference of the two normalized
            # anchors is exactly the offset in model space.
            aligned[:valid] += _transforms.pad_to_dim(self._rtc_prev_anchor - anchor, prev.shape[-1])
        kwargs["prev_chunk"] = jnp.asarray(aligned)[np.newaxis, ...]
        kwargs["inference_delay"] = jnp.asarray(request["inference_delay"], dtype=jnp.int32)
        kwargs["execution_horizon"] = jnp.asarray(request["execution_horizon"], dtype=jnp.int32)
        return kwargs

    @override
    def reset(self) -> None:
        """Drop the RTC history so the next chunk is sampled unconstrained."""
        self._rtc_prev_chunk = None
        self._rtc_prev_anchor = None
        self._rtc_chunk_id = 0

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        rtc_request = self._take_rtc_request(inputs)
        rtc_anchor = self._rtc_anchor(inputs) if self._rtc_config is not None else None
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        sample_kwargs.update(self._rtc_sample_kwargs(rtc_request, rtc_anchor))

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {"state": inputs["state"], "actions": actions}
        model_time = time.monotonic() - start_time

        if self._rtc_config is not None:
            # Stash in model space, before the output transforms.
            self._rtc_prev_chunk = np.asarray(actions[0, ...])
            self._rtc_prev_anchor = rtc_anchor
            self._rtc_chunk_id += 1
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if self._rtc_config is not None:
            outputs[RTC_CHUNK_ID] = self._rtc_chunk_id
            outputs[RTC_ACTION_HORIZON] = self._model.action_horizon
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
