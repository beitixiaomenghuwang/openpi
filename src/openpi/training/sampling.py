"""Task-balanced frame sampling for LeRobot datasets whose episodes carry a task hierarchy.

Built for the YUBI UMI Arena data, where every episode is one primitive action (PA) whose prompt is
``tasks[0]``, PAs belong to a short-horizon task (SHT, ``short_horizon_task``), and the PAs cut from
one recording share a ``uuid``. Uniform frame sampling lets the largest SHTs and the longest PAs
dominate training, so sampling is done in three levels:

1. SHT share proportional to frames**temperature; ``boosted_groups`` get ``boost`` on top and their
   combined share is capped at ``max_boosted_share``.
2. Within an SHT, prompt share proportional to episodes**temperature.
3. Within a prompt, every frame is equally likely.

No SHT or (SHT, prompt) gets more than ``max_upsample`` times its natural share (its fraction of the
training frames); the excess is spread over the rest. Held-out SHTs and a fraction of the recordings
of every SHT get zero probability, so validation data never needs a separate dataset.
"""

from collections.abc import Iterator, Mapping
import dataclasses
import hashlib
import logging
from typing import Any

import numpy as np
import torch.utils.data


@dataclasses.dataclass(frozen=True)
class SamplingPlan:
    # Probability of drawing each episode, in episode order; zero for held-out episodes.
    episode_probs: np.ndarray
    # Global index of each episode's first frame (the LeRobot `episode_data_index["from"]`).
    from_index: np.ndarray
    lengths: np.ndarray
    holdout: np.ndarray
    # Share of the draws and of the training frames per group.
    group_shares: dict[str, float]
    group_natural_shares: dict[str, float]


@dataclasses.dataclass(frozen=True)
class TaskBalancedSampler:
    """Factory for an infinite i.i.d. frame sampler with task-balanced probabilities."""

    temperature: float = 0.5
    max_upsample: float = 5.0
    boosted_groups: tuple[str, ...] = ()
    boost: float = 2.0
    max_boosted_share: float = 1.0
    holdout_groups: tuple[str, ...] = ()
    # Fraction of the recordings of each group to hold out, picked by hash so the split is stable.
    holdout_recording_fraction: float = 0.0

    group_key: str = "short_horizon_task"
    recording_key: str = "uuid"

    def plan(self, episodes: Mapping[int, Mapping[str, Any]]) -> SamplingPlan:
        """Computes the episode probabilities. `episodes` is LeRobot's `meta.episodes`."""
        ordered = [episodes[i] for i in sorted(episodes)]
        lengths = np.array([ep["length"] for ep in ordered], dtype=np.int64)
        groups = np.array([ep[self.group_key] for ep in ordered])
        recordings = np.array([ep[self.recording_key] for ep in ordered])
        # A prompt can occur in several groups, so prompts are keyed by (group, prompt).
        pairs = np.array([f"{ep[self.group_key]}\n{ep['tasks'][0]}" for ep in ordered])
        from_index = np.concatenate([[0], np.cumsum(lengths)[:-1]])

        group_names, group_ids = np.unique(groups, return_inverse=True)
        for name in (*self.boosted_groups, *self.holdout_groups):
            if name not in group_names:
                raise ValueError(f"Group {name!r} is not in the dataset.")
        held_recordings = self._held_out_recordings(group_names, group_ids, recordings)
        holdout = np.isin(groups, self.holdout_groups) | np.isin(recordings, held_recordings)
        train_lengths = np.where(holdout, 0, lengths)
        total_frames = train_lengths.sum()

        group_natural = np.bincount(group_ids, weights=train_lengths, minlength=len(group_names)) / total_frames
        group_shares = self._group_shares(group_names, group_natural)

        _, pair_ids = np.unique(pairs, return_inverse=True)
        pair_group = np.zeros(pair_ids.max() + 1, dtype=np.int64)
        pair_group[pair_ids] = group_ids
        pair_frames = np.bincount(pair_ids, weights=train_lengths)
        pair_episodes = np.bincount(pair_ids, weights=~holdout)
        pair_shares = np.zeros(len(pair_frames))
        for group in np.flatnonzero(group_shares):
            members = (pair_group == group) & (pair_frames > 0)
            # Cap each prompt at max_upsample times its natural share of all training frames.
            upper = self.max_upsample * (pair_frames[members] / total_frames) / group_shares[group]
            pair_shares[members] = group_shares[group] * _capped(pair_episodes[members] ** self.temperature, upper)

        probs = np.where(holdout, 0.0, pair_shares[pair_ids] * lengths / np.maximum(pair_frames[pair_ids], 1))
        in_training = group_natural > 0
        return SamplingPlan(
            episode_probs=probs / probs.sum(),
            from_index=from_index,
            lengths=lengths,
            holdout=holdout,
            group_shares=dict(zip(group_names[in_training].tolist(), group_shares[in_training].tolist(), strict=True)),
            group_natural_shares=dict(
                zip(group_names[in_training].tolist(), group_natural[in_training].tolist(), strict=True)
            ),
        )

    def create(self, episodes: Mapping[int, Mapping[str, Any]], *, seed: int) -> "FrameSampler":
        plan = self.plan(episodes)
        logging.info(
            "Task-balanced sampling: %d training / %d held-out episodes (%d / %d frames).",
            (~plan.holdout).sum(),
            plan.holdout.sum(),
            plan.lengths[~plan.holdout].sum(),
            plan.lengths[plan.holdout].sum(),
        )
        for name, share in sorted(plan.group_shares.items(), key=lambda item: -item[1]):
            natural = plan.group_natural_shares[name]
            logging.info("  %6.2f%% (natural %6.2f%%, x%.2f) %s", 100 * share, 100 * natural, share / natural, name)
        return FrameSampler(plan, seed=seed)

    def _held_out_recordings(self, group_names: np.ndarray, group_ids: np.ndarray, recordings: np.ndarray) -> list[str]:
        if self.holdout_recording_fraction <= 0:
            return []
        held = []
        for group in range(len(group_names)):
            unique = sorted(set(recordings[group_ids == group].tolist()), key=_stable_hash)
            held.extend(unique[: round(self.holdout_recording_fraction * len(unique))])
        return held

    def _group_shares(self, names: np.ndarray, natural: np.ndarray) -> np.ndarray:
        boosted = np.isin(names, self.boosted_groups)
        upper = self.max_upsample * natural
        shares = _capped(natural**self.temperature * np.where(boosted, self.boost, 1.0), upper)
        if boosted.any() and shares[boosted].sum() > self.max_boosted_share:
            # Pin the boosted groups to their maximum and cap both sides separately.
            limit = self.max_boosted_share
            shares[boosted] = limit * _capped(shares[boosted], upper[boosted] / limit)
            shares[~boosted] = (1 - limit) * _capped(shares[~boosted], upper[~boosted] / (1 - limit))
        return shares


class FrameSampler(torch.utils.data.Sampler[int]):
    """Draws global frame indices forever: an episode by its probability, then a uniform frame in it."""

    def __init__(self, plan: SamplingPlan, *, seed: int, block_size: int = 65_536):
        self._cdf = np.cumsum(plan.episode_probs)
        self._cdf /= self._cdf[-1]
        self._from_index = plan.from_index
        self._lengths = plan.lengths
        self._block_size = block_size
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[int]:
        while True:
            episodes = np.searchsorted(self._cdf, self._rng.random(self._block_size), side="right")
            episodes = np.minimum(episodes, len(self._cdf) - 1)
            offsets = (self._rng.random(self._block_size) * self._lengths[episodes]).astype(np.int64)
            yield from (self._from_index[episodes] + offsets).tolist()


def _capped(weights: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Normalizes `weights`, clipping entries at `upper` and spreading the excess over the rest in proportion."""
    if upper.sum() < 1 - 1e-9:
        raise ValueError("The caps leave less than the full probability mass.")
    shares = weights / weights.sum()
    capped = np.zeros(len(shares), dtype=bool)
    while (over := ~capped & (shares > upper)).any():
        capped |= over
        free = ~capped & (shares > 0)
        if not free.any():
            return np.where(capped, upper, 0.0) / upper[capped].sum()
        shares = np.where(capped, upper, shares * (1 - upper[capped].sum()) / shares[free].sum())
    return shares


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()
