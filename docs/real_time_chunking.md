# Real-Time Chunking (RTC)

By default the control loop stalls for a full inference round trip every
`open_loop_horizon` steps: `ActionChunkBroker` only queries the policy once the chunk it
is holding runs out, and the query is blocking. At 45 Hz with a ~200 ms inference, that is
a ~200 ms freeze roughly every 350 ms.

RTC removes the freeze. The policy is queried from a background thread while the robot
keeps draining the chunk it already has, and the *new* chunk is constrained to agree with
the *old* one over the steps that will have been executed by the time it arrives. Without
that constraint the two chunks disagree at the splice and the arm jerks.

- [Real-Time Chunking](https://www.physicalintelligence.company/research/real_time_chunking) (the `guided` mode)
- [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964) (the `trained` mode)

## Two modes

| | retraining | inference cost | constraint |
| --- | --- | --- | --- |
| `guided` (default) | none | ~1.5x | soft |
| `trained` | required | unchanged | hard |

Measured on one RTX 4090, `gemma_2b` + `gemma_300m`, `action_horizon=30`, `action_dim=32`,
`num_steps=10`, batch 1 (median of 10 runs):

| | pi0 | | pi05 | |
| --- | --- | --- | --- | --- |
| no RTC | 119 ms | 1.00x | 130 ms | 1.00x |
| `guided` | 180 ms | 1.51x | 202 ms | 1.56x |
| `trained` | 122 ms | 1.02x | 136 ms | 1.05x |

`trained` is free to within run-to-run noise on both. `guided` costs about half an
inference again, and the ratio barely differs between pi0 and pi05 — the extra work is the
same either way. pi05's larger `max_token_len` (200 vs 48) is why its baseline is slower,
but that cost sits in the prefix pass, which RTC does not touch.

**`guided`** treats the overlap as an inpainting problem and enforces it by correcting the
flow-matching velocity field at every denoising step. Nothing is clamped or overwritten,
so it works with any existing checkpoint. Each step needs one extra VJP, but only through
the action expert: the image and language prefix is a KV cache computed once outside the
denoising loop and is never differentiated. That is why the overhead is ~1.5x rather than
the ~3x a full backward pass would cost.

**`trained`** pins the prefix of the noisy chunk to the previous chunk at every step and
sets the per-token flow-matching timestep to 0 there, which tells the model those
positions are ground truth. There is no extra inference cost. It needs a checkpoint
trained with `Pi0Config(rtc_training_max_delay > 0)`: standard flow-matching training only
ever drives a whole chunk at a single timestep, so a mixed-timestep input is off the
training distribution and the velocity field there is undefined.

`RTCConfig(trained_prefix_noise=True)` is a fallback that runs `trained` on a checkpoint
that was *not* retrained: the prefix is re-noised to the current timestep instead of being
inserted clean, and the timestep stays uniform. Every token then sits on the training
marginal, but the model still does not know the prefix is pinned, so the suffix does not
adapt to it and you get the classic inpainting seam. Use it to compare, not in production.

## Serving

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_teleavatar_v2 \
    --policy.dir=checkpoints/pi05_teleavatar_v2/my_experiment/20000 \
    --rtc.enabled
```

| flag | default | meaning |
| --- | --- | --- |
| `--rtc.enabled` | off | turn RTC on |
| `--rtc.mode` | `GUIDED` | `GUIDED` or `TRAINED` |
| `--rtc.prefix-attention-schedule` | `EXP` | `LINEAR`, `EXP`, `ONES`, `ZEROS` |
| `--rtc.max-guidance-weight` | `10.0` | clip on the guidance weight (`GUIDED` only) |
| `--rtc.trained-prefix-noise` | off | training-free `TRAINED` fallback |

The server keeps the last chunk it returned in *model* space (before the output
transforms), because guidance has to happen in the space the model predicts in. The client
never round-trips actions back; it only reports how much of that chunk it has consumed.
One RTC client per server: a mismatched chunk id raises rather than silently guiding
against the wrong history.

An RTC-unaware client can still talk to an RTC-enabled server — it just never sends the
`rtc/*` fields, so every chunk is sampled unconstrained.

## Robot client

```bash
python examples/teleavatar_v2/main.py --remote-host 127.0.0.1 --rtc
```

| flag | default | meaning |
| --- | --- | --- |
| `--rtc` | off | use `RTCActionBroker` instead of `ActionChunkBroker` |
| `--rtc-warmup-steps` | `2` | inferences discarded before timing (the first JAX call compiles) |
| `--rtc-calibration-steps` | `5` | inferences timed to size the delay; `0` disables |
| `--rtc-inference-delay` | measured | override, in control steps |
| `--rtc-execution-horizon` | `2 x delay` | override, in control steps |

### Warm start

The two numbers RTC needs are properties of your machine, not of the policy:

- `inference_delay` — how many control steps one inference costs. The first
  `inference_delay` steps of a new chunk are pinned hard: the robot has already executed
  them and cannot take them back.
- `execution_horizon` — how far ahead the chunk is committed. Beyond it the new chunk is
  free to react to the new observation. It also sets the re-inference period.

Rather than guessing, the client measures them. The first `infer()` call runs
`warmup_steps + calibration_steps` inferences and returns only when it is done, so nothing
is commanded to the robot while measuring — on a platform where "stop publishing" means
"safe stop", the arm simply holds. It then reports:

```
RTC calibrated: inference_delay=9 steps (200 ms @ 45.0 Hz), execution_horizon=18, chunk=30
```

`inference_delay = ceil(quantile(latencies) * control_frequency) + delay_margin_steps`,
with `calibration_quantile=1.0` (worst case) by default — underestimating the delay leaves
already-executed steps unconstrained, which is the failure you are trying to avoid. The
delay keeps being tracked during the episode (`adapt_delay`), so a machine that gets
slower under load is followed rather than trusted from startup.

## Weight schedule

`get_prefix_weights(start=inference_delay, end=execution_horizon, total=action_horizon)`:

| range | weight | meaning |
| --- | --- | --- |
| `[0, inference_delay)` | 1 | already executed, physically cannot be revised |
| `[inference_delay, execution_horizon)` | 1 → 0 | soft transition |
| `[execution_horizon, H)` | 0 | free to react to the new observation |

`ZEROS` and `ONES` are the two hard-mask degenerate cases of that curve; `EXP` decays
faster than `LINEAR` and is the recommended default.

The guidance weight itself follows the paper's clipped posterior weighting,
`beta = (t^2 + (1-t)^2) / (t (1-t))`, which is symmetric about `t = 0.5` (minimum 2) and
diverges at both ends, hence the clip. Being symmetric, it is identical under openpi's
flipped time convention (`t = 1` is noise here, the opposite of the pi0 paper) and needs no
conversion.

## Training for `trained` mode

Set `rtc_training_max_delay` on the model config; `pi05_teleavatar_v2_rtc` is a ready-made
example:

```python
model=pi0_config.Pi0Config(pi05=True, action_horizon=30, rtc_training_max_delay=12)
```

Each sample gets a clean action prefix of a random length in `[0, rtc_training_max_delay]`,
driven at timestep 0 and excluded from the loss. Size it to the worst delay you expect in
control steps — at 45 Hz with a 200 ms inference that is ~9 steps, so 12 leaves headroom.
Do not oversize it: with a uniform delay, only `1 / (max_delay + 1)` of samples train the
plain unconditional objective.

Then serve with `--rtc.mode=TRAINED`. A `TRAINED` config on a checkpoint with
`rtc_training_max_delay=0` logs a warning rather than failing, so you can A/B it.

## Limitations

- **JAX only.** The PyTorch port in `openpi/models_pytorch` raises `NotImplementedError`.
- **Absolute actions only.** Guidance assumes the previous chunk stays meaningful as the
  robot moves, which holds for absolute joint targets. Delta actions
  (`use_delta_joint_actions=True`, e.g. `pi0_teleavatar_v2`) are anchored to the state they
  were predicted from and would need re-anchoring first; that is not implemented.
- **One RTC client per server.**
