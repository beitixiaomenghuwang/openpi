#!/usr/bin/env python3
"""Run a UMI relative end-effector policy on the TA2 robot.

Start the server with ``--policy.config=pi05_umi_endeffector``, then run:
``python examples/teleavatar_v2/main_relative_endeffector.py``.
"""

import dataclasses
import logging
import pathlib
import sys

import numpy as np
from openpi_client import base_policy
from openpi_client import websocket_client_policy
from openpi_client.runtime import runtime
from openpi_client.runtime.agents import policy_agent
import tyro

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.teleavatar_v2 import env_relative_endeffector as _env  # noqa: E402
from openpi.policies.umi_policy import relative_actions_to_absolute  # noqa: E402


class RelativeActionChunkBroker(base_policy.BasePolicy):
    """Convert one predicted relative action chunk using one state snapshot."""

    def __init__(self, policy: base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._last_results = None
        self._cur_step = 0

    def infer(self, obs: dict) -> dict:
        if self._last_results is None:
            result = self._policy.infer(obs)
            raw_actions = np.asarray(result["actions"], dtype=np.float32)
            current_state = np.asarray(obs["observation/state"], dtype=np.float32)
            absolute_actions = relative_actions_to_absolute(current_state, raw_actions)
            absolute_actions[:, [7, 15]] = (absolute_actions[:, [7, 15]] > 0.5).astype(np.float32)
            self._last_results = {**result, "actions": absolute_actions}
            self._cur_step = 0

        result = {
            key: value[self._cur_step] if isinstance(value, np.ndarray) else value
            for key, value in self._last_results.items()
        }
        self._cur_step += 1
        if self._cur_step >= self._action_horizon:
            self._last_results = None
        return result

    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


@dataclasses.dataclass
class Args:
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000
    control_frequency: float = 30.0
    open_loop_horizon: int = 30
    prompt: str = "pick up the marker using the right gripper, hand it to the left gripper, and put it in the cup"
    num_episodes: int = 100
    max_episode_steps: int = 0


def main(args: Args) -> None:
    logging.info("Connecting to relative end-effector policy at ws://%s:%d", args.remote_host, args.remote_port)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.remote_host, port=args.remote_port)
    environment = _env.TeleavatarRelativeEndEffectorEnvironment(prompt=args.prompt)
    agent = policy_agent.PolicyAgent(
        policy=RelativeActionChunkBroker(policy, action_horizon=args.open_loop_horizon)
    )
    runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[],
        max_hz=args.control_frequency,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    ).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    main(tyro.cli(Args))
