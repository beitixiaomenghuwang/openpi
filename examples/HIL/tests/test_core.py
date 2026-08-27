#!/usr/bin/env python3
"""Hardware-independent tests for the minimal HIL runtime."""

from __future__ import annotations

import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hil.action_chunk import (
    ActionFrame,
    EndpointCommand,
    decode_action_chunk,
    encode_action_chunk,
)
from hil.hil_supervisor import (
    HILSupervisor,
    PoseArray,
    SourceState,
    policy_handover_target_reached,
)
from hil.pose_math import (
    anchored_pose,
    limit_pose_step,
    quaternion_angle,
    quaternion_to_matrix,
)
from hil.sec_dev_protocol import decode_sec_dev_pose
from hil.topics import VR_ZENOH_DEVICEPOSE_KEY
from hil import topics
from hil.policy_topic_adapter import FourTopicBuffer, PolicyTopicAdapter
from hil.vr_teleop_server import VRTeleopNode, parse_args as parse_vr_teleop_args
from hil.xr_protocol import parse_device_pose
from hil.zenoh_pose_sampler import summarize_activity
from hil.zenoh_wire import parse_wire_fields
from tools.api_teleop_test import parse_args as parse_api_teleop_args
from tools.zenoh_button_test import (
    describe_edges,
    extract_controller_inputs,
    extract_input_mask,
    extract_stick_values,
)


FIXTURES = Path(__file__).with_name("fixtures")


class PoseMathTest(unittest.TestCase):
    def test_anchor_is_continuous_at_takeover(self):
        position, quaternion = anchored_pose(
            (1.0, 2.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 2.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.1, -0.2, -0.4),
            (0.0, 0.0, 0.0, 1.0),
        )
        np.testing.assert_allclose(position, (0.1, -0.2, -0.4))
        np.testing.assert_allclose(quaternion, (0.0, 0.0, 0.0, 1.0))

    def test_anchor_preserves_relative_translation(self):
        position, _ = anchored_pose(
            (1.1, 1.8, 3.3),
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 2.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.2, 0.0, -0.4),
            (0.0, 0.0, 0.0, 1.0),
            position_scale=0.5,
        )
        np.testing.assert_allclose(position, (0.25, -0.1, -0.25))

    def test_anchor_matches_old_hil_offset_transform(self):
        root_half = np.sqrt(0.5)
        raw_anchor_q = (root_half, 0.0, 0.0, root_half)
        robot_anchor_q = (0.0, 0.0, root_half, root_half)
        raw_q = (0.0, root_half, 0.0, root_half)
        position, quaternion = anchored_pose(
            (0.4, -0.1, 0.2),
            raw_q,
            (0.3, -0.2, 0.1),
            raw_anchor_q,
            (0.1, 0.2, -0.4),
            robot_anchor_q,
        )
        np.testing.assert_allclose(position, (0.2, 0.3, -0.3))
        expected_rotation = (
            quaternion_to_matrix(robot_anchor_q)
            @ quaternion_to_matrix(raw_anchor_q).T
            @ quaternion_to_matrix(raw_q)
        )
        np.testing.assert_allclose(
            quaternion_to_matrix(quaternion), expected_rotation, atol=1e-12
        )

    def test_step_limits_translation(self):
        position, quaternion = limit_pose_step(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            max_translation=0.01,
            max_rotation=0.1,
        )
        self.assertAlmostEqual(float(np.linalg.norm(position)), 0.01)
        self.assertLessEqual(
            quaternion_angle((0.0, 0.0, 0.0, 1.0), quaternion), 0.100001
        )


class XRProtocolTest(unittest.TestCase):
    def test_zenoh_pose_key_matches_sec_devicepose_contract(self):
        self.assertEqual(VR_ZENOH_DEVICEPOSE_KEY, "sec/xr/devicepose")
        args = parse_vr_teleop_args([])
        self.assertEqual(args.pose_transport, "zenoh")
        self.assertEqual(args.zenoh_key, VR_ZENOH_DEVICEPOSE_KEY)

    def test_short_packet_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_device_pose(b"short")


class ZenohPoseSamplerTest(unittest.TestCase):
    def test_current_sec_dev_payload_is_wire_parseable(self):
        payload = (FIXTURES / "pose_payload.bin").read_bytes()
        fields = parse_wire_fields(payload)
        self.assertEqual(
            [field.number for field in fields],
            [1, 2, 3, 4, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28],
        )
        field14 = next(field for field in fields if field.number == 14)
        self.assertEqual(field14.wire_type, 2)
        self.assertEqual(len(field14.raw), 28)

    def test_activity_summary_detects_pose_changes_without_assigning_semantics(self):
        first = (FIXTURES / "pose_payload.bin").read_bytes()
        second = (FIXTURES / "pose_payload_button.bin").read_bytes()
        summary = summarize_activity(
            [parse_wire_fields(first), parse_wire_fields(second)]
        )
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(
            summary["fields"]["14"][0]["changes_between_samples"], 1
        )
        self.assertEqual(
            summary["fields"]["8"][0]["changes_between_samples"], 0
        )


class ZenohButtonTest(unittest.TestCase):
    def test_button_mask_edges(self):
        self.assertEqual(describe_edges(0, 0x05), [
            "pressed: left X (0x01)",
            "pressed: right A (0x04)",
        ])
        self.assertEqual(describe_edges(0x05, 0), [
            "released: left X (0x01)",
            "released: right A (0x04)",
        ])

    def test_capture_payload_has_no_button_mask_when_idle(self):
        payload = (FIXTURES / "pose_payload.bin").read_bytes()
        self.assertEqual(extract_input_mask(payload), 0)

    def test_stick_field_mapping(self):
        import struct

        payload = (
            bytes((8 << 3 | 2, 8))
            + struct.pack("<ff", -0.25, 0.75)
            + bytes((9 << 3 | 2, 8))
            + struct.pack("<ff", 0.5, -1.0)
        )
        self.assertEqual(
            extract_stick_values(payload),
            {"left stick": (-0.25, 0.75), "right stick": (0.5, -1.0)},
        )

    def test_omitted_analog_fields_are_not_fabricated(self):
        payload = (FIXTURES / "pose_payload.bin").read_bytes()
        mask, sticks, analog = extract_controller_inputs(payload)
        self.assertEqual(mask, 0)
        self.assertEqual(sticks["left stick"], (0.0, 0.0))
        self.assertEqual(sticks["right stick"], (0.0, 0.0))
        self.assertEqual(analog, {})


class DirectAPITeleopDecodeTest(unittest.TestCase):
    def test_api_teleop_args_build_wrist_offset_mapping(self):
        args = parse_api_teleop_args([])
        np.testing.assert_allclose(
            args.wrist_offsets["left"], (0.13205294, -0.03416498, 0.03397751)
        )
        np.testing.assert_allclose(
            args.wrist_offsets["right"], (0.12516124, 0.04039848, 0.03931866)
        )

    def test_sec_dev_pose_payload_decodes_both_controllers(self):
        payload = (FIXTURES / "pose_payload.bin").read_bytes()
        decoded = decode_sec_dev_pose(payload)
        self.assertEqual(decoded.input_mask, 0)
        self.assertEqual(set(decoded.poses), {"head", "left", "right"})
        self.assertEqual(decoded.analog, {"left": {}, "right": {}})
        for pose in decoded.poses.values():
            self.assertEqual(pose.position.shape, (3,))
            self.assertEqual(pose.quaternion.shape, (4,))
            self.assertAlmostEqual(float(np.linalg.norm(pose.quaternion)), 1.0)

    def test_sec_dev_button_payload_decodes_bitmask(self):
        # Add a field-6 varint to a captured pose packet. The historical
        # button fixture predates the sec_dev input-mask field.
        payload = b"\x30\x04" + (FIXTURES / "pose_payload.bin").read_bytes()
        decoded = decode_sec_dev_pose(payload)
        self.assertEqual(decoded.input_mask, 0x04)


class HILTeleopSecDevTest(unittest.TestCase):
    def _node_without_ros(self):
        args = parse_vr_teleop_args([])
        node = VRTeleopNode.__new__(VRTeleopNode)
        node.wrist_offsets = {
            "left": np.asarray(args.left_wrist_offset, dtype=np.float64),
            "right": np.asarray(args.right_wrist_offset, dtype=np.float64),
        }
        node.gripper_input = "trigger"
        node._sec_grippers = {"left": 0.0, "right": 0.0}
        return node

    def test_trigger_press_and_release_are_forwarded_to_gripper(self):
        import struct

        base = (FIXTURES / "pose_payload.bin").read_bytes()
        pressed = b"\x55" + struct.pack("<f", 1.0) + b"\x5d" + struct.pack("<f", 1.0) + base
        released = b"\x55" + struct.pack("<f", 0.0) + b"\x5d" + struct.pack("<f", 0.0) + base
        node = self._node_without_ros()

        pressed_packet = decode_sec_dev_pose(pressed)
        pressed_action = node._sec_controller_command("left", pressed_packet)
        self.assertEqual(pressed_action.gripper, 1.0)

        released_packet = decode_sec_dev_pose(released)
        released_action = node._sec_controller_command("left", released_packet)
        self.assertEqual(released_action.gripper, 0.0)

    def test_missing_trigger_field_uses_mask_to_distinguish_hold_and_release(self):
        import struct

        base = (FIXTURES / "pose_payload.bin").read_bytes()
        cases = (
            ("left", 0x10, 0x55),
            ("right", 0x20, 0x5D),
        )
        for side, trigger_mask, trigger_tag in cases:
            with self.subTest(side=side):
                node = self._node_without_ros()
                pressed = (
                    b"\x30"
                    + bytes((trigger_mask, trigger_tag))
                    + struct.pack("<f", 1.0)
                    + base
                )
                held_without_analog = b"\x30" + bytes((trigger_mask,)) + base

                pressed_action = node._sec_controller_command(
                    side, decode_sec_dev_pose(pressed)
                )
                held_action = node._sec_controller_command(
                    side, decode_sec_dev_pose(held_without_analog)
                )
                released_action = node._sec_controller_command(
                    side, decode_sec_dev_pose(base)
                )

                self.assertEqual(pressed_action.gripper, 1.0)
                self.assertEqual(held_action.gripper, 1.0)
                self.assertEqual(released_action.gripper, 0.0)

    def test_sec_dev_a_uses_rising_edges_and_allows_retry(self):
        class Publisher:
            def __init__(self):
                self.values = []

            def publish(self, message):
                self.values.append(message.data)

        node = self._node_without_ros()
        node._last_input_mask = None
        node.mode_publisher = Publisher()
        node.String = lambda data: type("Message", (), {"data": data})()

        node._publish_sec_mode_edges(0)
        node._publish_sec_mode_edges(0x04)
        node._publish_sec_mode_edges(0x04)
        node._publish_sec_mode_edges(0)
        node._publish_sec_mode_edges(0x04)
        node._publish_sec_mode_edges(0)
        node._publish_sec_mode_edges(0x08)
        self.assertEqual(
            node.mode_publisher.values, ["teleop", "teleop", "pause"]
        )


class ActionChunkTest(unittest.TestCase):
    def test_round_trip_normalizes_and_preserves_bimanual_action(self):
        action = ActionFrame(
            left=EndpointCommand((0.1, 0.2, -0.3), (0.0, 0.0, 0.0, 2.0), 0.2),
            right=EndpointCommand((0.1, -0.2, -0.3), (0.0, 0.0, 0.0, 1.0), 0.3),
        )
        frames = {
            "left": "left_shoulder_base",
            "right": "right_shoulder_base",
        }
        payload = encode_action_chunk(
            source="policy",
            sequence=7,
            control_hz=60.0,
            frame_ids=frames,
            actions=[action],
            created_at_ns=123,
        )
        chunk = decode_action_chunk(
            payload,
            expected_source="policy",
            expected_frame_ids=frames,
        )
        self.assertEqual(chunk.sequence, 7)
        self.assertEqual(chunk.created_at_ns, 123)
        self.assertEqual(len(chunk.actions), 1)
        np.testing.assert_allclose(
            chunk.actions[0].left.quaternion, (0.0, 0.0, 0.0, 1.0)
        )

    def test_wrong_source_and_oversized_chunk_are_rejected(self):
        action = ActionFrame(
            left=EndpointCommand((0, 0, 0), (0, 0, 0, 1), 0.1),
            right=EndpointCommand((0, 0, 0), (0, 0, 0, 1), 0.1),
        )
        frames = {"left": "left", "right": "right"}
        payload = encode_action_chunk(
            source="teleop",
            sequence=1,
            control_hz=60.0,
            frame_ids=frames,
            actions=[action, action],
        )
        with self.assertRaises(ValueError):
            decode_action_chunk(payload, expected_source="policy")
        with self.assertRaises(ValueError):
            decode_action_chunk(payload, max_actions=1)


class FourTopicBufferTest(unittest.TestCase):
    def test_requires_every_topic_to_update_before_each_output(self):
        buffer = FourTopicBuffer(sync_slop=0.05)
        for index, name in enumerate(FourTopicBuffer.NAMES):
            buffer.update(name, name, received_at=1.0 + index * 0.005)
        values, spread = buffer.take()
        self.assertEqual(set(values), set(FourTopicBuffer.NAMES))
        self.assertAlmostEqual(spread, 0.015)

        buffer.update("left_pose", "new", received_at=2.0)
        self.assertEqual(buffer.take(), (None, None))
        for name in FourTopicBuffer.NAMES[1:]:
            buffer.update(name, "new", received_at=2.0)
        values, _ = buffer.take()
        self.assertEqual(values["left_pose"], "new")

    def test_drops_complete_set_outside_sync_window(self):
        buffer = FourTopicBuffer(sync_slop=0.05)
        for name in FourTopicBuffer.NAMES:
            buffer.update(name, name, received_at=1.0)
        buffer.update("right_gripper", "late", received_at=1.10)
        values, spread = buffer.take()
        self.assertIsNone(values)
        self.assertAlmostEqual(spread, 0.10)
        self.assertEqual(buffer.take(), (None, None))

    def test_adapter_update_attempts_immediate_publish(self):
        adapter = PolicyTopicAdapter.__new__(PolicyTopicAdapter)
        adapter.input_buffer = FourTopicBuffer(sync_slop=0.05)
        publish_attempts = []
        adapter._try_publish = lambda: publish_attempts.append(True)

        adapter._update("left_pose", "value")

        self.assertEqual(publish_attempts, [True])


class SupervisorQueueTest(unittest.TestCase):
    @staticmethod
    def _action(x: float) -> ActionFrame:
        return ActionFrame(
            left=EndpointCommand((x, 0.2, 0.1), (0, 0, 0, 1), 0.1),
            right=EndpointCommand((x, -0.2, 0.1), (0, 0, 0, 1), 0.1),
        )

    def _supervisor_without_ros(self) -> HILSupervisor:
        supervisor = HILSupervisor.__new__(HILSupervisor)
        supervisor.args = SimpleNamespace(
            max_chunk_actions=240,
            max_source_jump=0.15,
            max_source_rotation_jump=1.0,
            max_arm_reach=0.55,
            dry_run=False,
        )
        supervisor.lock = threading.RLock()
        supervisor.sources = {"policy": SourceState(), "teleop": SourceState()}
        supervisor.selected_mode = "pause"
        return supervisor

    def _send(self, supervisor, source: str, sequence: int, actions) -> None:
        payload = encode_action_chunk(
            source=source,
            session_id=f"test-{source}",
            sequence=sequence,
            control_hz=60.0,
            frame_ids={
                "left": (
                    "hil_teleop_left_relative"
                    if source == "teleop"
                    else "left_shoulder_base"
                ),
                "right": (
                    "hil_teleop_right_relative"
                    if source == "teleop"
                    else "right_shoulder_base"
                ),
            },
            actions=actions,
        )
        supervisor._on_source_chunk(source, SimpleNamespace(data=payload))

    def test_teleop_keeps_only_latest_sample(self):
        supervisor = self._supervisor_without_ros()
        self._send(supervisor, "teleop", 1, [self._action(0.30)])
        self._send(supervisor, "teleop", 2, [self._action(0.31)])
        latest_action = supervisor.sources["teleop"].latest_action
        self.assertIsNotNone(latest_action)
        self.assertAlmostEqual(latest_action.left.position[0], 0.31)
        self.assertEqual(supervisor.sources["teleop"].superseded_actions, 1)

    def test_teleop_new_session_skips_cross_session_jump_check(self):
        supervisor = self._supervisor_without_ros()
        frames = {
            "left": "hil_teleop_left_relative",
            "right": "hil_teleop_right_relative",
        }
        for session_id, sequence, x in (("old", 10, 0.0), ("new", 0, 0.40)):
            payload = encode_action_chunk(
                source="teleop",
                session_id=session_id,
                sequence=sequence,
                control_hz=60.0,
                frame_ids=frames,
                actions=[self._action(x)],
            )
            supervisor._on_source_chunk("teleop", SimpleNamespace(data=payload))

        state = supervisor.sources["teleop"]
        self.assertEqual(state.latest_chunk.session_id, "new")
        self.assertEqual(state.latest_sequence, 0)
        self.assertAlmostEqual(state.latest_action.left.position[0], 0.40)
        self.assertEqual(state.invalid_reason, "")

    def test_teleop_control_loop_keeps_moving_toward_latest_target(self):
        supervisor = self._supervisor_without_ros()
        supervisor.args.dry_run = True
        supervisor.args.control_hz = 60.0
        supervisor.args.teleop_position_scale = 1.0
        supervisor.args.max_teleop_displacement = 0.5
        supervisor.args.max_translation_speed = 0.6
        supervisor.args.max_rotation_speed = 1.2
        supervisor.args.max_gripper_rate = 2.0
        supervisor.args.max_gripper_release_rate = 10.0
        supervisor.selected_mode = "teleop"
        supervisor.phase = "active"
        supervisor._source_ready = lambda source, now: (True, "ok")
        supervisor.last_outputs = {}
        supervisor.last_grippers = {"left": 0.1, "right": 0.1}
        supervisor.robot_poses = {}
        supervisor.teleop_raw_anchors = {}
        supervisor.teleop_robot_anchors = {}
        anchor_action = self._action(0.0)
        for side in ("left", "right"):
            raw = anchor_action.command(side)
            supervisor.teleop_raw_anchors[side] = PoseArray(
                raw.position.copy(), raw.quaternion.copy()
            )
            zero = PoseArray(np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0)))
            supervisor.teleop_robot_anchors[side] = zero
            supervisor.last_outputs[side] = PoseArray(
                zero.position.copy(), zero.quaternion.copy()
            )
        supervisor.sources["teleop"].latest_action = self._action(0.30)

        supervisor._publish_outputs_locked(1.0)
        first_x = supervisor.last_outputs["left"].position[0]
        supervisor._publish_outputs_locked(1.01)
        second_x = supervisor.last_outputs["left"].position[0]

        self.assertAlmostEqual(first_x, 0.01)
        self.assertAlmostEqual(second_x, 0.02)

    def test_active_policy_forwards_each_single_action_without_queueing(self):
        supervisor = self._supervisor_without_ros()
        supervisor.selected_mode = "policy"
        supervisor.phase = "active"
        supervisor.fsm_state = SimpleNamespace(value=2)
        supervisor._runtime_ready = lambda source, now: (True, "ok")
        forwarded = []
        supervisor._publish_policy_action_locked = forwarded.append

        self._send(supervisor, "policy", 1, [self._action(0.30)])
        self._send(supervisor, "policy", 2, [self._action(0.32)])

        self.assertIsNone(supervisor.sources["policy"].latest_action)
        self.assertEqual(
            [round(action.left.position[0], 2) for action in forwarded],
            [0.30, 0.32],
        )

    def test_policy_handover_updates_target_without_direct_forwarding(self):
        supervisor = self._supervisor_without_ros()
        supervisor.selected_mode = "policy"
        supervisor.phase = "policy_handover"
        forwarded = []
        supervisor._publish_policy_action_locked = forwarded.append

        self._send(supervisor, "policy", 1, [self._action(0.30)])
        self._send(supervisor, "policy", 2, [self._action(0.32)])

        self.assertEqual(forwarded, [])
        self.assertAlmostEqual(
            supervisor.sources["policy"].latest_chunk.actions[0].left.position[0],
            0.32,
        )

    def test_policy_action_is_forwarded_without_motion_shaping(self):
        supervisor = self._supervisor_without_ros()
        supervisor.args.dry_run = True
        supervisor.last_outputs = {"left": None, "right": None}
        supervisor.last_grippers = {"left": 0.10, "right": 0.10}
        action = self._action(1.20)

        supervisor._publish_policy_action_locked(action)

        self.assertAlmostEqual(supervisor.last_outputs["left"].position[0], 1.20)
        self.assertAlmostEqual(supervisor.last_outputs["right"].position[0], 1.20)
        self.assertAlmostEqual(supervisor.last_grippers["left"], 0.10)

    def test_multi_action_policy_chunk_is_rejected(self):
        supervisor = self._supervisor_without_ros()
        self._send(supervisor, "policy", 1, [self._action(0.10), self._action(0.20)])

        state = supervisor.sources["policy"]
        self.assertIsNone(state.latest_action)
        self.assertEqual(state.accepted_chunks, 0)
        self.assertEqual(state.rejected_chunks, 1)
        self.assertIn("exactly one action", state.last_rejection_reason)

    def test_multi_action_teleop_chunk_is_rejected(self):
        supervisor = self._supervisor_without_ros()
        self._send(supervisor, "teleop", 1, [self._action(0.10), self._action(0.20)])

        state = supervisor.sources["teleop"]
        self.assertIsNone(state.latest_action)
        self.assertEqual(state.accepted_chunks, 0)
        self.assertEqual(state.rejected_chunks, 1)
        self.assertIn("exactly one action", state.last_rejection_reason)

    def test_dry_run_does_not_wait_for_unsent_fsm_transitions(self):
        supervisor = HILSupervisor.__new__(HILSupervisor)
        supervisor.args = SimpleNamespace(dry_run=True)
        supervisor.fsm_state = SimpleNamespace(value=99)

        self.assertTrue(supervisor._fsm_transition_reached(0))
        self.assertTrue(supervisor._fsm_transition_reached(2))

        supervisor.args.dry_run = False
        self.assertFalse(supervisor._fsm_transition_reached(2))
        supervisor.fsm_state.value = 2
        self.assertTrue(supervisor._fsm_transition_reached(2))

    def test_policy_chunk_needs_no_hil_status_metadata(self):
        supervisor = self._supervisor_without_ros()
        payload = encode_action_chunk(
            source="policy",
            session_id="test-policy",
            sequence=1,
            control_hz=60.0,
            frame_ids={
                "left": "left_shoulder_base",
                "right": "right_shoulder_base",
            },
            actions=[self._action(0.30)],
        )
        supervisor._on_source_chunk("policy", SimpleNamespace(data=payload))
        state = supervisor.sources["policy"]
        self.assertIsNotNone(state.latest_chunk)
        self.assertEqual(state.accepted_chunks, 1)
        self.assertEqual(state.rejected_chunks, 0)
        self.assertEqual(state.invalid_reason, "")


class SupervisorPolicyRequestTest(unittest.TestCase):
    @staticmethod
    def _policy_chunk(x: float = 0.30):
        action = SupervisorQueueTest._action(x)
        return decode_action_chunk(
            encode_action_chunk(
                source="policy",
                session_id="external-policy",
                sequence=1,
                control_hz=15.0,
                frame_ids={
                    "left": "left_shoulder_base",
                    "right": "right_shoulder_base",
                },
                actions=[action],
            )
        )

    def test_policy_request_clears_old_input_and_waits_for_next_chunk(self):
        supervisor = HILSupervisor.__new__(HILSupervisor)
        supervisor.lock = threading.RLock()
        supervisor.sources = {"policy": SourceState(), "teleop": SourceState()}
        old_state = supervisor.sources["policy"]
        old_state.latest_action = SupervisorQueueTest._action(0.30)
        old_state.latest_chunk = self._policy_chunk()
        old_state.chunk_received_at = 1.0
        supervisor.policy_request_received_at = 0.0
        supervisor.selected_mode = "teleop"
        supervisor.pending_mode = None
        supervisor.phase = "active"
        supervisor.phase_started_at = 0.0
        supervisor.last_reason = "teleop active"
        supervisor._base_ready = lambda now: (True, "ok")
        supervisor._publish_enable_locked = lambda value: None

        supervisor._on_mode_request(SimpleNamespace(data="policy"))

        self.assertEqual(supervisor.pending_mode, "policy")
        self.assertEqual(supervisor.selected_mode, "pause")
        self.assertEqual(supervisor.phase, "waiting_for_pause")
        self.assertIsNone(supervisor.sources["policy"].latest_action)
        self.assertIsNone(supervisor.sources["policy"].latest_chunk)
        self.assertGreater(supervisor.policy_request_received_at, 0.0)

    def test_handover_only_accepts_action_received_after_request(self):
        supervisor = HILSupervisor.__new__(HILSupervisor)
        supervisor.args = SimpleNamespace(
            policy_handover_position_limit=0.03,
            policy_handover_rotation_limit=0.262,
        )
        supervisor.sources = {"policy": SourceState(), "teleop": SourceState()}
        supervisor.policy_request_received_at = 10.0
        state = supervisor.sources["policy"]
        state.latest_chunk = self._policy_chunk(0.30)
        action = state.latest_chunk.actions[0]
        supervisor.robot_poses = {
            side: SimpleNamespace(value=PoseArray(
                action.command(side).position.copy(),
                action.command(side).quaternion.copy(),
            ))
            for side in ("left", "right")
        }

        state.chunk_received_at = 9.9
        valid, reason = supervisor._policy_handover_ready()
        self.assertFalse(valid)
        self.assertIn("predates", reason)

        state.chunk_received_at = 10.1
        valid, reason = supervisor._policy_handover_ready()
        self.assertTrue(valid, reason)

    def test_handover_accepts_pose_discontinuity_for_smooth_transition(self):
        supervisor = HILSupervisor.__new__(HILSupervisor)
        supervisor.sources = {"policy": SourceState(), "teleop": SourceState()}
        supervisor.policy_request_received_at = 10.0
        state = supervisor.sources["policy"]
        state.latest_chunk = self._policy_chunk(0.40)
        state.chunk_received_at = 10.1

        valid, reason = supervisor._policy_handover_ready()

        self.assertTrue(valid, reason)

    def test_handover_target_reports_pose_discontinuity(self):
        action = SupervisorQueueTest._action(0.40)
        robot_poses = {
            side: PoseArray(
                SupervisorQueueTest._action(0.30).command(side).position.copy(),
                np.array((0.0, 0.0, 0.0, 1.0)),
            )
            for side in ("left", "right")
        }

        reached, reason = policy_handover_target_reached(
            action,
            robot_poses,
            position_limit=0.03,
            rotation_limit=0.262,
        )

        self.assertFalse(reached)
        self.assertIn("0.100 m", reason)

    def test_handover_moves_toward_latest_policy_target_at_bounded_speed(self):
        supervisor = HILSupervisor.__new__(HILSupervisor)
        supervisor.args = SimpleNamespace(
            control_hz=60.0,
            max_translation_speed=0.6,
            max_rotation_speed=1.2,
            max_gripper_rate=2.0,
            max_gripper_release_rate=10.0,
            policy_handover_position_limit=0.01,
            policy_handover_rotation_limit=0.087,
            dry_run=True,
        )
        supervisor.last_outputs = {
            side: PoseArray(
                np.array((0.0, 0.2 if side == "left" else -0.2, 0.1)),
                np.array((0.0, 0.0, 0.0, 1.0)),
            )
            for side in ("left", "right")
        }
        supervisor.robot_poses = {
            side: SimpleNamespace(value=supervisor.last_outputs[side])
            for side in ("left", "right")
        }
        supervisor.last_grippers = {"left": 0.1, "right": 0.1}

        reached = supervisor._publish_policy_handover_locked(
            SupervisorQueueTest._action(0.30)
        )

        self.assertFalse(reached)
        self.assertAlmostEqual(supervisor.last_outputs["left"].position[0], 0.01)
        self.assertAlmostEqual(supervisor.last_outputs["right"].position[0], 0.01)

        for _ in range(30):
            reached = supervisor._publish_policy_handover_locked(
                SupervisorQueueTest._action(0.30)
            )

        self.assertTrue(reached)
        self.assertAlmostEqual(supervisor.last_outputs["left"].position[0], 0.30)


class SupervisorShutdownTest(unittest.TestCase):
    def test_invalid_ros_context_skips_shutdown_publish(self):
        destroyed = []
        supervisor = HILSupervisor.__new__(HILSupervisor)
        supervisor.rclpy = SimpleNamespace(ok=lambda: False)
        supervisor.node = SimpleNamespace(
            destroy_node=lambda: destroyed.append(True)
        )

        supervisor.close()

        self.assertEqual(destroyed, [True])


if __name__ == "__main__":
    unittest.main()
