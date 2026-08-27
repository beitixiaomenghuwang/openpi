#!/usr/bin/env python3
"""Read-only field sampler for the second-S100 Zenoh XR pose stream.

This tool deliberately does not import ROS and never publishes commands.  It
opens a Zenoh peer with automatic discovery, subscribes to
``sec/xr/devicepose``, decodes only the protobuf wire envelope, and records
field-level changes.  Field numbers are reported without assigning semantic
names; controller/button meanings must be established by controlled motion
captures.

Examples::

    # Capture one 10-second segment using multicast/peer auto-discovery.
    python3 -m hil.zenoh_pose_sampler --seconds 10 --output /tmp/xr.jsonl

    # Capture the standard one-variable mapping phases interactively.
    python3 -m hil.zenoh_pose_sampler --interactive --seconds 5 \
        --output /tmp/xr_mapping.jsonl --raw-dir /tmp/xr_mapping_raw

An optional ``--endpoint`` can be used when peer discovery is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import queue
import struct
import sys
import threading
import time
from typing import Any

try:
    from .zenoh_wire import WireDecodeError, WireField, parse_wire_fields
except ImportError:
    from zenoh_wire import WireDecodeError, WireField, parse_wire_fields


DEFAULT_KEY = "sec/xr/devicepose"
DEFAULT_PHASES = (
    "baseline",
    "head_only",
    "right_move",
    "left_move",
    "right_trigger",
    "left_trigger",
    "right_squeeze",
    "left_squeeze",
    "right_A",
    "right_B",
    "left_X",
    "left_Y",
)


def _finite_float_list(raw: bytes) -> list[float] | None:
    if len(raw) == 0 or len(raw) % 4:
        return None
    values = list(struct.unpack("<" + "f" * (len(raw) // 4), raw))
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def field_to_json(field: WireField) -> dict[str, Any]:
    """Return a compact, JSON-safe representation suitable for diffing."""
    item: dict[str, Any] = {
        "wire_type": field.wire_type,
        "byte_length": len(field.raw),
        "raw_sha256": hashlib.sha256(field.raw).hexdigest(),
    }
    if field.wire_type == 0:
        item["varint"] = int(field.value)
    elif field.wire_type == 1:
        item["float64"] = float(field.value)
    elif field.wire_type == 5:
        item["float32"] = float(field.value)
    elif field.wire_type == 2:
        floats = _finite_float_list(field.raw)
        if floats is not None and len(floats) <= 16:
            item["float32_values"] = floats
        elif floats is not None:
            item["float32_count"] = len(floats)
            item["float32_first"] = floats[:4]
            item["float32_last"] = floats[-4:]
        if len(field.raw) <= 32:
            item["raw_hex"] = field.raw.hex()
    return item


def fields_to_json(fields: list[WireField]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        grouped.setdefault(str(field.number), []).append(field_to_json(field))
    return grouped


def _field_values(fields: list[WireField]) -> dict[int, list[WireField]]:
    grouped: dict[int, list[WireField]] = {}
    for field in fields:
        grouped.setdefault(field.number, []).append(field)
    return grouped


def _max_float_delta(left: bytes, right: bytes) -> float | None:
    left_values = _finite_float_list(left)
    right_values = _finite_float_list(right)
    if left_values is None or right_values is None or len(left_values) != len(right_values):
        return None
    return max((abs(a - b) for a, b in zip(left_values, right_values)), default=0.0)


def summarize_activity(samples: list[list[WireField]]) -> dict[str, Any]:
    """Summarize per-field activity across one capture phase."""
    if not samples:
        return {"sample_count": 0, "fields": {}}

    by_field = [_field_values(fields) for fields in samples]
    all_numbers = sorted({number for grouped in by_field for number in grouped})
    result: dict[str, Any] = {}
    for number in all_numbers:
        occurrences = max(len(grouped.get(number, [])) for grouped in by_field)
        occurrence_results = []
        for occurrence in range(occurrences):
            values = [grouped.get(number, [])[occurrence] if len(grouped.get(number, [])) > occurrence else None for grouped in by_field]
            present = [value for value in values if value is not None]
            changes = 0
            max_delta = 0.0
            have_delta = False
            for previous, current in zip(values, values[1:]):
                if previous is None or current is None:
                    continue
                if previous.raw != current.raw:
                    changes += 1
                delta = _max_float_delta(previous.raw, current.raw)
                if delta is not None:
                    max_delta = max(max_delta, delta)
                    have_delta = True
            item: dict[str, Any] = {
                "present_samples": len(present),
                "changes_between_samples": changes,
                "first": field_to_json(present[0]) if present else None,
                "last": field_to_json(present[-1]) if present else None,
            }
            if have_delta:
                item["max_abs_float_delta"] = max_delta
            occurrence_results.append(item)
        result[str(number)] = occurrence_results
    return {"sample_count": len(samples), "fields": result}


def _payload_bytes(payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if hasattr(payload, "to_bytes"):
        return payload.to_bytes()
    return bytes(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--endpoint", default="", help="Optional Zenoh connect endpoint")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("xr_pose_capture.jsonl"))
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Optional directory for one raw .bin file per received sample",
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--phases",
        default=",".join(DEFAULT_PHASES),
        help="Comma-separated phase names used with --interactive",
    )
    parser.add_argument("--log-every", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.seconds <= 0.0:
        parser.error("--seconds must be greater than zero")
    if args.log_every <= 0.0:
        parser.error("--log-every must be greater than zero")
    args.phase_names = tuple(name.strip() for name in args.phases.split(",") if name.strip())
    if args.interactive and not args.phase_names:
        parser.error("--phases must contain at least one phase")
    return args


def _write_jsonl(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
    handle.flush()


def _capture_phase(
    sample_queue: queue.Queue[tuple[int, bytes]],
    phase: str,
    seconds: float,
    output,
    raw_dir: Path | None,
    sample_sequence: int,
    log_every: float,
) -> tuple[int, dict[str, Any]]:
    # Samples can arrive while the operator is preparing the next phase. Do
    # not attribute those stale packets to the newly-labelled action.
    while True:
        try:
            sample_queue.get_nowait()
        except queue.Empty:
            break
    started_ns = time.time_ns()
    deadline = time.monotonic() + seconds
    fields_for_summary: list[list[WireField]] = []
    records = 0
    rejected = 0
    next_log = time.monotonic() + log_every
    raw_phase_dir = None
    if raw_dir is not None:
        raw_phase_dir = raw_dir / phase
        raw_phase_dir.mkdir(parents=True, exist_ok=True)

    while time.monotonic() < deadline:
        remaining = max(0.01, deadline - time.monotonic())
        try:
            received_ns, payload = sample_queue.get(timeout=min(0.2, remaining))
        except queue.Empty:
            continue
        try:
            fields = parse_wire_fields(payload)
        except WireDecodeError as exc:
            rejected += 1
            _write_jsonl(
                output,
                {
                    "type": "rejected",
                    "phase": phase,
                    "received_at_ns": received_ns,
                    "payload_bytes": len(payload),
                    "error": str(exc),
                    "payload_head_hex": payload[:32].hex(),
                },
            )
            continue

        fields_for_summary.append(fields)
        raw_file = None
        if raw_phase_dir is not None:
            raw_file = raw_phase_dir / f"{sample_sequence:06d}.bin"
            raw_file.write_bytes(payload)
        _write_jsonl(
            output,
            {
                "type": "sample",
                "phase": phase,
                "sequence": sample_sequence,
                "received_at_ns": received_ns,
                "payload_bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "raw_file": str(raw_file) if raw_file is not None else None,
                "fields": fields_to_json(fields),
            },
        )
        records += 1
        sample_sequence += 1
        if time.monotonic() >= next_log:
            print(f"[CAPTURE] phase={phase} samples={records} rejected={rejected}")
            next_log = time.monotonic() + log_every

    summary = summarize_activity(fields_for_summary)
    summary.update(
        {
            "type": "phase_summary",
            "phase": phase,
            "started_at_ns": started_ns,
            "duration_s": seconds,
            "rejected": rejected,
        }
    )
    _write_jsonl(output, summary)
    print(
        f"[DONE] phase={phase} samples={summary['sample_count']} "
        f"rejected={rejected} fields={','.join(summary['fields']) or 'none'}"
    )
    return sample_sequence, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import zenoh
    except ImportError as exc:
        print("eclipse-zenoh is required: python3 -m pip install eclipse-zenoh", file=sys.stderr)
        raise SystemExit(2) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.raw_dir is not None:
        args.raw_dir.mkdir(parents=True, exist_ok=True)

    config = zenoh.Config()
    if args.endpoint:
        config.insert_json5("connect/endpoints", json.dumps([args.endpoint]))

    sample_queue: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=2000)
    dropped = 0
    dropped_lock = threading.Lock()

    session = zenoh.open(config)
    print(
        f"[OK] zenoh session opened ({args.endpoint or 'peer, multicast auto-discovery'})"
    )

    def listener(sample) -> None:
        nonlocal dropped
        try:
            item = (time.time_ns(), _payload_bytes(sample.payload))
            try:
                sample_queue.put_nowait(item)
            except queue.Full:
                try:
                    sample_queue.get_nowait()
                    with dropped_lock:
                        dropped += 1
                except queue.Empty:
                    pass
                try:
                    sample_queue.put_nowait(item)
                except queue.Full:
                    with dropped_lock:
                        dropped += 1
        except Exception as exc:
            print(f"[WARN] failed to enqueue Zenoh sample: {exc}", file=sys.stderr)

    subscriber = session.declare_subscriber(args.key, listener)
    print(f"[SUB] listening on '{args.key}'")
    print("[INFO] read-only sampler: no ROS node and no robot command publisher")

    summaries: list[dict[str, Any]] = []
    sample_sequence = 0
    try:
        with args.output.open("w", encoding="utf-8") as output:
            _write_jsonl(
                output,
                {
                    "type": "metadata",
                    "key": args.key,
                    "endpoint": args.endpoint or "auto-discovery",
                    "format": "protobuf-like top-level wire inspection",
                    "semantic_mapping": "intentionally_not_assigned",
                    "started_at_ns": time.time_ns(),
                },
            )
            if args.interactive:
                for phase in args.phase_names:
                    input(f"\n准备阶段 '{phase}'，完成动作后按 Enter 开始采集 {args.seconds:g}s ... ")
                    sample_sequence, summary = _capture_phase(
                        sample_queue,
                        phase,
                        args.seconds,
                        output,
                        args.raw_dir,
                        sample_sequence,
                        args.log_every,
                    )
                    summaries.append(summary)
            else:
                sample_sequence, summary = _capture_phase(
                    sample_queue,
                    "capture",
                    args.seconds,
                    output,
                    args.raw_dir,
                    sample_sequence,
                    args.log_every,
                )
                summaries.append(summary)

            with dropped_lock:
                dropped_total = dropped
            _write_jsonl(
                output,
                {
                    "type": "capture_summary",
                    "key": args.key,
                    "phases": [summary["phase"] for summary in summaries],
                    "total_samples": sum(summary["sample_count"] for summary in summaries),
                    "queue_dropped": dropped_total,
                },
            )
    except KeyboardInterrupt:
        print("\n[STOP] interrupted; partial JSONL capture is valid")
    finally:
        try:
            subscriber.undeclare()
        except Exception as exc:
            print(f"[WARN] subscriber cleanup failed: {exc}", file=sys.stderr)
        try:
            session.close()
        except Exception as exc:
            print(f"[WARN] Zenoh session cleanup failed: {exc}", file=sys.stderr)

    print(f"[OUT] wrote {args.output}")
    if args.raw_dir is not None:
        print(f"[RAW] wrote raw payloads under {args.raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
