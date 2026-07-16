#!/usr/bin/env python3
"""Inspect and edit the persistent semantic map memory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.memory import MapMemory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query and edit DAAAM's versioned map memory."
    )
    parser.add_argument("--database", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--include-deleted", action="store_true")

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("entity_id")

    find_parser = subparsers.add_parser("find")
    find_parser.add_argument("name")

    name_parser = subparsers.add_parser("name")
    name_parser.add_argument("entity_id")
    name_parser.add_argument("canonical_name")
    name_parser.add_argument("--alias", action="append", default=[])
    name_parser.add_argument("--unlocked", action="store_true")
    name_parser.add_argument("--sensor-time-ns", type=int)

    alias_parser = subparsers.add_parser("alias")
    alias_parser.add_argument("entity_id")
    alias_parser.add_argument("alias")
    alias_parser.add_argument("--sensor-time-ns", type=int)

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("entity_id")
    delete_parser.add_argument("--sensor-time-ns", type=int)

    revision_parser = subparsers.add_parser("revision")
    revision_parser.add_argument("reason")
    revision_parser.add_argument("--sensor-time-ns", type=int)

    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("target_revision", type=int)
    rollback_parser.add_argument("--sensor-time-ns", type=int)

    create_session = subparsers.add_parser("create-session")
    create_session.add_argument("session_id")
    create_session.add_argument("--started-ns", type=int)
    create_session.add_argument("--canonical", action="store_true")

    register_session = subparsers.add_parser("register-session")
    register_session.add_argument("session_id")
    register_session.add_argument("--transform-json", type=Path, required=True)
    register_session.add_argument("--covariance-json", type=Path, required=True)
    register_session.add_argument("--inliers", type=int, required=True)
    register_session.add_argument("--rms-error-m", type=float, required=True)

    subparsers.add_parser("sessions")
    subparsers.add_parser("stats")
    return parser.parse_args()


def _timestamp(value: int | None) -> int:
    timestamp = time.time_ns() if value is None else int(value)
    if timestamp <= 0:
        raise ValueError("sensor time must be a positive absolute nanosecond value")
    return timestamp


def _read_array(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(json.loads(path.read_text()), dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f"{path} must contain an array with shape {shape}")
    return value


def execute(args: argparse.Namespace) -> dict | list:
    with MapMemory(args.database.expanduser().resolve()) as memory:
        if args.command == "list":
            return memory.list_entities(include_deleted=args.include_deleted)
        if args.command == "show":
            return memory.get_entity(args.entity_id)
        if args.command == "find":
            return memory.find_by_name(args.name)
        if args.command == "name":
            memory.set_user_name(
                args.entity_id,
                args.canonical_name,
                sensor_time_ns=_timestamp(args.sensor_time_ns),
                aliases=args.alias,
                lock=not args.unlocked,
            )
            return memory.get_entity(args.entity_id)
        if args.command == "alias":
            memory.add_user_alias(
                args.entity_id,
                args.alias,
                sensor_time_ns=_timestamp(args.sensor_time_ns),
            )
            return memory.get_entity(args.entity_id)
        if args.command == "delete":
            memory.delete_entity(
                args.entity_id,
                sensor_time_ns=_timestamp(args.sensor_time_ns),
            )
            return memory.get_entity(args.entity_id)
        if args.command == "revision":
            revision = memory.advance_revision(
                args.reason,
                _timestamp(args.sensor_time_ns),
            )
            return {"current_revision": revision}
        if args.command == "rollback":
            revision = memory.rollback_to_revision(
                args.target_revision,
                sensor_time_ns=_timestamp(args.sensor_time_ns),
            )
            return {
                "rolled_back_to": args.target_revision,
                "current_revision": revision,
            }
        if args.command == "create-session":
            memory.create_session(
                args.session_id,
                _timestamp(args.started_ns),
                canonical=args.canonical,
            )
            return next(
                item
                for item in memory.list_sessions()
                if item["session_id"] == args.session_id
            )
        if args.command == "register-session":
            registration = memory.register_session(
                args.session_id,
                _read_array(args.transform_json, (4, 4)),
                _read_array(args.covariance_json, (6, 6)),
                inlier_count=args.inliers,
                rms_error_m=args.rms_error_m,
            )
            return asdict(registration)
        if args.command == "sessions":
            return memory.list_sessions()
        if args.command == "stats":
            return memory.stats()
    raise ValueError(f"unsupported command: {args.command}")


def main() -> int:
    args = parse_args()
    try:
        result = execute(args)
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "error": str(error)},
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {"status": "ok", "result": result},
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
