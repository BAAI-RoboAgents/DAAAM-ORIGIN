#!/usr/bin/env python3
"""Serve the local DAAAM mapping workflow dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.dashboard import create_dashboard_app  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the local DAAAM mapping control and observability UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "output",
        help="Only first-level run directories below this path are exposed.",
    )
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: the dashboard can start local mapping subprocesses and has "
            "no authentication; bind beyond loopback only on a trusted network.",
            file=sys.stderr,
        )
    app = create_dashboard_app(
        REPOSITORY_ROOT,
        output_root=args.output_root,
    )
    print(
        f"DAAAM mapping dashboard: http://{args.host}:{args.port} "
        f"(output root: {args.output_root.resolve()})"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
