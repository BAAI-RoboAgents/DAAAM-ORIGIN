#!/usr/bin/env python3
"""Serve the standalone DAAAM semantic-map query interface."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.semantic_query_ui import create_semantic_query_app  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the independent DAAAM semantic-map query UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "output",
        help="Only semantic-map directories below this root may be loaded.",
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
    output_root = args.output_root.expanduser().resolve()
    if not output_root.is_dir():
        raise SystemExit(f"--output-root does not exist: {output_root}")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: the semantic query UI has no authentication; bind beyond "
            "loopback only on a trusted network.",
            file=sys.stderr,
        )
    app = create_semantic_query_app(output_root)
    print(
        f"DAAAM semantic query UI: http://{args.host}:{args.port} "
        f"(output root: {output_root})"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
