#!/usr/bin/env python3
"""Compute connected-component quality evidence for a Hydra PLY mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.quality import analyze_ascii_ply_mesh  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weld-tolerance-m", type=float, default=1.0e-4)
    args = parser.parse_args()
    metrics = analyze_ascii_ply_mesh(
        args.mesh,
        weld_tolerance_m=args.weld_tolerance_m,
    )
    payload = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
