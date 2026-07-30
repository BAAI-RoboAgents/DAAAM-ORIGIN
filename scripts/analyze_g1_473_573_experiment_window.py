#!/usr/bin/env python3
"""Create a visual, reproducible audit of G1 source frames 473-573."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


DEFAULT_RAW = Path("/home/user/datasets/g1_20260724")
DEFAULT_PREPARED = Path("output/g1_20260724_v1_v2_rectified_prepared")
DEFAULT_GEOMETRY = Path("output/g1_20260724_v1_v2_geometry")
DEFAULT_SEMANTIC = Path("output/g1_20260724_v1_v2_semantic_map")
DEFAULT_OUTPUT = Path("docs/assets/g1_semantic_map_experiments_v1")
SPLITS = {
    "calibration": (473, 487),
    "development": (488, 527),
    "stress": (528, 557),
    "held-out": (558, 573),
}
SPLIT_COLORS = {
    "calibration": "#dbeafe",
    "development": "#dcfce7",
    "stress": "#ffedd5",
    "held-out": "#fee2e2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--semantic", type=Path, default=DEFAULT_SEMANTIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=473)
    parser.add_argument("--end", type=int, default=573)
    parser.add_argument(
        "--plan-version",
        default="V1",
        help="Experiment-plan label used by the generated protocol diagrams.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def yaw_deg(quaternion_xyzw: list[float]) -> float:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    return math.degrees(
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )


def load_raw_records(raw: Path, start: int, end: int) -> list[dict[str, Any]]:
    records = []
    with (raw / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            tick = int(record["tick"])
            if start <= tick <= end:
                records.append(record)
    expected = end - start + 1
    if len(records) != expected:
        raise ValueError(f"expected {expected} raw frames, found {len(records)}")
    return records


def image_metrics(path: Path) -> tuple[float, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    brightness = float(np.mean(image))
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return brightness, sharpness


def mutual_sift_vertical_residual(
    left_path: Path, right_path: Path, *, scale: float = 0.5
) -> dict[str, Any]:
    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        return {"matches": 0, "median_px": None, "p95_px": None}
    if scale != 1.0:
        left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    detector = cv2.SIFT_create(nfeatures=1600, contrastThreshold=0.02)
    keypoints_left, descriptors_left = detector.detectAndCompute(left, None)
    keypoints_right, descriptors_right = detector.detectAndCompute(right, None)
    if descriptors_left is None or descriptors_right is None:
        return {"matches": 0, "median_px": None, "p95_px": None}
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def ratio_map(query: np.ndarray, train: np.ndarray) -> dict[int, int]:
        accepted: dict[int, int] = {}
        for pair in matcher.knnMatch(query, train, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                accepted[int(pair[0].queryIdx)] = int(pair[0].trainIdx)
        return accepted

    forward = ratio_map(descriptors_left, descriptors_right)
    reverse = ratio_map(descriptors_right, descriptors_left)
    residuals = [
        abs(
            keypoints_left[left_index].pt[1]
            - keypoints_right[right_index].pt[1]
        )
        / scale
        for left_index, right_index in forward.items()
        if reverse.get(right_index) == left_index
    ]
    if not residuals:
        return {"matches": 0, "median_px": None, "p95_px": None}
    return {
        "matches": len(residuals),
        "median_px": float(np.median(residuals)),
        "p95_px": float(np.percentile(residuals, 95)),
    }


def split_for(frame: int) -> str:
    for name, (start, end) in SPLITS.items():
        if start <= frame <= end:
            return name
    raise ValueError(f"frame {frame} is not assigned to a split")


def shade_splits(axis: Any) -> None:
    for name, (start, end) in SPLITS.items():
        axis.axvspan(
            start - 0.5,
            end + 0.5,
            color=SPLIT_COLORS[name],
            alpha=0.55,
            linewidth=0,
        )


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def fit_panel(image: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def depth_color(depth_raw: np.ndarray, maximum_m: float = 8.0) -> np.ndarray:
    depth_m = depth_raw.astype(np.float32) / 1000.0
    valid = (depth_m > 0.0) & np.isfinite(depth_m)
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        255.0 * (1.0 - depth_m[valid] / maximum_m), 0, 255
    ).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def semantic_overlay(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)
    maximum = int(labels.max(initial=0))
    palette = rng.integers(32, 255, size=(maximum + 1, 3), dtype=np.uint8)
    palette[0] = 0
    colors = palette[labels]
    output = rgb.copy()
    mask = labels > 0
    output[mask] = (
        0.48 * output[mask].astype(np.float32)
        + 0.52 * colors[mask].astype(np.float32)
    ).astype(np.uint8)
    return output


def plot_overview(raw: Path, output: Path) -> None:
    frames = [473, 487, 488, 527, 528, 557, 558, 573]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.4), constrained_layout=True)
    for axis, frame in zip(axes.flat, frames):
        path = raw / "2d_rect/cam0/000000" / f"{frame:06d}.png"
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        axis.imshow(image)
        split = split_for(frame)
        axis.set_title(
            f"{split} | source frame {frame}",
            fontsize=12,
            color="#111827",
            backgroundcolor=SPLIT_COLORS[split],
        )
        axis.axis("off")
    fig.suptitle(
        "G1 473–573: one continuous close-range indoor scan (high visual overlap)",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(output / "01_window_scene_overview.png", dpi=150)
    plt.close(fig)


def plot_input_audit(rows: list[dict[str, Any]], output: Path) -> None:
    frames = np.asarray([row["source_frame"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for axis in axes.flat:
        shade_splits(axis)
        axis.grid(alpha=0.2)
        axis.set_xlim(frames.min() - 0.5, frames.max() + 0.5)
    axes[0, 0].plot(frames, [row["stereo_delta_ms"] for row in rows], color="#2563eb")
    for threshold, color in ((2.0, "#16a34a"), (5.0, "#d97706"), (10.0, "#dc2626")):
        axes[0, 0].axhline(threshold, color=color, linestyle="--", label=f"{threshold:g} ms")
    axes[0, 0].set_title("Stereo timestamp delta")
    axes[0, 0].set_ylabel("absolute cam0-cam1 delta (ms)")
    axes[0, 0].legend(ncol=3)

    axes[0, 1].plot(frames, [row["cam0_lidar_skew_ms"] for row in rows], label="cam0-LiDAR")
    axes[0, 1].plot(frames, [row["cam1_lidar_skew_ms"] for row in rows], label="cam1-LiDAR")
    axes[0, 1].axhline(60.0, color="#dc2626", linestyle="--")
    axes[0, 1].axhline(-60.0, color="#dc2626", linestyle="--")
    axes[0, 1].set_title("LiDAR-anchored camera skew")
    axes[0, 1].set_ylabel("signed skew (ms)")
    axes[0, 1].legend()

    axes[1, 0].plot(frames, [row["brightness"] for row in rows], color="#7c3aed")
    axes[1, 0].set_title("Left-image exposure stability")
    axes[1, 0].set_ylabel("mean grayscale value")

    axes[1, 1].plot(frames, [row["sharpness"] for row in rows], color="#0f766e")
    axes[1, 1].set_title("Left-image sharpness / motion-blur proxy")
    axes[1, 1].set_ylabel("variance of Laplacian")
    for axis in axes[1, :]:
        axis.set_xlabel("raw source frame")
    fig.suptitle("Input-contract audit for source frames 473–573", fontsize=18, fontweight="bold")
    fig.savefig(output / "02_input_sync_exposure_audit.png", dpi=150)
    plt.close(fig)


def plot_pose_and_epipolar(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), constrained_layout=True)
    for name, (start, end) in SPLITS.items():
        subset = [row for row in rows if start <= row["source_frame"] <= end]
        axes[0].plot(
            [row["map_x_m"] for row in subset],
            [row["map_y_m"] for row in subset],
            marker=".",
            label=f"{name} ({start}-{end})",
            color={
                "calibration": "#2563eb",
                "development": "#16a34a",
                "stress": "#ea580c",
                "held-out": "#dc2626",
            }[name],
        )
    axes[0].scatter(rows[0]["map_x_m"], rows[0]["map_y_m"], marker="s", s=70, color="black")
    axes[0].scatter(rows[-1]["map_x_m"], rows[-1]["map_y_m"], marker="*", s=100, color="black")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_title("Map-frame trajectory: a single smooth maneuver")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    frames = [row["source_frame"] for row in rows]
    shade_splits(axes[1])
    axes[1].plot(
        frames,
        [row.get("raw_epipolar_median_px", np.nan) for row in rows],
        label="raw mono-undistorted pair",
        color="#dc2626",
        alpha=0.75,
    )
    axes[1].plot(
        frames,
        [row.get("rectified_epipolar_median_px", np.nan) for row in rows],
        label="V1/V2 materialized rectification",
        color="#16a34a",
        linewidth=2,
    )
    axes[1].axhline(1.0, color="#111827", linestyle="--", label="1 px target")
    axes[1].set_title("Mutual-SIFT vertical residual (diagnostic, not ground truth)")
    axes[1].set_xlabel("raw source frame")
    axes[1].set_ylabel("median |yL-yR| (px)")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Pose diversity and local stereo geometry", fontsize=18, fontweight="bold")
    fig.savefig(output / "03_pose_and_epipolar_audit.png", dpi=150)
    plt.close(fig)


def plot_module_effects(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [row for row in rows if row.get("selected_frame") is not None]
    frames = [row["source_frame"] for row in selected]
    fig, axes = plt.subplots(4, 1, figsize=(17, 15), sharex=True, constrained_layout=True)
    for axis in axes:
        shade_splits(axis)
        axis.grid(alpha=0.22)
    axes[0].plot(frames, [row["depth_valid_ratio"] for row in selected], label="valid depth")
    axes[0].plot(frames, [row["lr_consistency"] for row in selected], label="left-right consistency")
    axes[0].plot(frames, [row["mean_confidence"] for row in selected], label="mean confidence")
    axes[0].set_ylabel("ratio")
    axes[0].set_title("Fast-FoundationStereo output")
    axes[0].legend(ncol=3)

    axes[1].plot(
        frames,
        [row.get("temporal_agreement", np.nan) for row in selected],
        color="#7c3aed",
        label="adjacent reprojection agreement",
    )
    axes[1].axhline(0.70, color="#dc2626", linestyle="--", label="current gate")
    axes[1].set_ylabel("agreement")
    axes[1].set_title("Temporal depth diagnosis")
    axes[1].legend()

    axes[2].plot(frames, [row["filter_input_ratio"] for row in selected], label="input")
    axes[2].plot(frames, [row["filter_output_ratio"] for row in selected], label="after filter")
    axes[2].fill_between(
        frames,
        [row["filter_output_ratio"] for row in selected],
        [row["filter_input_ratio"] for row in selected],
        alpha=0.25,
        color="#dc2626",
        label="removed",
    )
    axes[2].set_ylabel("valid depth ratio")
    axes[2].set_title("Multi-neighbor temporal depth filter")
    axes[2].legend(ncol=3)

    axes[3].plot(
        frames,
        [row["semantic_nonzero_ratio"] for row in selected],
        label="semantic pixel coverage",
    )
    axes[3].plot(
        frames,
        [row["dynamic_ratio"] for row in selected],
        label="dynamic mask",
    )
    axes[3].plot(
        frames,
        [row["unknown_ratio"] for row in selected],
        label="unknown mask",
    )
    axes[3].set_ylabel("pixel ratio")
    axes[3].set_title("Semantic labels and static-fusion isolation")
    axes[3].set_xlabel("raw source frame")
    axes[3].legend(ncol=3)
    fig.suptitle("Observed module effects on the existing full-run artifacts", fontsize=18, fontweight="bold")
    fig.savefig(output / "04_module_effect_timeseries.png", dpi=150)
    plt.close(fig)


def plan_filename_token(plan_version: str) -> str:
    token = plan_version.strip().lower().replace(".", "_")
    return token if token else "plan"


def plot_pipeline_flow(output: Path, plan_version: str) -> None:
    groups = [
        (
            "Geometry front-end",
            "#dbeafe",
            [
                ("E1", "sync + V1/V2\nrectification"),
                ("E2", "content-safe\nkeyframes"),
                ("E3", "stereo depth +\nLR confidence"),
                ("E4", "LiDAR depth\nscale"),
                ("E5", "temporal\ndiagnosis"),
            ],
        ),
        (
            "Trajectory / depth refinement",
            "#dcfce7",
            [
                ("E6", "local RGB-D\nodometry"),
                ("E7", "loop\nclosure"),
                ("E8", "global pose\ngraph"),
                ("E9", "temporal depth\nfilter"),
            ],
        ),
        (
            "Dynamic semantic map",
            "#ffedd5",
            [
                ("E10", "dynamic /\nunknown mask"),
                ("E11", "FastSAM"),
                ("E12", "BotSort +\nCLIP ReID"),
                ("E13", "MapMemory\nmerge"),
                ("E14", "DAM-3B\ngrounding"),
                ("E15", "semantic\nincrement"),
                ("E16", "Hydra TSDF /\nobjects"),
                ("E17", "entity↔mesh\nbinding"),
                ("E18", "exact-label\npostpass"),
                ("Q1", "multilingual\nquery utility"),
            ],
        ),
    ]
    fig, axis = plt.subplots(figsize=(31, 8))
    axis.set_xlim(0, 31)
    axis.set_ylim(0, 8)
    axis.axis("off")
    x = 0.45
    previous_right = None
    for group_name, color, modules in groups:
        width = len(modules) * 1.55 + 0.35
        axis.add_patch(
            FancyBboxPatch(
                (x - 0.15, 1.15),
                width,
                5.7,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor=color,
                edgecolor="#64748b",
                linewidth=1.5,
                alpha=0.55,
            )
        )
        axis.text(x, 6.45, group_name, fontsize=15, fontweight="bold")
        for experiment, label in modules:
            axis.add_patch(
                FancyBboxPatch(
                    (x, 2.55),
                    1.28,
                    2.55,
                    boxstyle="round,pad=0.03,rounding_size=0.08",
                    facecolor="white",
                    edgecolor="#334155",
                    linewidth=1.3,
                )
            )
            axis.text(x + 0.64, 4.65, experiment, ha="center", fontsize=13, fontweight="bold")
            axis.text(x + 0.64, 3.55, label, ha="center", va="center", fontsize=10)
            if previous_right is not None:
                axis.add_patch(
                    FancyArrowPatch(
                        (previous_right, 3.82),
                        (x, 3.82),
                        arrowstyle="-|>",
                        mutation_scale=13,
                        color="#475569",
                        linewidth=1.2,
                    )
                )
            previous_right = x + 1.28
            x += 1.55
        x += 0.25
    axis.text(
        0.45,
        0.5,
        f"{plan_version} principle: every arrow must preserve frame/time/calibration provenance; "
        "every module must expose a numeric report + representative success/failure visualization.",
        fontsize=14,
        color="#7f1d1d",
        fontweight="bold",
    )
    filename = f"05_{plan_filename_token(plan_version)}_pipeline_experiment_flow.png"
    fig.savefig(output / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostic_protocol(output: Path, plan_version: str) -> None:
    if plan_version.strip().upper() != "V1.1":
        return

    fig, axis = plt.subplots(figsize=(18, 10), constrained_layout=True)
    axis.set_xlim(0, 18)
    axis.set_ylim(0, 10)
    axis.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        body: str,
        color: str,
    ) -> tuple[float, float]:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.05,rounding_size=0.12",
                facecolor=color,
                edgecolor="#334155",
                linewidth=1.4,
            )
        )
        axis.text(
            x + width / 2,
            y + height * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        axis.text(
            x + width / 2,
            y + height * 0.32,
            body,
            ha="center",
            va="center",
            fontsize=9.5,
        )
        return x + width, y + height / 2

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                color="#475569",
                linewidth=1.4,
            )
        )

    axis.text(
        9,
        9.65,
        "V1.1 diagnostic protocol: observe, isolate, perturb, then localize",
        ha="center",
        fontsize=19,
        fontweight="bold",
    )

    frozen = box(
        0.4,
        6.7,
        2.7,
        1.7,
        "Frozen evidence",
        "101-frame input\nGT + calibration + hashes",
        "#dbeafe",
    )
    paths = [
        ("Nominal chain", "measure natural failures", "#dcfce7"),
        ("Isolated module", "oracle/frozen upstream input", "#fef3c7"),
        ("Fault injection", "known fault and expected alarm", "#fee2e2"),
        ("Interaction pair", "two-factor causal check", "#ede9fe"),
    ]
    path_centers: list[tuple[float, float]] = []
    for index, (title, body, color) in enumerate(paths):
        y = 7.65 - index * 1.75
        box(4.0, y, 3.1, 1.25, title, body, color)
        center = (4.0, y + 0.625)
        path_centers.append(center)
        arrow(frozen, center)

    observe = box(
        8.2,
        5.35,
        3.4,
        2.15,
        "Common observability bundle",
        "metric + uncertainty\naligned visual evidence\ntelemetry + lineage\nfailure taxonomy",
        "#e0f2fe",
    )
    for center in path_centers:
        arrow((7.1, center[1]), (8.2, 6.425))

    decisions = [
        ("Local defect", "fails with oracle input", "#fecaca"),
        ("Upstream propagation", "oracle input removes failure", "#fed7aa"),
        ("Interaction defect", "only pair reproduces failure", "#ddd6fe"),
        ("Observer blind spot", "injection is not detected", "#fbcfe8"),
        ("Robust candidate", "passes gates with useful margin", "#bbf7d0"),
    ]
    for index, (title, body, color) in enumerate(decisions):
        y = 8.05 - index * 1.55
        box(13.0, y, 4.3, 1.05, title, body, color)
        arrow(observe, (13.0, y + 0.525))

    axis.text(
        0.5,
        0.35,
        "A variant cannot be selected merely because its output looks better: "
        "the protocol must also show which failure class changed and why.",
        fontsize=11.5,
        color="#7f1d1d",
        fontweight="bold",
    )
    fig.savefig(output / "08_v1_1_diagnostic_protocol.png", dpi=150)
    plt.close(fig)


def plot_failure_localization_tree(output: Path, plan_version: str) -> None:
    if plan_version.strip().upper() != "V1.1":
        return

    fig, axis = plt.subplots(figsize=(18, 8.2), constrained_layout=True)
    axis.set_xlim(0, 18)
    axis.set_ylim(0, 8.2)
    axis.axis("off")

    def node(
        x: float,
        y: float,
        width: float,
        text: str,
        color: str,
    ) -> tuple[float, float, float, float]:
        height = 1.05
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.04,rounding_size=0.1",
                facecolor=color,
                edgecolor="#334155",
                linewidth=1.3,
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )
        return x, y, width, height

    def connect(
        source: tuple[float, float, float, float],
        target: tuple[float, float, float, float],
        label: str,
    ) -> None:
        start = (source[0] + source[2], source[1] + source[3] / 2)
        end = (target[0], target[1] + target[3] / 2)
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                color="#475569",
                linewidth=1.3,
            )
        )
        axis.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.17,
            label,
            ha="center",
            fontsize=9,
            color="#334155",
        )

    def connect_down(
        source: tuple[float, float, float, float],
        target: tuple[float, float, float, float],
        label: str,
    ) -> None:
        start = (source[0] + source[2] / 2, source[1])
        end = (target[0] + target[2] / 2, target[1] + target[3])
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                color="#475569",
                linewidth=1.3,
            )
        )
        axis.text(
            start[0] + 0.15,
            (start[1] + end[1]) / 2,
            label,
            fontsize=9,
            color="#334155",
        )

    axis.text(
        9,
        7.8,
        "Failure localization decision tree",
        ha="center",
        fontsize=19,
        fontweight="bold",
    )
    symptom = node(0.4, 3.55, 2.2, "Observed symptom", "#fee2e2")
    contract = node(3.25, 3.55, 2.7, "Input + lineage\ncontract passes?", "#dbeafe")
    input_fault = node(
        3.25,
        1.25,
        2.7,
        "F-INPUT data/lineage\ncontract defect",
        "#fecaca",
    )
    isolated = node(6.7, 3.55, 2.7, "Target module passes\nwith oracle input?", "#fef3c7")
    local = node(6.7, 1.25, 2.7, "Local module defect", "#fecaca")
    upstream = node(10.15, 3.55, 2.7, "Replacing upstream\nremoves failure?", "#e0f2fe")
    propagated = node(10.15, 1.25, 2.7, "Upstream-propagated\ndefect", "#fed7aa")
    interaction = node(13.6, 3.55, 2.7, "Planned pair test\nreproduces failure?", "#ede9fe")
    pair_fault = node(13.6, 1.25, 2.7, "Cross-module\ninteraction defect", "#ddd6fe")
    unresolved = node(13.6, 5.75, 3.8, "GT / metric / unmodeled\nfailure: retain as unresolved", "#fbcfe8")

    connect(symptom, contract, "")
    connect_down(contract, input_fault, "no")
    connect(contract, isolated, "yes")
    connect_down(isolated, local, "no")
    connect(isolated, upstream, "yes")
    connect_down(upstream, propagated, "yes")
    connect(upstream, interaction, "no")
    connect_down(interaction, pair_fault, "yes")
    start = (
        interaction[0] + interaction[2],
        interaction[1] + interaction[3] / 2,
    )
    end = (unresolved[0], unresolved[1] + unresolved[3] / 2)
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle="arc3,rad=-0.36",
            arrowstyle="-|>",
            mutation_scale=13,
            color="#475569",
            linewidth=1.3,
        )
    )
    axis.text(16.4, 5.15, "no", fontsize=9, color="#334155")
    axis.text(
        0.5,
        0.35,
        "Every failed sample receives a primary cause, optional secondary cause, "
        "and evidence links; unresolved is valid, silent dropping is not.",
        fontsize=11.5,
        color="#7f1d1d",
        fontweight="bold",
    )
    fig.savefig(output / "09_v1_1_failure_localization_tree.png", dpi=150)
    plt.close(fig)


def plot_stage_montage(
    geometry: Path,
    semantic: Path,
    selected_row: dict[str, Any],
    output: Path,
) -> int:
    index = int(selected_row["selected_frame"])
    frame_name = f"{index:08d}.png"
    rgb = cv2.imread(str(geometry / "02_selected/rgb" / frame_name))
    right = cv2.imread(str(geometry / "02_selected/stereo_right" / frame_name))
    if rgb is None or right is None:
        raise FileNotFoundError(f"missing selected RGB pair for frame {index}")
    stereo = np.hstack((cv2.resize(rgb, (640, 480)), cv2.resize(right, (640, 480))))
    for y in range(40, stereo.shape[0], 55):
        cv2.line(stereo, (0, y), (stereo.shape[1] - 1, y), (0, 255, 255), 1)
    stereo = cv2.resize(stereo, (640, 480))

    disparity = cv2.imread(
        str(geometry / "02_selected/raw_disparity_visualization" / frame_name)
    )
    depth_overlay = cv2.imread(
        str(geometry / "02_selected/raw_depth_overlay_5m" / frame_name)
    )
    consistency = cv2.imread(
        str(geometry / "02_selected/depth_consistency" / frame_name)
    )
    filtered_raw = cv2.imread(
        str(geometry / "08_temporal_depth_filtered/depth" / frame_name),
        cv2.IMREAD_UNCHANGED,
    )
    filtered = depth_color(filtered_raw)
    dynamic = cv2.imread(
        str(semantic / "dynamic_masks" / frame_name), cv2.IMREAD_GRAYSCALE
    )
    unknown = cv2.imread(
        str(semantic / "unknown_masks" / frame_name), cv2.IMREAD_GRAYSCALE
    )
    isolation = rgb.copy()
    isolation[unknown > 0] = (
        0.45 * isolation[unknown > 0] + 0.55 * np.asarray([0, 215, 255])
    ).astype(np.uint8)
    isolation[dynamic > 0] = (
        0.35 * isolation[dynamic > 0] + 0.65 * np.asarray([0, 0, 255])
    ).astype(np.uint8)
    labels = cv2.imread(
        str(semantic / "semantic_sidecar/label_frames" / frame_name),
        cv2.IMREAD_UNCHANGED,
    )
    semantics = semantic_overlay(rgb, labels)
    confidence = cv2.imread(
        str(geometry / "02_selected/depth_confidence" / frame_name)
    )
    panels = [
        (stereo, "rectified stereo + epipolar guides"),
        (disparity, "raw disparity"),
        (depth_overlay, "raw depth overlay (0-5 m)"),
        (consistency, "left-right consistency"),
        (confidence, "depth confidence"),
        (filtered, "filtered metric depth (0-8 m)"),
        (isolation, "static isolation: yellow unknown / red dynamic"),
        (semantics, "exact semantic-label overlay"),
    ]
    rendered = [
        label_panel(fit_panel(panel), label) for panel, label in panels
    ]
    canvas = np.vstack((np.hstack(rendered[:4]), np.hstack(rendered[4:])))
    cv2.imwrite(str(output / "06_representative_module_montage.png"), canvas)
    return index


def plot_stage_scorecard(metrics: dict[str, Any], output: Path) -> None:
    stages = [
        ("Input sync", metrics["raw"]["stereo_sync_pass_10ms_ratio"], "10 ms pass ratio"),
        ("Rectification", metrics["rectification"]["prepared_coverage_ratio"], "materialized coverage"),
        ("Keyframes", metrics["selection"]["selected_ratio"], "retained frames"),
        ("Stereo depth", metrics["depth"]["valid_ratio"]["mean"], "mean valid ratio"),
        ("LR check", metrics["depth"]["left_right_consistency"]["mean"], "mean LR consistency"),
        ("Temporal", metrics["temporal"]["agreement_rate"]["mean"], "mean agreement"),
        ("Depth filter", metrics["filter"]["output_valid_ratio"]["mean"], "mean output valid"),
        ("Semantics", metrics["semantics"]["label_frame_coverage"], "label-frame coverage"),
        ("Evidence", 0.0, "manual GT missing"),
    ]
    names = [stage[0] for stage in stages]
    values = [float(stage[1] or 0.0) for stage in stages]
    colors = ["#16a34a" if value >= 0.8 else "#d97706" if value >= 0.5 else "#dc2626" for value in values]
    colors[-1] = "#dc2626"
    fig, axis = plt.subplots(figsize=(13, 6.8), constrained_layout=True)
    bars = axis.barh(names[::-1], values[::-1], color=colors[::-1])
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("diagnostic coverage / ratio (not a unified quality score)")
    axis.set_title("Current evidence readiness for the 473–573 window", fontsize=18, fontweight="bold")
    for bar, (_, value, note) in zip(bars, stages[::-1]):
        axis.text(
            min(float(value or 0.0) + 0.015, 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"{float(value or 0.0):.3f}  {note}",
            va="center",
            fontsize=10,
        )
    axis.grid(axis="x", alpha=0.2)
    fig.savefig(output / "07_evidence_readiness_scorecard.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    raw = args.raw.expanduser().resolve()
    prepared = args.prepared.expanduser().resolve()
    geometry = args.geometry.expanduser().resolve()
    semantic = args.semantic.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = load_raw_records(raw, args.start, args.end)

    prepared_tick = read_json(prepared / "tick_index.json")
    prepared_by_source = {
        int(frame["source_idx"]): frame for frame in prepared_tick["frames"]
    }
    selected_root = geometry / "02_selected"
    selected_tick = read_json(selected_root / "tick_index.json")
    selected_by_source = {
        int(frame["source_idx"]): frame for frame in selected_tick["frames"]
    }
    selected_by_index = {
        int(frame["idx"]): frame for frame in selected_tick["frames"]
    }
    depth_run = read_json(selected_root / "fast_foundation_stereo_run.json")
    depth_by_index = {
        int(frame["frame_idx"]): frame for frame in depth_run["frame_stats"]
    }
    temporal_report = read_json(
        geometry / "04_temporal_input/temporal_depth_consistency_report.json"
    )
    temporal_by_index: dict[int, list[float]] = {}
    for pair in temporal_report["pairs"]:
        reference = int(pair["reference_frame"])
        neighbor = int(pair["neighbor_frame"])
        raw_reference = int(selected_by_index[reference]["source_idx"])
        raw_neighbor = int(selected_by_index[neighbor]["source_idx"])
        if (
            args.start <= raw_reference <= args.end
            and args.start <= raw_neighbor <= args.end
        ):
            temporal_by_index.setdefault(reference, []).append(
                float(pair["agreement_rate"])
            )
    filter_report = read_json(
        geometry / "08_temporal_depth_filtered/temporal_depth_filter_report.json"
    )
    filter_by_index = {
        int(frame["frame"]): frame for frame in filter_report["per_frame"]
    }

    row_by_source: dict[int, dict[str, Any]] = {}
    raw_epipolar_values = []
    rectified_epipolar_values = []
    for record in records:
        source_frame = int(record["tick"])
        image_by_camera = {
            str(image["camera"]): raw / image["path"] for image in record["images"]
        }
        cam0_time = int(
            next(image["sensor_time_ns"] for image in record["images"] if image["camera"] == "cam0")
        )
        cam1_time = int(
            next(image["sensor_time_ns"] for image in record["images"] if image["camera"] == "cam1")
        )
        lidar_time = int(record["lidar"][0]["sensor_time_ns"])
        brightness, sharpness = image_metrics(image_by_camera["cam0"])
        pose = record["poses"]["values"]["map"]["pose_xyz_quat_xyzw"]
        raw_epi = mutual_sift_vertical_residual(
            image_by_camera["cam0"], image_by_camera["cam1"]
        )
        prepared_frame = prepared_by_source.get(source_frame)
        rectified_epi = (
            mutual_sift_vertical_residual(
                Path(prepared_frame["cam0"]), Path(prepared_frame["cam1"])
            )
            if prepared_frame is not None
            else {"matches": 0, "median_px": None, "p95_px": None}
        )
        if raw_epi["median_px"] is not None:
            raw_epipolar_values.append(float(raw_epi["median_px"]))
        if rectified_epi["median_px"] is not None:
            rectified_epipolar_values.append(float(rectified_epi["median_px"]))
        row = {
            "source_frame": source_frame,
            "split": split_for(source_frame),
            "cam0_sensor_time_ns": cam0_time,
            "cam1_sensor_time_ns": cam1_time,
            "lidar_sensor_time_ns": lidar_time,
            "stereo_delta_ms": abs(cam0_time - cam1_time) / 1.0e6,
            "cam0_lidar_skew_ms": (cam0_time - lidar_time) / 1.0e6,
            "cam1_lidar_skew_ms": (cam1_time - lidar_time) / 1.0e6,
            "brightness": brightness,
            "sharpness": sharpness,
            "map_x_m": float(pose[0]),
            "map_y_m": float(pose[1]),
            "map_z_m": float(pose[2]),
            "map_yaw_deg": yaw_deg(pose[3:]),
            "prepared_frame": (
                int(prepared_frame["idx"]) if prepared_frame is not None else None
            ),
            "raw_epipolar_matches": int(raw_epi["matches"]),
            "raw_epipolar_median_px": raw_epi["median_px"],
            "raw_epipolar_p95_px": raw_epi["p95_px"],
            "rectified_epipolar_matches": int(rectified_epi["matches"]),
            "rectified_epipolar_median_px": rectified_epi["median_px"],
            "rectified_epipolar_p95_px": rectified_epi["p95_px"],
        }
        selected_frame = selected_by_source.get(source_frame)
        if selected_frame is not None:
            index = int(selected_frame["idx"])
            depth = depth_by_index[index]
            filtered = filter_by_index[index]
            labels = cv2.imread(
                str(
                    semantic
                    / "semantic_sidecar/label_frames"
                    / f"{index:08d}.png"
                ),
                cv2.IMREAD_UNCHANGED,
            )
            dynamic = cv2.imread(
                str(semantic / "dynamic_masks" / f"{index:08d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            unknown = cv2.imread(
                str(semantic / "unknown_masks" / f"{index:08d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            unique_labels = np.unique(labels)
            row.update(
                {
                    "selected_frame": index,
                    "selection_reason": str(selected_frame["selection_reason"]),
                    "depth_valid_ratio": float(depth["valid_ratio"]),
                    "lr_consistency": float(depth["left_right_consistency"]),
                    "mean_confidence": float(depth["mean_confidence"]),
                    "median_depth_m": float(depth["median_depth_m"]),
                    "temporal_agreement": (
                        float(np.mean(temporal_by_index[index]))
                        if temporal_by_index.get(index)
                        else None
                    ),
                    "filter_input_ratio": float(filtered["input_valid_ratio"]),
                    "filter_output_ratio": float(filtered["output_valid_ratio"]),
                    "filter_rejected_ratio": float(filtered["rejected_valid_ratio"]),
                    "semantic_label_count": int(np.count_nonzero(unique_labels)),
                    "semantic_nonzero_ratio": float(np.count_nonzero(labels))
                    / float(labels.size),
                    "dynamic_ratio": float(np.count_nonzero(dynamic))
                    / float(dynamic.size),
                    "unknown_ratio": float(np.count_nonzero(unknown))
                    / float(unknown.size),
                }
            )
        row_by_source[source_frame] = row
    rows = [row_by_source[index] for index in range(args.start, args.end + 1)]

    positions = np.asarray(
        [[row["map_x_m"], row["map_y_m"], row["map_z_m"]] for row in rows]
    )
    raw_path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    yaws = np.unwrap(np.radians([row["map_yaw_deg"] for row in rows]))
    selected_rows = [row for row in rows if row.get("selected_frame") is not None]

    time_min = min(int(row["cam0_sensor_time_ns"]) for row in rows)
    time_max = max(int(row["cam0_sensor_time_ns"]) for row in rows)
    memory_path = semantic / "map_memory.sqlite3"
    memory_counts = {"observations": 0, "entities": 0, "semantic_operations": 0}
    if memory_path.is_file():
        connection = sqlite3.connect(f"file:{memory_path}?mode=ro", uri=True)
        try:
            memory_counts["observations"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM entity_observations "
                    "WHERE sensor_time_ns BETWEEN ? AND ?",
                    (time_min, time_max),
                ).fetchone()[0]
            )
            memory_counts["entities"] = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT entity_id) FROM entity_observations "
                    "WHERE sensor_time_ns BETWEEN ? AND ?",
                    (time_min, time_max),
                ).fetchone()[0]
            )
            memory_counts["semantic_operations"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM semantic_operations "
                    "WHERE sensor_time_ns BETWEEN ? AND ?",
                    (time_min, time_max),
                ).fetchone()[0]
            )
        finally:
            connection.close()

    stereo_deltas = [row["stereo_delta_ms"] for row in rows]
    rectified_count = sum(row["prepared_frame"] is not None for row in rows)
    selected_count = len(selected_rows)
    metrics = {
        "schema": "daaam.g1_473_573_visual_audit.v1",
        "source": {
            "dataset": str(raw),
            "frames": [args.start, args.end],
            "frame_count": len(rows),
            "existing_artifacts_are_full_run_diagnostics": True,
        },
        "splits_under_review": {
            name: {"frames": [start, end], "count": end - start + 1}
            for name, (start, end) in SPLITS.items()
        },
        "raw": {
            "stereo_delta_ms": percentiles(stereo_deltas),
            "stereo_sync_pass_2ms": sum(value <= 2.0 for value in stereo_deltas),
            "stereo_sync_pass_5ms": sum(value <= 5.0 for value in stereo_deltas),
            "stereo_sync_pass_10ms": sum(value <= 10.0 for value in stereo_deltas),
            "stereo_sync_pass_10ms_ratio": sum(value <= 10.0 for value in stereo_deltas)
            / len(rows),
            "brightness": percentiles(row["brightness"] for row in rows),
            "sharpness": percentiles(row["sharpness"] for row in rows),
            "trajectory_path_length_m": raw_path_length,
            "trajectory_displacement_m": displacement,
            "yaw_change_deg": float(math.degrees(yaws[-1] - yaws[0])),
        },
        "rectification": {
            "prepared_frames": rectified_count,
            "prepared_coverage_ratio": rectified_count / len(rows),
            "missing_source_frames": [
                row["source_frame"] for row in rows if row["prepared_frame"] is None
            ],
            "raw_mutual_sift_vertical_median_px": percentiles(raw_epipolar_values),
            "rectified_mutual_sift_vertical_median_px": percentiles(
                rectified_epipolar_values
            ),
            "diagnostic_only": True,
        },
        "selection": {
            "selected_frames": selected_count,
            "selected_ratio": selected_count / len(rows),
            "reasons": dict(Counter(row["selection_reason"] for row in selected_rows)),
            "selected_source_frames": [row["source_frame"] for row in selected_rows],
        },
        "depth": {
            "valid_ratio": percentiles(row["depth_valid_ratio"] for row in selected_rows),
            "left_right_consistency": percentiles(
                row["lr_consistency"] for row in selected_rows
            ),
            "mean_confidence": percentiles(row["mean_confidence"] for row in selected_rows),
            "median_depth_m": percentiles(row["median_depth_m"] for row in selected_rows),
        },
        "temporal": {
            "agreement_rate": percentiles(
                row["temporal_agreement"]
                for row in selected_rows
                if row["temporal_agreement"] is not None
            ),
        },
        "filter": {
            "input_valid_ratio": percentiles(
                row["filter_input_ratio"] for row in selected_rows
            ),
            "output_valid_ratio": percentiles(
                row["filter_output_ratio"] for row in selected_rows
            ),
            "rejected_valid_ratio": percentiles(
                row["filter_rejected_ratio"] for row in selected_rows
            ),
        },
        "semantics": {
            "label_frame_coverage": sum(
                row.get("semantic_label_count") is not None for row in selected_rows
            )
            / max(1, selected_count),
            "labels_per_frame": percentiles(
                row["semantic_label_count"] for row in selected_rows
            ),
            "semantic_nonzero_ratio": percentiles(
                row["semantic_nonzero_ratio"] for row in selected_rows
            ),
            "dynamic_ratio": percentiles(row["dynamic_ratio"] for row in selected_rows),
            "unknown_ratio": percentiles(row["unknown_ratio"] for row in selected_rows),
            "map_memory_window": memory_counts,
            "manual_ground_truth_available": False,
        },
        "limitations": [
            "The 101 frames form one continuous, highly overlapping indoor maneuver.",
            "Existing stage artifacts were produced by the 844-frame full run, not isolated 473-573 runs.",
            "No complete reviewed human instance/track/DSG ground truth exists for this window.",
            "Mutual-SIFT epipolar residual is a diagnostic proxy and must not replace audited correspondences.",
            "Loop closure, global consistency, room topology, and cross-scene generalization cannot be established from this window alone.",
        ],
    }

    plot_overview(raw, output)
    plot_input_audit(rows, output)
    plot_pose_and_epipolar(rows, output)
    plot_module_effects(rows, output)
    plot_pipeline_flow(output, args.plan_version)
    plot_diagnostic_protocol(output, args.plan_version)
    plot_failure_localization_tree(output, args.plan_version)
    representative = max(
        selected_rows,
        key=lambda row: (
            row["semantic_label_count"],
            row["semantic_nonzero_ratio"],
            -abs(row["source_frame"] - (args.start + args.end) / 2),
        ),
    )
    representative_index = plot_stage_montage(
        geometry, semantic, representative, output
    )
    metrics["representative_frame"] = {
        "source_frame": representative["source_frame"],
        "selected_frame": representative_index,
        "semantic_label_count": representative["semantic_label_count"],
    }
    plot_stage_scorecard(metrics, output)

    metrics_path = output / "g1_473_573_visual_audit.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = output / "g1_473_573_per_frame.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    artifact_manifest = {
        "schema": "daaam.g1_473_573_visual_artifacts.v1",
        "source_metrics": metrics_path.name,
        "files": [
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_manifest.json"
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
