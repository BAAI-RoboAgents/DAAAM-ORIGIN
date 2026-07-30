#!/usr/bin/env python3
"""Estimate a fixed image-frame transform from floor planes across a capture.

Unlike the single-batch calibrator, this tool fits an independent floor plane
in uniformly distributed frames.  The changing camera attitude makes the
otherwise ambiguous rotation about gravity observable, so one fixed transform
can be estimated without modifying the source RGB-D dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_g1_floor_geometry import ransac_plane


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        help="Save per-frame floor fits and a sequence summary without subsampling.",
    )
    parser.add_argument("--frame-count", type=int, default=150)
    parser.add_argument("--floor-world-z-m", type=float, default=0.0)
    parser.add_argument("--roi-x-min", type=float, default=0.03)
    parser.add_argument("--roi-x-max", type=float, default=0.97)
    parser.add_argument("--roi-y-min", type=float, default=0.64)
    parser.add_argument("--roi-y-max", type=float, default=1.0)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--residual-threshold", type=float, default=0.02)
    parser.add_argument("--max-trials", type=int, default=500)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.32)
    parser.add_argument("--minimum-floor-distance-m", type=float, default=0.7)
    parser.add_argument("--maximum-floor-distance-m", type=float, default=2.5)
    parser.add_argument("--maximum-median-angular-residual-deg", type=float, default=5.0)
    parser.add_argument("--maximum-p90-angular-residual-deg", type=float, default=10.0)
    parser.add_argument(
        "--maximum-frame-angular-residual-deg",
        type=float,
        default=10.0,
        help=(
            "Reject a per-frame plane whose normal cannot be reconciled with "
            "the recorded camera gravity direction by the shared fixed "
            "rotation. This removes wall/table fits from the floor ROI before "
            "the final quality statistics are evaluated."
        ),
    )
    parser.add_argument(
        "--yaw-policy",
        choices=("minimal", "wahba"),
        default="minimal",
        help=(
            "minimal applies only the shortest gravity-alignment rotation and "
            "preserves source yaw. wahba also estimates yaw, but is rejected "
            "when the floor-normal observations are rank deficient."
        ),
    )
    parser.add_argument(
        "--minimum-yaw-observability-ratio", type=float, default=0.005
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], found {value}")


def load_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64).reshape(-1, 4, 4)
    if not np.isfinite(poses).all() or not np.allclose(
        poses[:, 3, :], [0.0, 0.0, 0.0, 1.0]
    ):
        raise ValueError(f"Invalid homogeneous poses in {path}")
    return poses


def fit_shared_rotation(
    records: list[dict],
    yaw_policy: str,
    minimum_yaw_observability_ratio: float,
    maximum_frame_angular_residual_deg: float,
) -> tuple[Rotation, float, np.ndarray, np.ndarray, float, bool, np.ndarray]:
    """Robustly fit one image-frame rotation and reject inconsistent planes."""

    normals_all = np.asarray(
        [record["floor_normal_image_frame"] for record in records],
        dtype=np.float64,
    )
    current_ups_all = np.asarray(
        [record["current_up_image_frame"] for record in records],
        dtype=np.float64,
    )
    weights_all = np.asarray(
        [record["inlier_ratio"] for record in records], dtype=np.float64
    )
    retained = np.ones(len(records), dtype=bool)
    correction = Rotation.identity()
    rssd = 0.0
    observability_singular_values = np.zeros(3, dtype=np.float64)
    yaw_observability_ratio = 0.0
    yaw_observable = False
    angular_residuals_all = np.full(len(records), np.inf, dtype=np.float64)

    for _ in range(10):
        normals = normals_all[retained]
        current_ups = current_ups_all[retained]
        weights = weights_all[retained]
        cross_covariance = np.einsum(
            "n,ni,nj->ij", weights, current_ups, normals
        )
        observability_singular_values = np.linalg.svd(
            cross_covariance, compute_uv=False
        )
        yaw_observability_ratio = float(
            observability_singular_values[-1]
            / max(observability_singular_values[0], 1.0e-15)
        )
        yaw_observable = (
            yaw_observability_ratio >= minimum_yaw_observability_ratio
        )
        if yaw_policy == "wahba":
            if not yaw_observable:
                raise RuntimeError(
                    "Floor normals do not observe a stable yaw correction: "
                    f"singular-value ratio={yaw_observability_ratio:.6g} < "
                    f"{minimum_yaw_observability_ratio:.6g}. Use "
                    "--yaw-policy minimal and validate yaw against LiDAR."
                )
            correction, rssd = Rotation.align_vectors(
                current_ups, normals, weights=weights
            )
        else:
            mean_normal = np.average(normals, axis=0, weights=weights)
            mean_normal /= np.linalg.norm(mean_normal)
            mean_current_up = np.average(current_ups, axis=0, weights=weights)
            mean_current_up /= np.linalg.norm(mean_current_up)
            correction, rssd = Rotation.align_vectors(
                mean_current_up.reshape(1, 3), mean_normal.reshape(1, 3)
            )
        corrected_normals_all = correction.apply(normals_all)
        angular_residuals_all = np.rad2deg(
            np.arccos(
                np.clip(
                    np.sum(corrected_normals_all * current_ups_all, axis=1),
                    -1.0,
                    1.0,
                )
            )
        )
        updated = (
            angular_residuals_all <= maximum_frame_angular_residual_deg
        )
        if np.array_equal(updated, retained):
            break
        retained = updated
        if int(retained.sum()) < 3:
            break

    return (
        correction,
        float(rssd),
        retained,
        angular_residuals_all,
        yaw_observability_ratio,
        yaw_observable,
        observability_singular_values,
    )


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    report_path = (
        args.output_report.expanduser().resolve()
        if args.output_report is not None
        else dataset / "multiframe_floor_geometry_calibration.json"
    )
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists: {report_path}. Pass --overwrite.")
    if args.frame_count < 10:
        raise ValueError("--frame-count must be at least 10")
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be positive")
    for name in (
        "roi_x_min",
        "roi_x_max",
        "roi_y_min",
        "roi_y_max",
        "minimum_inlier_ratio",
    ):
        validate_fraction(f"--{name.replace('_', '-')}", getattr(args, name))
    if args.roi_x_min >= args.roi_x_max or args.roi_y_min >= args.roi_y_max:
        raise ValueError("Floor ROI bounds are empty")

    camera = json.loads((dataset / "camera_info.json").read_text())
    tick_index = json.loads((dataset / "tick_index.json").read_text())
    if camera.get("model") != "pinhole" or tick_index.get("projection_model") != "pinhole":
        raise ValueError("Multiframe floor calibration requires a pinhole dataset")
    poses = load_poses(dataset / "pose" / "poses.txt")
    depth_paths = sorted((dataset / "depth").glob("*.png"))
    if len(depth_paths) != len(poses):
        raise ValueError("Pose and depth counts must agree")

    frame_indices = np.unique(
        np.linspace(0, len(depth_paths) - 1, min(args.frame_count, len(depth_paths)))
        .round()
        .astype(int)
    )
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    width = int(camera["width"])
    height = int(camera["height"])
    x0 = int(round(args.roi_x_min * width))
    x1 = int(round(args.roi_x_max * width))
    y0 = int(round(args.roi_y_min * height))
    y1 = int(round(args.roi_y_max * height))
    max_depth = float(tick_index.get("recommended_max_depth_m", 5.0))
    visualization_dir = (
        args.visualization_dir.expanduser().resolve()
        if args.visualization_dir is not None
        else None
    )
    if visualization_dir is not None:
        if visualization_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"Floor visualization directory exists: {visualization_dir}"
            )
        visualization_dir.mkdir(parents=True, exist_ok=True)

    def save_frame_visualization(
        frame_index: int,
        label: str,
        color: tuple[int, int, int],
        sample_u: np.ndarray | None = None,
        sample_v: np.ndarray | None = None,
        inliers: np.ndarray | None = None,
    ) -> str | None:
        if visualization_dir is None:
            return None
        image = cv2.imread(
            str(tick_index["frames"][frame_index]["cam0"]),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise FileNotFoundError(tick_index["frames"][frame_index]["cam0"])
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 4)
        if sample_u is not None and sample_v is not None and inliers is not None:
            selected = np.linspace(
                0,
                len(sample_u) - 1,
                min(1500, len(sample_u)),
            ).astype(int)
            for index in selected:
                point_color = (
                    (40, 210, 40) if bool(inliers[index]) else (30, 30, 220)
                )
                cv2.circle(
                    image,
                    (int(sample_u[index]), int(sample_v[index])),
                    1,
                    point_color,
                    -1,
                )
        cv2.rectangle(image, (0, 0), (image.shape[1], 62), (0, 0, 0), -1)
        cv2.putText(
            image,
            label,
            (18, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
            cv2.LINE_AA,
        )
        path = visualization_dir / f"{frame_index:06d}.jpg"
        if not cv2.imwrite(
            str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]
        ):
            raise RuntimeError(f"Could not write floor visualization: {path}")
        return str(path)

    def write_failure_report(
        reason: str,
        accepted_records: list[dict],
        rejected_records: list[dict],
        **details: object,
    ) -> None:
        failure = {
            "schema": "daaam.g1_multiframe_floor_geometry.v1",
            "status": "failed_quality_gate",
            "reason": reason,
            "dataset": str(dataset),
            "source_dataset_unmodified": True,
            "requested_frame_count": args.frame_count,
            "sampled_frame_count": int(len(frame_indices)),
            "accepted_frame_count": len(accepted_records),
            "rejected_frame_count": len(rejected_records),
            "floor_roi_pixels": [x0, y0, x1, y1],
            "details": details,
            "accepted_frames": accepted_records,
            "rejected_frames": rejected_records,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(failure, indent=2) + "\n")
        if visualization_dir is not None:
            canvas = np.full((420, 1400, 3), 250, dtype=np.uint8)
            cv2.putText(
                canvas,
                "FLOOR CALIBRATION FAILED QUALITY GATE",
                (45, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.35,
                (20, 20, 210),
                3,
                cv2.LINE_AA,
            )
            lines = (
                f"reason={reason}",
                f"accepted={len(accepted_records)}/{len(frame_indices)}",
                f"rejected={len(rejected_records)}",
                json.dumps(details, ensure_ascii=False),
            )
            for index, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line[:170],
                    (45, 165 + 55 * index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (30, 30, 30),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imwrite(
                str(visualization_dir / "floor_calibration_failure.png"),
                canvas,
            )

    accepted = []
    rejected = []
    for ordinal, frame_index in enumerate(frame_indices, start=1):
        depth_path = depth_paths[int(frame_index)]
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None or depth_mm.dtype != np.uint16:
            raise ValueError(f"Expected uint16 depth image: {depth_path}")
        depth = depth_mm.astype(np.float64) / 1000.0
        v, u = np.mgrid[y0:y1:args.sample_stride, x0:x1:args.sample_stride]
        z = depth[y0:y1:args.sample_stride, x0:x1:args.sample_stride]
        valid = (z >= 0.25) & (z < max_depth)
        if int(valid.sum()) < 1000:
            rejected.append(
                {
                    "frame_index": int(frame_index),
                    "reason": "too_few_samples",
                    "visualization": save_frame_visualization(
                        int(frame_index),
                        f"frame={int(frame_index)} REJECT too_few_samples",
                        (30, 30, 220),
                    ),
                }
            )
            continue
        design = np.column_stack(
            (
                (u[valid] - cx) / fx,
                (v[valid] - cy) / fy,
                np.ones(int(valid.sum())),
            )
        )
        coefficients, inlier_mask, residuals = ransac_plane(
            design,
            1.0 / z[valid],
            args.residual_threshold,
            args.max_trials,
            args.seed + int(frame_index),
        )
        normal_scale = float(np.linalg.norm(coefficients))
        floor_normal = coefficients / normal_scale
        plane_offset = -1.0 / normal_scale
        current_up = poses[int(frame_index), :3, :3].T @ np.array([0.0, 0.0, 1.0])
        current_up /= np.linalg.norm(current_up)
        if np.dot(floor_normal, current_up) < 0.0:
            floor_normal *= -1.0
            plane_offset *= -1.0
        inlier_ratio = float(inlier_mask.mean())
        reason = None
        if inlier_ratio < args.minimum_inlier_ratio:
            reason = "low_inlier_ratio"
        elif not args.minimum_floor_distance_m <= plane_offset <= args.maximum_floor_distance_m:
            reason = "implausible_floor_distance"
        if reason is not None:
            rejected.append(
                {
                    "frame_index": int(frame_index),
                    "reason": reason,
                    "inlier_ratio": inlier_ratio,
                    "floor_distance_m": plane_offset,
                    "visualization": save_frame_visualization(
                        int(frame_index),
                        (
                            f"frame={int(frame_index)} REJECT {reason} "
                            f"inliers={inlier_ratio:.3f} floor={plane_offset:.3f}m"
                        ),
                        (30, 30, 220),
                        u[valid],
                        v[valid],
                        inlier_mask,
                    ),
                }
            )
            continue
        camera_height = float(
            poses[int(frame_index), 2, 3] - args.floor_world_z_m
        )
        accepted.append(
            {
                "frame_index": int(frame_index),
                "sample_count": int(valid.sum()),
                "inlier_ratio": inlier_ratio,
                "median_inverse_depth_residual": float(
                    np.median(residuals[inlier_mask])
                ),
                "floor_normal_image_frame": floor_normal.tolist(),
                "current_up_image_frame": current_up.tolist(),
                "floor_distance_m": plane_offset,
                "camera_height_m": camera_height,
                "depth_scale": camera_height / plane_offset,
                "visualization": save_frame_visualization(
                    int(frame_index),
                    (
                        f"frame={int(frame_index)} ACCEPT "
                        f"inliers={inlier_ratio:.3f} floor={plane_offset:.3f}m"
                    ),
                    (40, 210, 40),
                    u[valid],
                    v[valid],
                    inlier_mask,
                ),
            }
        )
        if ordinal % 25 == 0 or ordinal == len(frame_indices):
            print(
                f"Fitted {ordinal}/{len(frame_indices)} frames; accepted={len(accepted)}",
                flush=True,
            )

    minimum_accepted = max(20, len(frame_indices) // 2)
    if len(accepted) < minimum_accepted:
        write_failure_report(
            "too_few_reliable_floor_planes",
            accepted,
            rejected,
            minimum_accepted=minimum_accepted,
        )
        raise RuntimeError(
            f"Too few reliable floor planes: {len(accepted)}/{len(frame_indices)}"
        )
    (
        correction,
        rssd,
        retained,
        angular_residuals_all,
        yaw_observability_ratio,
        yaw_observable,
        observability_singular_values,
    ) = fit_shared_rotation(
        accepted,
        args.yaw_policy,
        args.minimum_yaw_observability_ratio,
        args.maximum_frame_angular_residual_deg,
    )
    inconsistent = [record for record, keep in zip(accepted, retained) if not keep]
    for record, keep, angular_residual in zip(
        accepted, retained, angular_residuals_all
    ):
        if not keep:
            rejected.append(
                {
                    "frame_index": record["frame_index"],
                    "reason": "inconsistent_floor_normal",
                    "angular_residual_deg": float(angular_residual),
                    "inlier_ratio": record["inlier_ratio"],
                    "floor_distance_m": record["floor_distance_m"],
                }
            )
    accepted = [record for record, keep in zip(accepted, retained) if keep]
    if len(accepted) < minimum_accepted:
        write_failure_report(
            "too_few_mutually_consistent_floor_planes",
            accepted,
            rejected,
            minimum_accepted=minimum_accepted,
            inconsistent_count=len(inconsistent),
        )
        raise RuntimeError(
            "Too few mutually consistent floor planes after gravity filtering: "
            f"{len(accepted)}/{len(frame_indices)}; rejected={len(inconsistent)}"
        )
    normals = np.asarray(
        [record["floor_normal_image_frame"] for record in accepted], dtype=np.float64
    )
    current_ups = np.asarray(
        [record["current_up_image_frame"] for record in accepted], dtype=np.float64
    )
    correction_matrix = correction.as_matrix()
    corrected_normals = correction.apply(normals)
    angular_residuals = np.rad2deg(
        np.arccos(np.clip(np.sum(corrected_normals * current_ups, axis=1), -1.0, 1.0))
    )
    median_angular_residual = float(np.median(angular_residuals))
    p90_angular_residual = float(np.percentile(angular_residuals, 90.0))
    if median_angular_residual > args.maximum_median_angular_residual_deg:
        write_failure_report(
            "median_floor_normal_residual_too_large",
            accepted,
            rejected,
            median_angular_residual_deg=median_angular_residual,
            maximum_median_angular_residual_deg=(
                args.maximum_median_angular_residual_deg
            ),
        )
        raise RuntimeError(
            f"Median floor-normal residual is too large: {median_angular_residual:.3f} deg"
        )
    if p90_angular_residual > args.maximum_p90_angular_residual_deg:
        write_failure_report(
            "p90_floor_normal_residual_too_large",
            accepted,
            rejected,
            p90_angular_residual_deg=p90_angular_residual,
            maximum_p90_angular_residual_deg=(
                args.maximum_p90_angular_residual_deg
            ),
        )
        raise RuntimeError(
            f"P90 floor-normal residual is too large: {p90_angular_residual:.3f} deg"
        )

    depth_scales = np.asarray([record["depth_scale"] for record in accepted])
    depth_scale = float(np.median(depth_scales))
    if not 0.75 <= depth_scale <= 1.5:
        write_failure_report(
            "stereo_scale_correction_implausible",
            accepted,
            rejected,
            depth_scale=depth_scale,
            allowed_range=[0.75, 1.5],
        )
        raise RuntimeError(f"Stereo scale correction is implausible: {depth_scale:.3f}")
    source_baseline = float(camera["baseline"])
    effective_baseline = source_baseline * depth_scale
    for record, angular_residual in zip(accepted, angular_residuals):
        record["angular_residual_deg"] = float(angular_residual)

    report = {
        "schema": "daaam.g1_multiframe_floor_geometry.v1",
        "method": f"uniform_multiframe_floor_planes_{args.yaw_policy}",
        "dataset": str(dataset),
        "source_dataset_unmodified": True,
        "requested_frame_count": args.frame_count,
        "sampled_frame_count": int(len(frame_indices)),
        "accepted_frame_count": len(accepted),
        "rejected_frame_count": len(rejected),
        "floor_roi_pixels": [x0, y0, x1, y1],
        "depth_scale": depth_scale,
        "depth_scale_percentiles": np.percentile(
            depth_scales, [0, 5, 25, 50, 75, 95, 100]
        ).tolist(),
        "source_baseline_m": source_baseline,
        "effective_baseline_m": effective_baseline,
        "tf_camera_R_image_camera": correction_matrix.tolist(),
        "tf_camera_R_image_camera_euler_xyz_deg": correction.as_euler(
            "xyz", degrees=True
        ).tolist(),
        "correction_angle_deg": float(np.rad2deg(correction.magnitude())),
        "floor_normal_angular_residual_deg_percentiles": np.percentile(
            angular_residuals, [0, 5, 25, 50, 75, 90, 95, 100]
        ).tolist(),
        "wahba_rssd": float(rssd),
        "observability_singular_values": observability_singular_values.tolist(),
        "yaw_policy": args.yaw_policy,
        "yaw_observable_from_floor_normals": yaw_observable,
        "yaw_observability_ratio": yaw_observability_ratio,
        "fit_parameters": {
            "residual_threshold": args.residual_threshold,
            "max_trials": args.max_trials,
            "minimum_inlier_ratio": args.minimum_inlier_ratio,
            "floor_distance_bounds_m": [
                args.minimum_floor_distance_m,
                args.maximum_floor_distance_m,
            ],
            "maximum_median_angular_residual_deg": (
                args.maximum_median_angular_residual_deg
            ),
            "maximum_p90_angular_residual_deg": args.maximum_p90_angular_residual_deg,
            "maximum_frame_angular_residual_deg": (
                args.maximum_frame_angular_residual_deg
            ),
            "minimum_yaw_observability_ratio": (
                args.minimum_yaw_observability_ratio
            ),
            "seed": args.seed,
        },
        "accepted_frames": accepted,
        "rejected_frames": rejected,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if visualization_dir is not None:
        canvas = np.full((900, 1600, 3), 250, dtype=np.uint8)
        scales = np.asarray([record["depth_scale"] for record in accepted])
        residuals = np.asarray(
            [record["angular_residual_deg"] for record in accepted]
        )
        indices = np.asarray([record["frame_index"] for record in accepted])
        for values, top, bottom, color, title in (
            (scales, 100, 400, (180, 90, 30), "per-frame depth scale"),
            (
                residuals,
                500,
                800,
                (30, 140, 50),
                "floor-normal angular residual (deg)",
            ),
        ):
            minimum = float(values.min())
            maximum = float(values.max())
            extent = max(maximum - minimum, 1.0e-9)
            points = np.column_stack(
                (
                    80
                    + (indices - indices.min())
                    / max(1, int(indices.max() - indices.min()))
                    * 1440,
                    bottom - (values - minimum) / extent * (bottom - top),
                )
            ).astype(np.int32)
            cv2.polylines(
                canvas,
                [points.reshape(-1, 1, 2)],
                False,
                color,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{title}: min={minimum:.4f} max={maximum:.4f}",
                (60, top - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            (
                f"accepted={len(accepted)} rejected={len(rejected)} "
                f"scale={depth_scale:.6f} correction={report['correction_angle_deg']:.3f}deg"
            ),
            (60, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(
            str(visualization_dir / "floor_calibration_summary.png"),
            canvas,
        ):
            raise RuntimeError("Could not write floor calibration summary")
    print(
        f"Calibrated fixed image frame from {len(accepted)}/{len(frame_indices)} floor planes\n"
        f"  angular residual median/p90={median_angular_residual:.3f}/"
        f"{p90_angular_residual:.3f} deg\n"
        f"  correction={report['correction_angle_deg']:.3f} deg, "
        f"depth scale={depth_scale:.6f}\n"
        f"  report={report_path}\n"
        "  source poses, metadata, and depth images were not modified"
    )


if __name__ == "__main__":
    main()
