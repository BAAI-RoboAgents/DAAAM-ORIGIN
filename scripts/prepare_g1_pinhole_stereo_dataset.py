#!/usr/bin/env python3
"""Reproject synchronized G1 fisheye stereo into a pinhole RGB-D dataset."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from build_synchronized_stereo_dataset import (
    camera_timestamps,
    compose_global_camera_poses,
    load_jsonl,
    map_pose_sample,
    monotonic_matches,
    odom_pose_sample,
)

PREPARATION_CONTRACT_VERSION = 4
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert synchronized G1 Kannala-Brandt stereo images to a common "
            "pinhole camera for FoundationStereo and Hydra."
        )
    )
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence", default="000000")
    parser.add_argument(
        "--base-pose-source",
        choices=("odom", "map"),
        default="odom",
        help=(
            "World-frame base pose stream. Use map for recorded map_T_base_link "
            "poses in state/<sequence>/map_pose.jsonl."
        ),
    )
    parser.add_argument(
        "--calibration-source",
        type=Path,
        help=(
            "Capture root providing calibrations/<sequence>. This is required "
            "when the image capture omitted numeric calibration; current-capture "
            "epipolar geometry is still validated."
        ),
    )
    parser.add_argument("--max-delta-ms", type=float, default=10.0)
    parser.add_argument(
        "--input-projection-model",
        choices=(
            "auto",
            "kannala_brandt",
            "pinhole_rectified",
            "pinhole_unrectified",
        ),
        default="auto",
        help=(
            "Projection model of the stored input pixels. Auto treats zero-"
            "distortion images under 2d_rect with roi.do_rectify=true as "
            "already rectified pinhole images, avoiding a second fisheye warp."
        ),
    )
    parser.add_argument("--horizontal-fov-deg", type=float, default=100.0)
    parser.add_argument(
        "--down-fov-deg",
        type=float,
        default=28.0,
        help="Vertical angle below the virtual optical axis retained in the output.",
    )
    parser.add_argument(
        "--rectification-roll-deg",
        type=float,
        default=0.0,
        help=(
            "Optional common rectification rotation about optical X. The "
            "default keeps the calibrated camera view unchanged."
        ),
    )
    parser.add_argument(
        "--camera-quaternion-order",
        choices=("auto", "xyzw", "wxyz"),
        default="auto",
        help=(
            "Storage order of head_camera orientation values. Auto compares "
            "both interpretations with the capture's recorded poses.txt "
            "rotation matrices and rejects inputs without that reference."
        ),
    )
    parser.add_argument(
        "--stereo-calibration-report",
        type=Path,
        help=(
            "Reuse a validated pinhole_preparation_report.json from the same "
            "unchanged stereo rig. Source intrinsics, distortion, baseline, "
            "rotation, translation, and current-capture epipolar geometry are "
            "validated before the report is accepted."
        ),
    )
    parser.add_argument(
        "--right-rectification-report",
        type=Path,
        help=(
            "LiDAR-validated right-only projective rectification report. The "
            "original left image and its intrinsics remain unchanged; only "
            "the stored right pixels are remapped."
        ),
    )
    parser.add_argument("--recommended-max-depth-m", type=float, default=5.0)
    parser.add_argument(
        "--source-indices",
        nargs="+",
        type=int,
        help=(
            "Optional raw manifest tick indices to prepare. This keeps a formal "
            "experiment subset from decoding or writing unrelated frames."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_calibration(src: Path, sequence: str):
    calibration_dir = src / "calibrations" / sequence
    cam0 = yaml.safe_load(
        (calibration_dir / "calib_cam0_intrinsics.yaml").read_text()
    )["intrinsics"]
    cam1 = yaml.safe_load(
        (calibration_dir / "calib_cam1_intrinsics.yaml").read_text()
    )["intrinsics"]

    for camera, values in (("cam0", cam0), ("cam1", cam1)):
        model = str(values.get("distortion_model", "")).lower()
        if model != "kannala_brandt":
            raise ValueError(
                f"{camera} must use Kannala-Brandt input, found {model!r}"
            )
        if len(values.get("D", [])) != 4:
            raise ValueError(f"{camera} must provide four fisheye coefficients")

    if cam0["K"] != cam1["K"] or cam0["R"] != cam1["R"]:
        raise ValueError("G1 stereo cameras do not share the same input geometry")
    if int(cam0["width"]) != int(cam1["width"]) or int(cam0["height"]) != int(
        cam1["height"]
    ):
        raise ValueError("G1 stereo image sizes differ")

    K = np.asarray(cam0["K"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(cam0["D"], dtype=np.float64).reshape(4, 1)
    transform = np.asarray(cam1["T"], dtype=np.float64).reshape(3, 4)
    baseline_vector = transform[:, 3]
    if np.linalg.norm(baseline_vector[1:]) > 1.0e-6:
        raise ValueError(
            "Input pair is not horizontally rectified: "
            f"baseline={baseline_vector.tolist()}"
        )
    baseline = abs(float(baseline_vector[0]))
    if baseline <= 0.0:
        raise ValueError("Stereo baseline must be positive")
    return (
        K,
        distortion,
        int(cam0["width"]),
        int(cam0["height"]),
        baseline,
        cam0,
    )


def load_declared_rectified_stereo_calibration(
    src: Path,
    sequence: str,
    baseline: float,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Load the authoritative optical-frame transform for rectified pixels.

    The capture file declares ``left_T_right``. OpenCV stereo geometry uses
    ``right_T_left``, so the returned transform is its explicit inverse.
    """

    path = src / "calibrations" / sequence / "calib_cam0_to_cam1.yaml"
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text())
    transform_record = document.get("transform")
    if not isinstance(transform_record, dict):
        return None
    if (
        transform_record.get("target_frame")
        != "head_left_camera_color_optical_frame"
        or transform_record.get("source_frame")
        != "head_right_camera_color_optical_frame"
    ):
        raise ValueError(
            "Declared stereo transform must map the right optical frame into "
            "the left optical frame"
        )
    try:
        left_T_right = np.asarray(
            transform_record["matrix_row_major"], dtype=np.float64
        ).reshape(4, 4)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid declared stereo transform: {path}") from error
    if not np.isfinite(left_T_right).all() or not np.allclose(
        left_T_right[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12
    ):
        raise ValueError("Declared stereo transform is not homogeneous")
    left_R_right = left_T_right[:3, :3]
    if not np.allclose(left_R_right.T @ left_R_right, np.eye(3), atol=1.0e-8):
        raise ValueError("Declared stereo rotation is not orthonormal")
    if not np.isclose(np.linalg.det(left_R_right), 1.0, atol=1.0e-8):
        raise ValueError("Declared stereo rotation is not proper")

    right_T_left = np.linalg.inv(left_T_right)
    right_R_left = right_T_left[:3, :3]
    right_t_left = right_T_left[:3, 3]
    translation_norm = float(np.linalg.norm(right_t_left))
    if not np.isclose(translation_norm, baseline, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            "Declared stereo transform baseline disagrees with camera "
            f"intrinsics: {translation_norm} vs {baseline}"
        )
    if not np.allclose(right_R_left, np.eye(3), atol=1.0e-8) or not np.allclose(
        right_t_left[1:], 0.0, atol=1.0e-9
    ):
        raise ValueError(
            "Stored pinhole_rectified images require identity relative "
            "rotation and an X-only stereo baseline"
        )
    return right_R_left, right_t_left, {
        "mode": "declared_rectified_optical_transform",
        "report_path": str(path.resolve()),
        "report_sha256": sha256_file(path),
        "declared_semantics": transform_record.get("semantics"),
        "declared_left_T_right": left_T_right.tolist(),
        "opencv_right_T_left": right_T_left.tolist(),
        # Keep the established report keys for downstream compatibility.
        "camera0_R_camera1": right_R_left.tolist(),
        "camera0_t_camera1_m": right_t_left.tolist(),
        "camera0_R_camera1_euler_xyz_deg": [0.0, 0.0, 0.0],
        "baseline_direction": (right_t_left / translation_norm).tolist(),
        "right_image_pixels_preserved": False,
        "right_vertical_rectification": {
            "model": "pending_current_capture_vertical_validation",
            "left_image_preserved": True,
            "right_image_preserved": False,
            "source": "authoritative_optical_frame_stereo_calibration",
        },
    }


def image_path(src: Path, record, camera: str) -> Path:
    images = {image["camera"]: image for image in record.get("images", [])}
    if camera not in images:
        raise ValueError(f"Missing {camera} image at tick {record.get('tick')}")
    path = Path(images[camera]["path"])
    return path.resolve() if path.is_absolute() else (src / path).resolve()


def resolve_input_projection_model(
    requested_model: str,
    records,
    source_calibration: dict,
    distortion: np.ndarray,
):
    """Resolve whether stored pixels are raw fisheye or already rectified.

    ROS camera metadata can retain the physical lens model after the driver has
    emitted rectified images.  Feeding those pixels through ``cv2.fisheye`` a
    second time bends straight scene lines even when every D coefficient is
    zero, because the zero-coefficient fisheye projection is still equidistant.
    """

    if requested_model not in {
        "auto",
        "kannala_brandt",
        "pinhole_rectified",
        "pinhole_unrectified",
    }:
        raise ValueError(f"Unsupported input projection model: {requested_model}")
    image_paths = [
        str(image.get("path", ""))
        for record in records
        for image in record.get("images", [])
        if image.get("camera") in {"cam0", "cam1"}
    ]
    paths_are_rectified = bool(image_paths) and all(
        "2d_rect" in Path(path).parts for path in image_paths
    )
    roi_declares_rectified = bool(
        source_calibration.get("roi", {}).get("do_rectify", False)
    )
    zero_distortion = bool(
        np.allclose(np.asarray(distortion), 0.0, rtol=0.0, atol=1.0e-12)
    )
    if requested_model == "auto":
        model = (
            "pinhole_rectified"
            if paths_are_rectified and roi_declares_rectified and zero_distortion
            else "kannala_brandt"
        )
        source = (
            "rectified_path_zero_distortion_and_roi"
            if model == "pinhole_rectified"
            else "declared_lens_model"
        )
    else:
        model = requested_model
        source = "explicit_cli"
    if model in {"pinhole_rectified", "pinhole_unrectified"} and not zero_distortion:
        raise ValueError(
            f"{model} input requires zero stored-image distortion; "
            "non-zero Kannala-Brandt coefficients describe raw fisheye pixels"
        )
    return model, {
        "requested": requested_model,
        "selected": model,
        "selection_source": source,
        "declared_distortion_model": str(
            source_calibration.get("distortion_model", "")
        ),
        "image_paths_are_2d_rect": paths_are_rectified,
        "roi_do_rectify": roi_declares_rectified,
        "zero_distortion": zero_distortion,
    }


def filter_matches_to_pose_coverage(
    matches,
    left_timestamps: np.ndarray,
    base_pose_timestamps,
    camera_pose_timestamps,
):
    """Drop image pairs that would require endpoint-clamped world poses."""

    lower_ns = max(min(base_pose_timestamps), min(camera_pose_timestamps))
    upper_ns = min(max(base_pose_timestamps), max(camera_pose_timestamps))
    retained = []
    dropped = []
    for match in matches:
        timestamp_ns = int(left_timestamps[match[0]])
        if lower_ns <= timestamp_ns <= upper_ns:
            retained.append(match)
        else:
            dropped.append(match)
    if not retained:
        raise ValueError("No stereo pairs fall inside base/camera pose coverage")
    return retained, dropped, int(lower_ns), int(upper_ns)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lidar_guided_right_rectification(
    report_path: Path,
    source_dataset: Path,
    K: np.ndarray,
    width: int,
    height: int,
    baseline: float,
) -> dict:
    """Load a LiDAR-validated right-source to rectified-pixel homography."""

    report_path = report_path.resolve()
    try:
        report = json.loads(report_path.read_text())
        method = report["method"]
        report_source = Path(report["source_dataset"]).resolve()
        geometry = report["camera_geometry"]
        homography = np.asarray(
            report["right_source_to_rectified_homography"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Invalid LiDAR-guided right rectification report: {report_path}"
        ) from error
    if method != "lidar_guided_right_only_projective_rectification":
        raise ValueError(f"Unsupported right rectification method: {method!r}")
    if report_source != source_dataset.resolve():
        raise ValueError(
            "Right rectification report belongs to a different source dataset: "
            f"{report_source} != {source_dataset.resolve()}"
        )
    expected_geometry = {
        "width": width,
        "height": height,
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "baseline": baseline,
    }
    for key, expected in expected_geometry.items():
        actual = geometry.get(key)
        if actual is None or not np.isclose(
            float(actual), float(expected), rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(
                "Right rectification camera geometry mismatch for "
                f"{key}: report={actual!r}, source={expected!r}"
            )
    if (
        report.get("left_image_policy") != "byte-for-byte original left PNG"
        or report.get("extra_left_rotation_applied") is not False
    ):
        raise ValueError(
            "Right rectification report does not guarantee original-left "
            "pixel and orientation preservation"
        )
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("Right rectification homography must be a finite 3x3 matrix")
    determinant = float(np.linalg.det(homography))
    if abs(determinant) < 1.0e-6:
        raise ValueError("Right rectification homography is singular")
    valid_ratio = float(report.get("right_warp_valid_ratio", 0.0))
    if valid_ratio < 0.95:
        raise ValueError(
            "Right rectification leaves too little valid image area: "
            f"{valid_ratio:.6f}"
        )
    return {
        "model": "right_projective_homography",
        "right_source_to_rectified_homography": homography.tolist(),
        "source": "lidar_guided_multiframe_heldout_validation",
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "right_warp_valid_ratio": valid_ratio,
        "training_frames": int(
            report.get("training", {}).get("selected_frames", 0)
        ),
        "training_associations": int(
            report.get("training", {}).get("associations", 0)
        ),
        "holdout_source_index": report.get("holdout_source_index"),
        "left_image_rotation_deg": 0.0,
        "left_image_preserved": True,
        "right_image_preserved": False,
    }


def load_fixed_stereo_calibration(
    report_path: Path,
    K: np.ndarray,
    distortion: np.ndarray,
    baseline: float,
    input_projection_model: str | None = None,
):
    report_path = report_path.resolve()
    try:
        report = json.loads(report_path.read_text())
        calibration = report["estimated_stereo_calibration"]
        report_K = np.asarray(report["source_intrinsics"], dtype=np.float64)
        report_distortion = np.asarray(
            report["source_distortion"], dtype=np.float64
        ).reshape(-1)
        camera0_R_camera1 = np.asarray(
            calibration["camera0_R_camera1"], dtype=np.float64
        )
        camera0_t_camera1 = np.asarray(
            calibration["camera0_t_camera1_m"], dtype=np.float64
        ).reshape(-1)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Invalid fixed stereo calibration report: {report_path}"
        ) from error

    if report_K.shape != (3, 3) or not np.allclose(
        report_K, K, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError("Fixed stereo calibration source intrinsics do not match")
    current_distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if report_distortion.shape != current_distortion.shape or not np.allclose(
        report_distortion, current_distortion, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("Fixed stereo calibration source distortion does not match")
    report_projection_model = report.get(
        "effective_input_projection_model",
        report.get("source_projection_model"),
    )
    if (
        input_projection_model is not None
        and report_projection_model is not None
        and report_projection_model != input_projection_model
    ):
        raise ValueError(
            "Fixed stereo calibration input projection model does not match: "
            f"report={report_projection_model!r}, current={input_projection_model!r}"
        )
    if camera0_R_camera1.shape != (3, 3) or not np.all(
        np.isfinite(camera0_R_camera1)
    ):
        raise ValueError("Fixed stereo calibration rotation must be a finite 3x3 matrix")
    if camera0_t_camera1.shape != (3,) or not np.all(
        np.isfinite(camera0_t_camera1)
    ):
        raise ValueError("Fixed stereo calibration translation must be a finite 3-vector")

    orthogonality_error = float(
        np.linalg.norm(camera0_R_camera1.T @ camera0_R_camera1 - np.eye(3))
    )
    determinant = float(np.linalg.det(camera0_R_camera1))
    if orthogonality_error > 1.0e-6 or abs(determinant - 1.0) > 1.0e-6:
        raise ValueError(
            "Fixed stereo calibration rotation is not in SO(3): "
            f"orthogonality_error={orthogonality_error:.3g}, det={determinant:.9g}"
        )

    translation_norm = float(np.linalg.norm(camera0_t_camera1))
    baseline_tolerance = max(1.0e-6, baseline * 1.0e-4)
    if abs(translation_norm - baseline) > baseline_tolerance:
        raise ValueError(
            "Fixed stereo calibration baseline does not match the source rig: "
            f"report={translation_norm:.9g}m, source={baseline:.9g}m"
        )
    baseline_direction = camera0_t_camera1 / translation_norm
    if abs(float(baseline_direction[0])) < 0.9:
        raise ValueError(
            "Fixed stereo calibration baseline is not predominantly horizontal: "
            f"direction={baseline_direction.tolist()}"
        )

    evidence = {
        "mode": "fixed_report",
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "source_geometry_validation": {
            "intrinsics_match": True,
            "distortion_match": True,
            "input_projection_model_match": (
                report_projection_model in {None, input_projection_model}
            ),
            "source_baseline_m": baseline,
            "report_translation_norm_m": translation_norm,
            "baseline_tolerance_m": baseline_tolerance,
            "rotation_orthogonality_error": orthogonality_error,
            "rotation_determinant": determinant,
            "baseline_direction": baseline_direction.tolist(),
        },
        "camera0_R_camera1": camera0_R_camera1.tolist(),
        "camera0_t_camera1_m": camera0_t_camera1.tolist(),
        "camera0_R_camera1_euler_xyz_deg": Rotation.from_matrix(
            camera0_R_camera1
        ).as_euler("xyz", degrees=True).tolist(),
        "baseline_direction": baseline_direction.tolist(),
    }
    return camera0_R_camera1, camera0_t_camera1, evidence


def normalized_sampson_residuals(
    camera0_R_camera1: np.ndarray,
    camera0_t_camera1: np.ndarray,
    normalized_left: np.ndarray,
    normalized_right: np.ndarray,
) -> np.ndarray:
    translation = camera0_t_camera1 / np.linalg.norm(camera0_t_camera1)
    tx = np.array(
        [
            [0.0, -translation[2], translation[1]],
            [translation[2], 0.0, -translation[0]],
            [-translation[1], translation[0], 0.0],
        ]
    )
    essential = tx @ camera0_R_camera1
    left_h = np.column_stack((normalized_left, np.ones(len(normalized_left))))
    right_h = np.column_stack((normalized_right, np.ones(len(normalized_right))))
    essential_left = (essential @ left_h.T).T
    essential_t_right = (essential.T @ right_h.T).T
    numerator = np.sum(right_h * essential_left, axis=1) ** 2
    denominator = (
        essential_left[:, 0] ** 2
        + essential_left[:, 1] ** 2
        + essential_t_right[:, 0] ** 2
        + essential_t_right[:, 1] ** 2
    )
    return np.sqrt(numerator / np.maximum(denominator, 1.0e-15))


def recover_stereo_pose(
    normalized_left: np.ndarray,
    normalized_right: np.ndarray,
    baseline: float,
):
    essential, inlier_mask = cv2.findEssentialMat(
        normalized_left,
        normalized_right,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.005,
        maxIters=10000,
    )
    if essential is None or inlier_mask is None:
        raise RuntimeError("Essential-matrix estimation failed")
    if essential.shape[0] > 3:
        essential = essential[:3]
    # recoverPose updates its mask argument in place to retain only points with
    # positive depth. Preserve the actual RANSAC count before that call.
    ransac_inliers = int(np.count_nonzero(inlier_mask))
    ransac_ratio = ransac_inliers / len(normalized_left)
    pose_inliers, camera0_R_camera1, translation, _ = cv2.recoverPose(
        essential,
        normalized_left,
        normalized_right,
        np.eye(3),
        mask=inlier_mask.copy(),
    )
    translation = translation.reshape(3)
    translation /= np.linalg.norm(translation)
    camera0_t_camera1 = translation * baseline
    if ransac_ratio < 0.5 or pose_inliers < 1000:
        raise RuntimeError(
            "Stereo calibration is underconstrained: "
            f"RANSAC={ransac_ratio:.3f}, pose_inliers={pose_inliers}"
        )
    if abs(translation[0]) < 0.9:
        raise RuntimeError(
            "Estimated stereo baseline is not predominantly horizontal: "
            f"direction={translation.tolist()}"
        )
    return camera0_R_camera1, camera0_t_camera1, {
        "mode": "estimated_from_current_capture",
        "feature_matches": len(normalized_left),
        "ransac_inliers": ransac_inliers,
        "ransac_inlier_ratio": ransac_ratio,
        "pose_inliers": int(pose_inliers),
        "camera0_R_camera1": camera0_R_camera1.tolist(),
        "camera0_t_camera1_m": camera0_t_camera1.tolist(),
        "camera0_R_camera1_euler_xyz_deg": Rotation.from_matrix(
            camera0_R_camera1
        ).as_euler("xyz", degrees=True).tolist(),
        "baseline_direction": translation.tolist(),
    }


def estimate_rectified_right_vertical_warp(
    pixels_left: np.ndarray,
    pixels_right: np.ndarray,
    K: np.ndarray | None = None,
    camera0_R_camera1: np.ndarray | None = None,
) -> dict:
    """Align an already-rectified right image without rotating the left image.

    The capture stores ``2d_rect`` pixels, so the left image defines the image
    camera used by TF and downstream semantics.  A second stereoRectify call
    would rotate both views.  Instead, fit only the residual vertical mapping
    of the right pixels while keeping their x coordinate unchanged.
    """

    left = np.asarray(pixels_left, dtype=np.float64).reshape(-1, 2)
    right_source = np.asarray(pixels_right, dtype=np.float64).reshape(-1, 2)
    if (K is None) != (camera0_R_camera1 is None):
        raise ValueError("K and camera0_R_camera1 must be provided together")
    if K is None:
        source_to_left_orientation = np.eye(3, dtype=np.float64)
    else:
        K_array = np.asarray(K, dtype=np.float64).reshape(3, 3)
        rotation = np.asarray(camera0_R_camera1, dtype=np.float64).reshape(3, 3)
        # recoverPose returns camera1_R_camera0.  Rotate right-camera rays back
        # into the left-camera orientation before fitting the small remaining
        # vertical alignment.  This preserves the left image while retaining
        # the physically meaningful horizontal disparity in the right view.
        source_to_left_orientation = (
            K_array @ rotation.T @ np.linalg.inv(K_array)
        )
        source_to_left_orientation /= source_to_left_orientation[2, 2]
    right = cv2.perspectiveTransform(
        right_source.reshape(-1, 1, 2), source_to_left_orientation
    ).reshape(-1, 2)
    if len(left) != len(right) or len(left) < 1000:
        raise ValueError("At least 1000 paired pixels are required for rectification")
    vertical_delta = left[:, 1] - right[:, 1]
    design = np.column_stack((right[:, 0], right[:, 1], np.ones(len(right))))
    affine = least_squares(
        lambda values: design @ values - vertical_delta,
        np.zeros(3, dtype=np.float64),
        loss="huber",
        f_scale=1.5,
        max_nfev=100,
    ).x
    initial = np.array(
        [affine[0], 1.0 + affine[1], affine[2], 0.0, 0.0],
        dtype=np.float64,
    )

    def residuals(values: np.ndarray) -> np.ndarray:
        denominator = values[3] * right[:, 0] + values[4] * right[:, 1] + 1.0
        predicted_y = (
            values[0] * right[:, 0]
            + values[1] * right[:, 1]
            + values[2]
        ) / denominator
        return predicted_y - left[:, 1]

    result = least_squares(
        residuals,
        initial,
        loss="huber",
        f_scale=1.5,
        max_nfev=100,
    )
    coefficients = result.x
    residual = residuals(coefficients)
    before = np.abs(vertical_delta)
    after = np.abs(residual)
    after_median = float(np.median(after))
    after_p90 = float(np.percentile(after, 90.0))
    if not result.success or after_median > 0.5 or after_p90 > 2.0:
        raise RuntimeError(
            "Right-only vertical stereo rectification is unreliable: "
            f"success={result.success}, median={after_median:.3f}px, "
            f"p90={after_p90:.3f}px"
        )
    return {
        "model": "right_y_projective_x_preserving",
        "coefficients_a_b_c_g_h": coefficients.tolist(),
        "feature_matches": len(left),
        "before_absolute_vertical_error_px_percentiles": np.percentile(
            before, [0, 25, 50, 75, 90, 95, 99, 100]
        ).tolist(),
        "after_absolute_vertical_error_px_percentiles": np.percentile(
            after, [0, 25, 50, 75, 90, 95, 99, 100]
        ).tolist(),
        "left_image_rotation_deg": 0.0,
        "left_image_preserved": True,
        "right_source_to_left_orientation_homography": (
            source_to_left_orientation.tolist()
        ),
    }


def invert_rectified_right_vertical_warp(
    aligned_x: np.ndarray,
    aligned_y: np.ndarray,
    rectification: dict,
) -> np.ndarray:
    if rectification.get("model") != "right_y_projective_x_preserving":
        raise ValueError("Unsupported right-image vertical rectification model")
    a, b, c, g, h = np.asarray(
        rectification["coefficients_a_b_c_g_h"], dtype=np.float64
    )
    denominator = aligned_y * h - b
    if np.any(np.abs(denominator) < 1.0e-6):
        raise RuntimeError("Right-image vertical rectification is singular")
    return (
        a * aligned_x + c - aligned_y * (g * aligned_x + 1.0)
    ) / denominator


def invert_rectified_right_remap(
    aligned_x: np.ndarray,
    aligned_y: np.ndarray,
    rectification: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Map left-oriented rectified pixels back into the right source image."""

    if rectification.get("model") == "declared_rectified_identity":
        return aligned_x.astype(np.float32), aligned_y.astype(np.float32)
    if rectification.get("model") == "right_projective_homography":
        source_to_rectified = np.asarray(
            rectification["right_source_to_rectified_homography"],
            dtype=np.float64,
        ).reshape(3, 3)
        rectified_to_source = np.linalg.inv(source_to_rectified)
        denominator = (
            rectified_to_source[2, 0] * aligned_x
            + rectified_to_source[2, 1] * aligned_y
            + rectified_to_source[2, 2]
        )
        if np.any(np.abs(denominator) < 1.0e-8):
            raise RuntimeError("Right-image projective rectification is singular")
        source_x = (
            rectified_to_source[0, 0] * aligned_x
            + rectified_to_source[0, 1] * aligned_y
            + rectified_to_source[0, 2]
        ) / denominator
        source_y = (
            rectified_to_source[1, 0] * aligned_x
            + rectified_to_source[1, 1] * aligned_y
            + rectified_to_source[1, 2]
        ) / denominator
        return source_x.astype(np.float32), source_y.astype(np.float32)
    oriented_y = invert_rectified_right_vertical_warp(
        aligned_x, aligned_y, rectification
    )
    source_to_left_orientation = np.asarray(
        rectification.get(
            "right_source_to_left_orientation_homography", np.eye(3)
        ),
        dtype=np.float64,
    ).reshape(3, 3)
    left_orientation_to_source = np.linalg.inv(source_to_left_orientation)
    denominator = (
        left_orientation_to_source[2, 0] * aligned_x
        + left_orientation_to_source[2, 1] * oriented_y
        + left_orientation_to_source[2, 2]
    )
    if np.any(np.abs(denominator) < 1.0e-8):
        raise RuntimeError("Right-image orientation homography is singular")
    source_x = (
        left_orientation_to_source[0, 0] * aligned_x
        + left_orientation_to_source[0, 1] * oriented_y
        + left_orientation_to_source[0, 2]
    ) / denominator
    source_y = (
        left_orientation_to_source[1, 0] * aligned_x
        + left_orientation_to_source[1, 1] * oriented_y
        + left_orientation_to_source[1, 2]
    ) / denominator
    return source_x.astype(np.float32), source_y.astype(np.float32)


def estimate_stereo_extrinsics(
    src: Path,
    records,
    matches,
    K: np.ndarray,
    distortion: np.ndarray,
    baseline: float,
    input_projection_model: str,
    calibration_report: Path | None = None,
    declared_calibration: tuple[np.ndarray, np.ndarray, dict] | None = None,
    sample_count: int = 24,
):
    """Estimate the omitted stereo rotation and baseline direction."""
    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create(nfeatures=5000)
    matcher = cv2.BFMatcher()
    sample_indices = np.linspace(
        0, len(matches) - 1, min(sample_count, len(matches)), dtype=int
    )
    left_points = []
    right_points = []
    center = np.array([K[0, 2], K[1, 2]])
    focal = float((K[0, 0] + K[1, 1]) / 2.0)
    for sample_index in sample_indices:
        left_idx, right_idx, _ = matches[sample_index]
        left = cv2.imread(
            str(image_path(src, records[left_idx], "cam0")), cv2.IMREAD_GRAYSCALE
        )
        right = cv2.imread(
            str(image_path(src, records[right_idx], "cam1")), cv2.IMREAD_GRAYSCALE
        )
        if left is None or right is None:
            raise RuntimeError(f"Failed to read calibration pair {left_idx}/{right_idx}")
        left_keypoints, left_descriptors = sift.detectAndCompute(left, None)
        right_keypoints, right_descriptors = sift.detectAndCompute(right, None)
        if left_descriptors is None or right_descriptors is None:
            continue
        candidates = matcher.knnMatch(
            left_descriptors, right_descriptors, k=2
        )
        good = [
            best
            for best, second in candidates
            if best.distance < 0.65 * second.distance
        ]
        points_left = np.float32(
            [left_keypoints[match.queryIdx].pt for match in good]
        )
        points_right = np.float32(
            [right_keypoints[match.trainIdx].pt for match in good]
        )
        # Rays near the 180-degree fisheye rim become numerically unstable in
        # normalized pinhole coordinates and add little calibration value.
        radius_left = np.linalg.norm((points_left - center) / focal, axis=1)
        radius_right = np.linalg.norm((points_right - center) / focal, axis=1)
        keep = (radius_left < 1.30) & (radius_right < 1.30)
        left_points.append(points_left[keep])
        right_points.append(points_right[keep])

    if not left_points:
        raise RuntimeError("No stereo features found for calibration")
    pixels_left = np.concatenate(left_points).reshape(-1, 1, 2)
    pixels_right = np.concatenate(right_points).reshape(-1, 1, 2)
    if input_projection_model == "kannala_brandt":
        normalized_left = cv2.fisheye.undistortPoints(
            pixels_left, K, distortion
        ).reshape(-1, 2)
        normalized_right = cv2.fisheye.undistortPoints(
            pixels_right, K, distortion
        ).reshape(-1, 2)
    elif input_projection_model in {
        "pinhole_rectified",
        "pinhole_unrectified",
    }:
        zero_distortion = np.zeros(5, dtype=np.float64)
        normalized_left = cv2.undistortPoints(
            pixels_left, K, zero_distortion
        ).reshape(-1, 2)
        normalized_right = cv2.undistortPoints(
            pixels_right, K, zero_distortion
        ).reshape(-1, 2)
    else:
        raise ValueError(
            f"Unsupported input projection model: {input_projection_model}"
        )
    finite = (
        np.all(np.isfinite(normalized_left), axis=1)
        & np.all(np.isfinite(normalized_right), axis=1)
        & (np.linalg.norm(normalized_left, axis=1) < 10.0)
        & (np.linalg.norm(normalized_right, axis=1) < 10.0)
    )
    normalized_left = normalized_left[finite]
    normalized_right = normalized_right[finite]
    if len(normalized_left) < 1000:
        raise RuntimeError(
            f"Too few finite stereo matches: {len(normalized_left)}"
        )

    if calibration_report is not None and declared_calibration is not None:
        raise ValueError(
            "Use either a fixed stereo calibration report or the declared "
            "current-capture calibration, not both"
        )
    if calibration_report is not None:
        camera0_R_camera1, camera0_t_camera1, evidence = (
            load_fixed_stereo_calibration(
                calibration_report,
                K,
                distortion,
                baseline,
                input_projection_model,
            )
        )
    elif declared_calibration is not None:
        camera0_R_camera1, camera0_t_camera1, evidence = declared_calibration
    else:
        camera0_R_camera1, camera0_t_camera1, evidence = recover_stereo_pose(
            normalized_left, normalized_right, baseline
        )
        evidence["sampled_pairs"] = len(sample_indices)

    if calibration_report is not None or declared_calibration is not None:
        residuals = normalized_sampson_residuals(
            camera0_R_camera1,
            camera0_t_camera1,
            normalized_left,
            normalized_right,
        )
        epipolar_threshold = 0.005
        inlier_ratio = float(np.mean(residuals < epipolar_threshold))
        median_residual = float(np.median(residuals))
        p95_residual = float(np.percentile(residuals, 95.0))
        strict_passed = (
            inlier_ratio >= 0.70
            and median_residual <= 0.0025
            and p95_residual <= 0.04
        )
        evidence["current_capture_epipolar_validation"] = {
            "sampled_pairs": len(sample_indices),
            "feature_matches": len(normalized_left),
            "normalized_sampson_threshold": epipolar_threshold,
            "inlier_ratio": inlier_ratio,
            "median_residual": median_residual,
            "p95_residual": p95_residual,
            "required_inlier_ratio": 0.70,
            "maximum_median_residual": 0.0025,
            "maximum_p95_residual": 0.04,
            "strict_empirical_thresholds_passed": strict_passed,
        }
        if declared_calibration is not None and not strict_passed:
            declared_evidence = evidence
            camera0_R_camera1, camera0_t_camera1, evidence = (
                recover_stereo_pose(
                    normalized_left,
                    normalized_right,
                    baseline,
                )
            )
            evidence["mode"] = (
                "estimated_from_current_capture_after_declared_validation_failure"
            )
            evidence["sampled_pairs"] = len(sample_indices)
            evidence["rejected_declared_calibration"] = declared_evidence
            declared_calibration = None
        elif not strict_passed:
            raise RuntimeError(
                "Fixed stereo calibration fails current-capture epipolar validation: "
                f"inlier_ratio={inlier_ratio:.3f}, median={median_residual:.6f}, "
                f"p95={p95_residual:.6f}"
            )
        if input_projection_model == "pinhole_rectified":
            evidence["right_vertical_rectification"] = (
                estimate_rectified_right_vertical_warp(
                    pixels_left,
                    pixels_right,
                    *(
                        (K, camera0_R_camera1)
                        if declared_calibration is None
                        else ()
                    ),
                )
            )
            if declared_calibration is not None:
                evidence["right_vertical_rectification"].update(
                    {
                        "source": (
                            "declared_identity_rotation_plus_current_capture_"
                            "vertical_residual"
                        ),
                        "horizontal_pixel_mapping": "identity",
                        "right_image_preserved": False,
                    }
                )
        return camera0_R_camera1, camera0_t_camera1, evidence

    if input_projection_model == "pinhole_rectified":
        evidence["right_vertical_rectification"] = (
            estimate_rectified_right_vertical_warp(
                pixels_left,
                pixels_right,
                K,
                camera0_R_camera1,
            )
        )
    return camera0_R_camera1, camera0_t_camera1, evidence


def camera_quaternion_matrices(aux_records, order: str) -> np.ndarray:
    matrices = []
    for record in aux_records:
        values = record["poses"]["head_camera"]["orientation_xyzw"]
        if order == "wxyz":
            values = [values[1], values[2], values[3], values[0]]
        matrices.append(Rotation.from_quat(values).as_matrix())
    return np.asarray(matrices)


def camera_quaternion_level_error(aux_records, order: str) -> float:
    base_R_camera = camera_quaternion_matrices(aux_records, order)
    # For an optical frame on a level robot, camera +Y (image down) should be
    # close to base -Z. Yaw motion does not affect this score.
    down_axes = base_R_camera[:, :, 1]
    base_down = np.array([0.0, 0.0, -1.0])
    return float(np.median(np.linalg.norm(down_axes - base_down, axis=1)))


def load_recorded_camera_rotation_matrices(
    source_dataset: Path,
    sequence: str,
    expected_count: int,
) -> np.ndarray | None:
    path = (
        source_dataset
        / "poses"
        / "dense_global"
        / sequence
        / "poses.txt"
    )
    if not path.is_file():
        return None
    matrices = np.loadtxt(path, dtype=np.float64)
    if matrices.ndim == 1:
        matrices = matrices[None, :]
    if matrices.shape != (expected_count, 16):
        raise ValueError(
            "Recorded head-camera pose matrix count/shape disagrees with "
            f"aux_poses.jsonl: {matrices.shape} vs ({expected_count}, 16)"
        )
    matrices = matrices.reshape(-1, 4, 4)
    if not np.isfinite(matrices).all():
        raise ValueError("Recorded head-camera pose matrices contain non-finite values")
    if not np.allclose(
        matrices[:, 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=1.0e-9,
    ):
        raise ValueError("Recorded head-camera poses are not homogeneous matrices")
    rotations = matrices[:, :3, :3]
    orthogonality = np.matmul(
        rotations.transpose(0, 2, 1),
        rotations,
    )
    if not np.allclose(orthogonality, np.eye(3), atol=1.0e-8):
        raise ValueError("Recorded head-camera rotations are not orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1.0e-8):
        raise ValueError("Recorded head-camera rotations are not proper")
    return rotations


def camera_quaternion_reference_errors_deg(
    aux_records,
    order: str,
    recorded_rotations: np.ndarray,
) -> np.ndarray:
    candidate_rotations = camera_quaternion_matrices(aux_records, order)
    if candidate_rotations.shape != recorded_rotations.shape:
        raise ValueError("Quaternion and recorded rotation counts differ")
    residual_rotations = np.matmul(
        recorded_rotations.transpose(0, 2, 1),
        candidate_rotations,
    )
    return np.rad2deg(
        Rotation.from_matrix(residual_rotations).magnitude()
    )


def resolve_camera_quaternion_order(
    aux_records,
    requested_order: str,
    recorded_rotations: np.ndarray | None = None,
):
    level_errors = {
        order: camera_quaternion_level_error(aux_records, order)
        for order in ("xyzw", "wxyz")
    }
    reference_errors = None
    if recorded_rotations is not None:
        reference_errors = {}
        for order in ("xyzw", "wxyz"):
            values = camera_quaternion_reference_errors_deg(
                aux_records,
                order,
                recorded_rotations,
            )
            reference_errors[order] = {
                "count": int(values.size),
                "minimum": float(np.min(values)),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "p99": float(np.percentile(values, 99)),
                "maximum": float(np.max(values)),
            }

    if requested_order == "auto":
        if reference_errors is None:
            raise ValueError(
                "Automatic camera quaternion-order selection requires the "
                "recorded head-camera poses.txt rotation matrices. Pass an "
                "explicit order only when the capture contract independently "
                "proves it."
            )
        order = min(
            reference_errors,
            key=lambda candidate: reference_errors[candidate]["median"],
        )
    else:
        order = requested_order

    if (
        reference_errors is not None
        and reference_errors[order]["maximum"] > 1.0e-5
    ):
        raise ValueError(
            "Selected camera quaternion order disagrees with the capture's "
            "recorded pose matrices: "
            f"selected={order}, reference_errors_deg={reference_errors}"
        )
    return order, level_errors, reference_errors


def build_virtual_camera(
    K: np.ndarray,
    distortion: np.ndarray,
    width: int,
    height: int,
    horizontal_fov_deg: float,
    down_fov_deg: float,
    optical_x_rotation_deg: float,
    camera0_R_camera1: np.ndarray,
    camera0_t_camera1: np.ndarray,
    input_projection_model: str = "kannala_brandt",
    right_vertical_rectification: dict | None = None,
):
    if input_projection_model == "kannala_brandt":
        if not 30.0 <= horizontal_fov_deg < 170.0:
            raise ValueError("--horizontal-fov-deg must be in [30, 170)")
        if not 5.0 <= down_fov_deg < 80.0:
            raise ValueError("--down-fov-deg must be in [5, 80)")
        focal = width / (2.0 * np.tan(np.deg2rad(horizontal_fov_deg / 2.0)))
        cx = width / 2.0
        cy = (height - 1) - focal * np.tan(np.deg2rad(down_fov_deg))
        if not 0.0 < cy < height:
            raise ValueError(
                "Requested FOV places the virtual principal point outside the image: "
                f"cy={cy:.3f}"
            )
        virtual_K = np.array(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        stereo_R_left, stereo_R_right, _, _, _ = cv2.fisheye.stereoRectify(
            K,
            distortion,
            K,
            distortion,
            (width, height),
            camera0_R_camera1,
            camera0_t_camera1,
            flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=(width, height),
            balance=0.0,
            fov_scale=1.0,
        )
    elif input_projection_model == "pinhole_unrectified":
        if abs(optical_x_rotation_deg) >= 1.0e-12:
            raise ValueError(
                "pinhole_unrectified currently requires zero "
                "--rectification-roll-deg"
            )
        zero_distortion = np.zeros(5, dtype=np.float64)
        (
            stereo_R_left,
            stereo_R_right,
            projection_left,
            _,
            _,
            _,
            _,
        ) = cv2.stereoRectify(
            K,
            zero_distortion,
            K,
            zero_distortion,
            (width, height),
            camera0_R_camera1,
            camera0_t_camera1,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
            newImageSize=(width, height),
        )
        virtual_K = projection_left[:3, :3].copy()
    elif input_projection_model == "pinhole_rectified":
        if abs(optical_x_rotation_deg) >= 1.0e-12:
            raise ValueError(
                "pinhole_rectified input preserves the original left image; "
                "--rectification-roll-deg must be zero"
            )
        if right_vertical_rectification is None:
            raise ValueError(
                "pinhole_rectified input requires right-only "
                "rectification evidence"
            )
        # The stored left pixels already define the image camera used by TF and
        # downstream semantics. Preserve both the pixel grid and its calibrated
        # intrinsics exactly. FOV arguments only apply when converting raw
        # Kannala-Brandt pixels into a virtual pinhole camera.
        virtual_K = np.asarray(K, dtype=np.float64).copy()
        stereo_R_left = np.eye(3, dtype=np.float64)
        stereo_R_right = np.eye(3, dtype=np.float64)
    else:
        raise ValueError(
            f"Unsupported input projection model: {input_projection_model}"
        )
    if input_projection_model == "kannala_brandt":
        # Apply a common rotation about the rectified optical X axis. It keeps
        # the virtual baseline horizontal while selecting the useful view.
        level_rotation = Rotation.from_euler(
            "x", optical_x_rotation_deg, degrees=True
        ).as_matrix()
        original_R_virtual_left = level_rotation @ stereo_R_left
        original_R_virtual_right = level_rotation @ stereo_R_right
        maps = [
            cv2.fisheye.initUndistortRectifyMap(
                K,
                distortion,
                rotation,
                virtual_K,
                (width, height),
                cv2.CV_32FC1,
            )
            for rotation in (original_R_virtual_left, original_R_virtual_right)
        ]
    elif input_projection_model == "pinhole_unrectified":
        original_R_virtual_left = stereo_R_left
        original_R_virtual_right = stereo_R_right
        zero_distortion = np.zeros(5, dtype=np.float64)
        maps = [
            cv2.initUndistortRectifyMap(
                K,
                zero_distortion,
                rotation,
                virtual_K,
                (width, height),
                cv2.CV_32FC1,
            )
            for rotation in (
                original_R_virtual_left,
                original_R_virtual_right,
            )
        ]
    else:
        original_R_virtual_left = np.eye(3, dtype=np.float64)
        original_R_virtual_right = np.eye(3, dtype=np.float64)
        aligned_x, aligned_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        maps = [
            (aligned_x, aligned_y),
            invert_rectified_right_remap(
                aligned_x,
                aligned_y,
                right_vertical_rectification,
            ),
        ]
    valid_ratios = []
    for map_x, map_y in maps:
        valid = (
            (map_x >= 0.0)
            & (map_x <= width - 1)
            & (map_y >= 0.0)
            & (map_y <= height - 1)
        )
        valid_ratios.append(float(valid.mean()))
    if input_projection_model == "kannala_brandt" and min(valid_ratios) < 0.995:
        raise ValueError(
            "Virtual view extends outside the fisheye images: "
            f"valid_ratios={valid_ratios}. Reduce the requested FOV."
        )
    if (
        input_projection_model == "pinhole_unrectified"
        and min(valid_ratios) < 0.95
    ):
        raise ValueError(
            "Pinhole stereo rectification leaves too little valid image area: "
            f"valid_ratios={valid_ratios}"
        )
    if input_projection_model == "pinhole_rectified":
        if valid_ratios[0] < 1.0:
            raise ValueError(
                "Original-left preservation produced an invalid left pixel map"
            )
        if valid_ratios[1] < 0.95:
            raise ValueError(
                "Right-only rectification leaves too little valid image area: "
                f"valid_ratio={valid_ratios[1]:.6f}"
            )
    return (
        virtual_K,
        original_R_virtual_left,
        original_R_virtual_right,
        maps[0],
        maps[1],
        valid_ratios,
    )


def prepare_output_directory(output: Path, overwrite: bool):
    image_dirs = (output / "rgb", output / "stereo_right")
    existing = [path for directory in image_dirs for path in directory.glob("*.png")]
    if existing and not overwrite:
        raise RuntimeError(
            f"Output already contains rectified images: {output}. Use --overwrite."
        )
    for directory in (*image_dirs, output / "depth", output / "pose"):
        directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for directory in (*image_dirs, output / "depth"):
            for path in directory.glob("*.png"):
                path.unlink()
        for path in (
            output / "foundation_stereo_run.json",
            output / "floor_geometry_calibration.json",
            output / "pose" / "poses_before_floor_calibration.txt",
        ):
            path.unlink(missing_ok=True)


def main():
    args = parse_args()
    src = args.src.resolve()
    output = args.output.resolve()
    calibration_source = (
        args.calibration_source.resolve()
        if args.calibration_source is not None
        else src
    )

    manifest = json.loads((src / "manifest.json").read_text())
    layout = manifest.get("layout_version") or manifest.get("layout")
    if layout != "capture4daaam_like":
        raise ValueError(f"Expected capture4daaam_like G1 data, found {layout!r}")
    quality = json.loads((src / "quality_report.json").read_text())
    if not quality.get("alignment", {}).get("ok"):
        raise ValueError("G1 quality report says the sequence is not aligned")

    records = load_jsonl(src / "manifest.jsonl")
    left_ts = camera_timestamps(records, "cam0")
    right_ts = camera_timestamps(records, "cam1")
    threshold_ns = int(round(args.max_delta_ms * 1.0e6))
    matches, skipped_left, skipped_right = monotonic_matches(
        left_ts, right_ts, threshold_ns
    )
    if not matches:
        raise ValueError("No synchronized stereo pairs found")
    requested_source_indices = None
    requested_unmatched_source_indices: list[int] = []
    if args.source_indices is not None:
        requested_source_indices = set(args.source_indices)
        if (
            not requested_source_indices
            or len(requested_source_indices) != len(args.source_indices)
            or min(requested_source_indices) < 0
        ):
            raise ValueError("source-indices must be unique non-negative values")
        available_source_indices = {int(record["tick"]) for record in records}
        missing_source_indices = requested_source_indices - available_source_indices
        if missing_source_indices:
            raise ValueError(
                f"Requested source frames are absent: {sorted(missing_source_indices)}"
            )
        matches = [
            match
            for match in matches
            if int(records[match[0]]["tick"]) in requested_source_indices
        ]
        matched_source_indices = {
            int(records[match[0]]["tick"]) for match in matches
        }
        requested_unmatched_source_indices = sorted(
            requested_source_indices - matched_source_indices
        )

    K, distortion, width, height, baseline, source_calibration = load_calibration(
        calibration_source, args.sequence
    )
    input_projection_model, input_projection_evidence = (
        resolve_input_projection_model(
            args.input_projection_model,
            records,
            source_calibration,
            distortion,
        )
    )
    declared_calibration = None
    if input_projection_model == "pinhole_rectified":
        declared_calibration = load_declared_rectified_stereo_calibration(
            calibration_source,
            args.sequence,
            baseline,
        )
    camera0_R_camera1, camera0_t_camera1, stereo_calibration = (
        estimate_stereo_extrinsics(
            src,
            records,
            matches,
            K,
            distortion,
            baseline,
            input_projection_model,
            args.stereo_calibration_report,
            declared_calibration,
        )
    )
    if args.right_rectification_report is not None:
        if input_projection_model != "pinhole_rectified":
            raise ValueError(
                "--right-rectification-report requires pinhole_rectified input"
            )
        stereo_calibration["visual_epipolar_rectification_replaced"] = (
            stereo_calibration.get("right_vertical_rectification")
        )
        stereo_calibration["right_vertical_rectification"] = (
            load_lidar_guided_right_rectification(
                args.right_rectification_report,
                src,
                K,
                width,
                height,
                baseline,
            )
        )
        stereo_calibration["right_rectification_selection"] = (
            "explicit_lidar_validated_report"
        )
    aux_records = load_jsonl(
        src / "poses" / "dense_global" / args.sequence / "aux_poses.jsonl"
    )
    recorded_camera_rotations = load_recorded_camera_rotation_matrices(
        src,
        args.sequence,
        len(aux_records),
    )
    (
        camera_quaternion_order,
        quaternion_level_errors,
        quaternion_reference_errors_deg,
    ) = (
        resolve_camera_quaternion_order(
            aux_records,
            args.camera_quaternion_order,
            recorded_camera_rotations,
        )
    )
    optical_x_rotation_deg = args.rectification_roll_deg

    base_pose_path = (
        src
        / "state"
        / args.sequence
        / ("map_pose.jsonl" if args.base_pose_source == "map" else "odom.jsonl")
    )
    base_pose_records = load_jsonl(base_pose_path)
    base_pose_samples = [
        (map_pose_sample(record) if args.base_pose_source == "map" else odom_pose_sample(record))
        for record in base_pose_records
    ]
    camera_pose_timestamps = [
        int(record["poses"]["head_camera"]["timestamp_ns"])
        for record in aux_records
    ]
    matches, pose_coverage_dropped, pose_coverage_start_ns, pose_coverage_end_ns = (
        filter_matches_to_pose_coverage(
            matches,
            left_ts,
            [sample[0] for sample in base_pose_samples],
            camera_pose_timestamps,
        )
    )

    (
        virtual_K,
        original_R_virtual_left,
        original_R_virtual_right,
        left_maps,
        right_maps,
        valid_ratios,
    ) = (
        build_virtual_camera(
            K,
            distortion,
            width,
            height,
            args.horizontal_fov_deg,
            args.down_fov_deg,
            optical_x_rotation_deg,
            camera0_R_camera1,
            camera0_t_camera1,
            input_projection_model,
            stereo_calibration.get("right_vertical_rectification"),
        )
    )
    prepare_output_directory(output, args.overwrite)

    selected_timestamps = left_ts[[left for left, _, _ in matches]]
    global_poses, base_pose_clamped, camera_clamped = compose_global_camera_poses(
        src,
        selected_timestamps,
        camera_quaternion_order=camera_quaternion_order,
        base_pose_source=args.base_pose_source,
        sequence=args.sequence,
    )
    for pose in global_poses:
        pose[:3, :3] = pose[:3, :3] @ original_R_virtual_left.T

    origin_ns = int(selected_timestamps[0])
    output_frames = []
    for output_idx, ((left_idx, right_idx, delta_ns), pose) in enumerate(
        zip(matches, global_poses)
    ):
        left_path = image_path(src, records[left_idx], "cam0")
        right_path = image_path(src, records[right_idx], "cam1")
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise RuntimeError(f"Failed to read stereo pair {left_idx}/{right_idx}")
        if left.shape[:2] != (height, width) or right.shape[:2] != (height, width):
            raise ValueError(
                f"Unexpected image size at pair {left_idx}/{right_idx}: "
                f"{left.shape[:2]} / {right.shape[:2]}"
            )

        left_output = output / "rgb" / f"{output_idx:08d}.png"
        right_output = output / "stereo_right" / f"{output_idx:08d}.png"
        write_options = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        if input_projection_model == "pinhole_rectified":
            # Copy the encoded source PNG instead of decoding and re-encoding
            # it. This makes preservation byte-for-byte auditable.
            if left_path.suffix.lower() != ".png":
                raise ValueError(
                    "pinhole_rectified left-image preservation requires PNG input: "
                    f"{left_path}"
                )
            with left_path.open("rb") as stream:
                if stream.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                    raise ValueError(f"Source image is not an encoded PNG: {left_path}")
            shutil.copyfile(left_path, left_output)
        else:
            virtual_left = cv2.remap(
                left,
                left_maps[0],
                left_maps[1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            if not cv2.imwrite(str(left_output), virtual_left, write_options):
                raise RuntimeError(f"Failed to write {left_output}")
        if stereo_calibration.get("right_image_pixels_preserved", False):
            if right_path.suffix.lower() != ".png":
                raise ValueError(
                    "Declared rectified right-image preservation requires PNG "
                    f"input: {right_path}"
                )
            with right_path.open("rb") as stream:
                if stream.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                    raise ValueError(
                        f"Source image is not an encoded PNG: {right_path}"
                    )
            shutil.copyfile(right_path, right_output)
        else:
            virtual_right = cv2.remap(
                right,
                right_maps[0],
                right_maps[1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            if not cv2.imwrite(str(right_output), virtual_right, write_options):
                raise RuntimeError(f"Failed to write {right_output}")

        output_frames.append(
            {
                "idx": output_idx,
                "source_idx": left_idx,
                "cam0_source_idx": left_idx,
                "cam1_source_idx": right_idx,
                "pose_row": output_idx,
                "cam0": str(left_output),
                "cam1": str(right_output),
                "timestamp": (int(left_ts[left_idx]) - origin_ns) / 1.0e9,
                "cam0_sensor_time_ns": int(left_ts[left_idx]),
                "cam1_sensor_time_ns": int(right_ts[right_idx]),
                "sensor_time_ns": int(left_ts[left_idx]),
                "pose_sensor_time_ns": int(left_ts[left_idx]),
                "stereo_delta_ms": delta_ns / 1.0e6,
            }
        )
        if (output_idx + 1) % 100 == 0:
            print(f"Rectified {output_idx + 1}/{len(matches)} stereo pairs", flush=True)

    pose_text = "".join(
        " ".join(f"{value:.12g}" for value in pose.reshape(-1)) + "\n"
        for pose in global_poses
    )
    (output / "pose" / "poses.txt").write_text(pose_text)
    (output / "pose" / "pose_timestamps_ns.txt").write_text(
        "".join(f"{int(timestamp)}\n" for timestamp in selected_timestamps)
    )

    virtual_K_list = virtual_K.tolist()
    camera_info = {
        "width": width,
        "height": height,
        "model": "pinhole",
        "intrinsics": virtual_K_list,
        "distortion": [0.0, 0.0, 0.0, 0.0],
        "fx": float(virtual_K[0, 0]),
        "fy": float(virtual_K[1, 1]),
        "cx": float(virtual_K[0, 2]),
        "cy": float(virtual_K[1, 2]),
        "baseline": baseline,
    }
    (output / "camera_info.json").write_text(
        json.dumps(camera_info, indent=2) + "\n"
    )
    tick_index = {
        "source": str(src),
        "source_layout": layout,
        "sequence": args.sequence,
        "projection_model": "pinhole",
        "pose_frame": args.base_pose_source,
        "base_pose_source": args.base_pose_source,
        "camera_quaternion_order": camera_quaternion_order,
        "pose_composition": (
            f"{args.base_pose_source}_T_base_link @ base_link_T_head_camera "
            "@ original_camera_T_virtual_camera"
        ),
        "fx": camera_info["fx"],
        "fy": camera_info["fy"],
        "cx": camera_info["cx"],
        "cy": camera_info["cy"],
        "baseline": baseline,
        "width": width,
        "height": height,
        "recommended_max_depth_m": args.recommended_max_depth_m,
        "time_origin_ns": origin_ns,
        "timebase": {
            "clock": "sensor_time_ns",
            "unit": "ns",
            "timestamp_definition": "(sensor_time_ns - time_origin_ns) / 1e9",
        },
        "pose_time_alignment": {
            "method": (
                f"interpolate_{args.base_pose_source}_base_and_head_camera_at_"
                "cam0_sensor_time_ns"
            ),
            "pose_timestamp_file": "pose/pose_timestamps_ns.txt",
            "pose_row_field": "pose_row",
        },
        "frames": output_frames,
    }
    (output / "tick_index.json").write_text(
        json.dumps(tick_index, indent=2) + "\n"
    )
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    translations = np.asarray([pose[:3, 3] for pose in global_poses])
    virtual_forward_z = np.asarray([pose[2, 2] for pose in global_poses])
    virtual_down_z = np.asarray([pose[2, 1] for pose in global_poses])
    effective_horizontal_fov_deg = float(
        np.degrees(
            np.arctan2(virtual_K[0, 2], virtual_K[0, 0])
            + np.arctan2(
                (width - 1) - virtual_K[0, 2],
                virtual_K[0, 0],
            )
        )
    )
    effective_down_fov_deg = float(
        np.degrees(
            np.arctan2(
                (height - 1) - virtual_K[1, 2],
                virtual_K[1, 1],
            )
        )
    )
    left_pixels_preserved = input_projection_model == "pinhole_rectified"
    right_pixels_preserved = bool(
        stereo_calibration.get("right_image_pixels_preserved", False)
    )
    report = {
        "preparation_contract_version": PREPARATION_CONTRACT_VERSION,
        "source_dataset": str(src),
        "calibration_source_dataset": str(calibration_source),
        "base_pose_source": args.base_pose_source,
        "base_pose_file": str(
            base_pose_path
        ),
        "source_projection_model": input_projection_model,
        "declared_source_projection_model": source_calibration[
            "distortion_model"
        ],
        "effective_input_projection_model": input_projection_model,
        "stereo_rectification_policy": (
            "preserve_declared_rectified_left_and_right_identity"
            if right_pixels_preserved
            else "preserve_left_identity_right_only_epipolar_remap"
            if input_projection_model == "pinhole_rectified"
            else "opencv_pinhole_stereo_rectify_from_empirical_extrinsics"
            if input_projection_model == "pinhole_unrectified"
            else "opencv_fisheye_stereo_rectify"
        ),
        "output_intrinsics_source": (
            "source_rectified_K"
            if input_projection_model == "pinhole_rectified"
            else "opencv_pinhole_stereo_rectify_projection"
            if input_projection_model == "pinhole_unrectified"
            else "virtual_pinhole_from_requested_fov"
        ),
        "left_image_orientation_preserved": (
            input_projection_model == "pinhole_rectified"
            and abs(optical_x_rotation_deg) < 1.0e-12
        ),
        "left_image_pixels_preserved": left_pixels_preserved,
        "right_image_pixels_preserved": right_pixels_preserved,
        "left_image_storage_policy": (
            "byte_for_byte_source_png_copy"
            if left_pixels_preserved
            else "opencv_remap_and_png_encode"
        ),
        "right_image_storage_policy": (
            "byte_for_byte_source_png_copy"
            if right_pixels_preserved
            else "opencv_remap_and_png_encode"
        ),
        "input_projection_model_evidence": input_projection_evidence,
        "source_intrinsics": K.tolist(),
        "source_distortion": distortion.reshape(-1).tolist(),
        "virtual_projection_model": "pinhole",
        "virtual_intrinsics": virtual_K_list,
        "estimated_stereo_calibration": stereo_calibration,
        "original_camera_R_virtual_camera": original_R_virtual_left.T.tolist(),
        "opencv_left_original_to_virtual_R": original_R_virtual_left.tolist(),
        "opencv_right_original_to_virtual_R": original_R_virtual_right.tolist(),
        "camera_quaternion_order": camera_quaternion_order,
        "camera_quaternion_level_errors": quaternion_level_errors,
        "camera_quaternion_reference_errors_deg": (
            quaternion_reference_errors_deg
        ),
        "applied_optical_x_rotation_deg": optical_x_rotation_deg,
        "horizontal_fov_deg": effective_horizontal_fov_deg,
        "down_fov_deg": effective_down_fov_deg,
        "requested_horizontal_fov_deg": args.horizontal_fov_deg,
        "requested_down_fov_deg": args.down_fov_deg,
        "remap_valid_ratios": valid_ratios,
        "source_pairs": len(records),
        "requested_source_indices": (
            sorted(requested_source_indices)
            if requested_source_indices is not None
            else None
        ),
        "requested_unmatched_source_indices": requested_unmatched_source_indices,
        "matched_pairs": len(matches),
        "skipped_cam0": len(skipped_left),
        "skipped_cam1": len(skipped_right),
        "max_matched_delta_ms": max(delta for _, _, delta in matches) / 1.0e6,
        "pose_coverage_start_ns": pose_coverage_start_ns,
        "pose_coverage_end_ns": pose_coverage_end_ns,
        "pose_coverage_skipped_pairs": len(pose_coverage_dropped),
        "pose_coverage_skipped_cam0_indices": [
            int(left) for left, _, _ in pose_coverage_dropped
        ],
        "base_pose_interpolation_clamped": base_pose_clamped,
        "odom_interpolation_clamped": (
            base_pose_clamped if args.base_pose_source == "odom" else None
        ),
        "map_pose_interpolation_clamped": (
            base_pose_clamped if args.base_pose_source == "map" else None
        ),
        "camera_pose_interpolation_clamped": camera_clamped,
        "global_translation_first_m": translations[0].tolist(),
        "global_translation_last_m": translations[-1].tolist(),
        "global_translation_path_length_m": float(
            np.linalg.norm(np.diff(translations, axis=0), axis=1).sum()
        ),
        "virtual_forward_world_z_median": float(np.median(virtual_forward_z)),
        "virtual_down_world_z_median": float(np.median(virtual_down_z)),
        "recommended_max_depth_m": args.recommended_max_depth_m,
        "skipped_cam0_indices": skipped_left,
        "skipped_cam1_indices": skipped_right,
    }
    (output / "pinhole_preparation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        f"Prepared {len(matches)}/{len(records)} pinhole stereo pairs at {output}\n"
        f"  K: fx={camera_info['fx']:.3f} fy={camera_info['fy']:.3f} "
        f"cx={camera_info['cx']:.3f} cy={camera_info['cy']:.3f}\n"
        f"  camera quaternion={camera_quaternion_order}, "
        f"optical-X correction={optical_x_rotation_deg:.3f} deg, "
        f"remap valid={min(valid_ratios):.3f}, "
        f"virtual forward/down world-z median="
        f"{np.median(virtual_forward_z):.4f}/{np.median(virtual_down_z):.4f}\n"
        f"  trajectory={report['global_translation_path_length_m']:.3f}m, "
        f"recommended max depth={args.recommended_max_depth_m:.2f}m"
    )


if __name__ == "__main__":
    main()
