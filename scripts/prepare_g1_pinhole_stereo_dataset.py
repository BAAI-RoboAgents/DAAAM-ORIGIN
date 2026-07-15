#!/usr/bin/env python3
"""Reproject synchronized G1 fisheye stereo into a pinhole RGB-D dataset."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from build_synchronized_stereo_dataset import (
    camera_timestamps,
    compose_global_camera_poses,
    load_jsonl,
    monotonic_matches,
)


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
    parser.add_argument("--max-delta-ms", type=float, default=10.0)
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
            "Storage order of head_camera orientation values. Auto selects "
            "the interpretation whose optical down axis best matches base -Z."
        ),
    )
    parser.add_argument("--recommended-max-depth-m", type=float, default=3.0)
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


def image_path(src: Path, record, camera: str) -> Path:
    images = {image["camera"]: image for image in record.get("images", [])}
    if camera not in images:
        raise ValueError(f"Missing {camera} image at tick {record.get('tick')}")
    path = Path(images[camera]["path"])
    return path.resolve() if path.is_absolute() else (src / path).resolve()


def estimate_stereo_extrinsics(
    src: Path,
    records,
    matches,
    K: np.ndarray,
    distortion: np.ndarray,
    baseline: float,
    sample_count: int = 24,
):
    """Estimate the omitted fisheye stereo rotation and baseline direction."""
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
    normalized_left = cv2.fisheye.undistortPoints(
        pixels_left, K, distortion
    ).reshape(-1, 2)
    normalized_right = cv2.fisheye.undistortPoints(
        pixels_right, K, distortion
    ).reshape(-1, 2)
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

    essential, inlier_mask = cv2.findEssentialMat(
        normalized_left,
        normalized_right,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.005,
        maxIters=10000,
    )
    if essential is None:
        raise RuntimeError("Essential-matrix estimation failed")
    if essential.shape[0] > 3:
        essential = essential[:3]
    pose_inliers, camera0_R_camera1, translation, pose_mask = cv2.recoverPose(
        essential,
        normalized_left,
        normalized_right,
        np.eye(3),
        mask=inlier_mask,
    )
    translation = translation.reshape(3)
    translation /= np.linalg.norm(translation)
    camera0_t_camera1 = translation * baseline
    ransac_inliers = int(np.count_nonzero(inlier_mask))
    ransac_ratio = ransac_inliers / len(normalized_left)
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
        "sampled_pairs": len(sample_indices),
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


def camera_quaternion_level_error(aux_records, order: str) -> float:
    matrices = []
    for record in aux_records:
        values = record["poses"]["head_camera"]["orientation_xyzw"]
        if order == "wxyz":
            values = [values[1], values[2], values[3], values[0]]
        matrices.append(Rotation.from_quat(values).as_matrix())
    base_R_camera = np.asarray(matrices)
    # For an optical frame on a level robot, camera +Y (image down) should be
    # close to base -Z. Yaw motion does not affect this score.
    down_axes = base_R_camera[:, :, 1]
    base_down = np.array([0.0, 0.0, -1.0])
    return float(np.median(np.linalg.norm(down_axes - base_down, axis=1)))


def resolve_camera_quaternion_order(aux_records, requested_order: str):
    errors = {
        order: camera_quaternion_level_error(aux_records, order)
        for order in ("xyzw", "wxyz")
    }
    order = min(errors, key=errors.get) if requested_order == "auto" else requested_order
    if errors[order] > 0.35:
        raise ValueError(
            "Neither camera quaternion interpretation produces a plausible "
            f"optical frame: selected={order}, level_errors={errors}"
        )
    return order, errors


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
):
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
    # Apply a common rotation about the rectified optical X axis. It keeps the
    # virtual baseline horizontal while selecting the useful forward/floor view.
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
    valid_ratios = []
    for map_x, map_y in maps:
        valid = (
            (map_x >= 0.0)
            & (map_x <= width - 1)
            & (map_y >= 0.0)
            & (map_y <= height - 1)
        )
        valid_ratios.append(float(valid.mean()))
    if min(valid_ratios) < 0.995:
        raise ValueError(
            "Virtual view extends outside the fisheye images: "
            f"valid_ratios={valid_ratios}. Reduce the requested FOV."
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

    K, distortion, width, height, baseline, source_calibration = load_calibration(
        src, args.sequence
    )
    camera0_R_camera1, camera0_t_camera1, stereo_calibration = (
        estimate_stereo_extrinsics(
            src, records, matches, K, distortion, baseline
        )
    )
    aux_records = load_jsonl(
        src / "poses" / "dense_global" / args.sequence / "aux_poses.jsonl"
    )
    camera_quaternion_order, quaternion_level_errors = (
        resolve_camera_quaternion_order(
            aux_records, args.camera_quaternion_order
        )
    )
    optical_x_rotation_deg = args.rectification_roll_deg

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
        )
    )
    prepare_output_directory(output, args.overwrite)

    selected_timestamps = left_ts[[left for left, _, _ in matches]]
    global_poses, odom_clamped, camera_clamped = compose_global_camera_poses(
        src,
        selected_timestamps,
        camera_quaternion_order=camera_quaternion_order,
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

        virtual_left = cv2.remap(
            left,
            left_maps[0],
            left_maps[1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        virtual_right = cv2.remap(
            right,
            right_maps[0],
            right_maps[1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        left_output = output / "rgb" / f"{output_idx:08d}.png"
        right_output = output / "stereo_right" / f"{output_idx:08d}.png"
        write_options = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        if not cv2.imwrite(str(left_output), virtual_left, write_options):
            raise RuntimeError(f"Failed to write {left_output}")
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
        "pose_frame": "odom",
        "camera_quaternion_order": camera_quaternion_order,
        "pose_composition": (
            "odom_T_base_link @ base_link_T_head_camera "
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
            "method": "interpolate_odom_and_head_camera_at_cam0_sensor_time_ns",
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
    report = {
        "source_dataset": str(src),
        "source_projection_model": source_calibration["distortion_model"],
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
        "applied_optical_x_rotation_deg": optical_x_rotation_deg,
        "horizontal_fov_deg": args.horizontal_fov_deg,
        "down_fov_deg": args.down_fov_deg,
        "remap_valid_ratios": valid_ratios,
        "source_pairs": len(records),
        "matched_pairs": len(matches),
        "skipped_cam0": len(skipped_left),
        "skipped_cam1": len(skipped_right),
        "max_matched_delta_ms": max(delta for _, _, delta in matches) / 1.0e6,
        "odom_interpolation_clamped": odom_clamped,
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
