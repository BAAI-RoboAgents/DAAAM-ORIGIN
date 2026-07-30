#!/usr/bin/env python3
"""Rebuild semantic-query geometry from RGB masks and map-frame LiDAR.

This is a reconstruction stage, not a query-time coordinate correction.
FastSAM supplies the persistent semantic masks, the raw map_T_base and
base_T_camera transforms supply the camera pose, a separately validated fixed
image rotation supplies the pixel-ray frame, and the LiDAR map supplies every
3D coordinate.  Failed objects are made image-only rather than retaining
unverified RGB-D geometry.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN


REPORT_SCHEMA = "daaam.lidar_semantic_query_geometry.v1"
SEMANTIC_SCHEMA = "daaam.semantic_query_index.v1"
EVIDENCE_SCHEMA = "daaam.query_evidence.v1"
LIDAR_GEOMETRY_SOURCE = "fastsam_masked_lidar_map_projection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-query-map", required=True, type=Path)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--prepared-dataset", required=True, type=Path)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--lidar-map", required=True, type=Path)
    parser.add_argument("--calibration-report", required=True, type=Path)
    parser.add_argument("--output-query-map", required=True, type=Path)
    parser.add_argument("--maximum-observation-frames", type=int, default=12)
    parser.add_argument("--minimum-cluster-points", type=int, default=8)
    parser.add_argument("--cluster-eps-m", type=float, default=0.18)
    parser.add_argument("--cluster-min-samples", type=int, default=4)
    parser.add_argument(
        "--projection-radius-m",
        type=float,
        default=0.0,
        help=(
            "Optional XY compute crop. Zero, the default, projects the whole "
            "LiDAR map and imposes no maximum semantic depth."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid JSONL object at {path}:{line_number}")
        records.append(value)
    return records


def transform_from_pose_xyzw(values: Any) -> np.ndarray:
    pose = np.asarray(values, dtype=np.float64).reshape(-1)
    if pose.size != 7 or not np.isfinite(pose).all():
        raise ValueError("Expected finite xyz + quaternion-xyzw pose")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def map_T_tf_camera(record: dict[str, Any]) -> np.ndarray:
    map_pose = record.get("map_pose", {}).get("value", {})
    camera = (
        record.get("poses", {})
        .get("values", {})
        .get("head_camera", {})
    )
    if (
        map_pose.get("target_frame") != "map"
        or map_pose.get("source_frame") != "base_link"
    ):
        raise ValueError("Raw pose is not map_T_base_link")
    if (
        camera.get("target_frame") != "base_link"
        or camera.get("source_frame")
        != "head_left_camera_color_optical_frame"
    ):
        raise ValueError("Raw camera pose is not base_link_T_head_camera")
    return transform_from_pose_xyzw(
        map_pose["pose_xyz_quat_xyzw"]
    ) @ transform_from_pose_xyzw(camera["pose_xyz_quat_xyzw"])


def verify_calibration(
    calibration: dict[str, Any],
    calibration_path: Path,
    raw_manifest_path: Path,
    tick_path: Path,
    lidar_map: Path,
) -> np.ndarray:
    if (
        calibration.get("schema")
        != "daaam.g1_image_lidar_rotation_calibration.v1"
        or calibration.get("status") != "passed"
    ):
        raise ValueError("Image/LiDAR rotation calibration did not pass")
    if calibration.get("coordinate_frame") != "map":
        raise ValueError("Calibration is not in map coordinates")
    if calibration.get("query_target_object_used") is not False:
        raise ValueError("Calibration must be independent of query targets")
    inputs = calibration.get("inputs", {})
    expected = (
        ("raw_manifest", raw_manifest_path),
        ("prepared_tick_index", tick_path),
        ("lidar_map", lidar_map),
    )
    for name, path in expected:
        declared = inputs.get(name, {})
        if declared.get("sha256") != sha256_file(path):
            raise ValueError(
                f"Calibration input checksum changed for {name}: {path}"
            )
    contract = calibration.get("quaternion_contract", {})
    if (
        contract.get("selected_order") != "xyzw"
        or float(contract["xyzw_error_deg"]["maximum"]) > 1.0e-6
    ):
        raise ValueError("Calibration did not prove the xyzw pose contract")
    correction = np.asarray(
        calibration["tf_camera_R_image_camera"], dtype=np.float64
    )
    if correction.shape != (3, 3):
        raise ValueError("Calibration rotation is not 3x3")
    if not np.allclose(correction.T @ correction, np.eye(3), atol=1.0e-8):
        raise ValueError("Calibration rotation is not orthonormal")
    if not np.isclose(np.linalg.det(correction), 1.0, atol=1.0e-8):
        raise ValueError("Calibration rotation is not proper")
    return correction


def scan_label_observations(
    label_dir: Path,
    frame_count: int,
    labels: set[int],
    shape: tuple[int, int],
) -> dict[int, list[dict[str, Any]]]:
    observations: dict[int, list[dict[str, Any]]] = {
        label: [] for label in labels
    }
    wanted = np.asarray(sorted(labels), dtype=np.int64)
    height, width = shape
    for frame_index in range(frame_count):
        label_path = label_dir / f"{frame_index:08d}.png"
        image = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != shape:
            raise ValueError(f"Missing or malformed label frame: {label_path}")
        present = np.intersect1d(np.unique(image), wanted, assume_unique=True)
        for raw_label in present:
            label = int(raw_label)
            ys, xs = np.where(image == label)
            if xs.size == 0:
                continue
            observations[label].append(
                {
                    "frame_index": frame_index,
                    "mask_pixels": int(xs.size),
                    "bbox_xyxy": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() + 1),
                        int(ys.max() + 1),
                    ],
                    "border_touch": bool(
                        xs.min() == 0
                        or ys.min() == 0
                        or xs.max() == width - 1
                        or ys.max() == height - 1
                    ),
                }
            )
    return observations


def uniformly_select_observations(
    observations: list[dict[str, Any]],
    maximum: int,
    evidence_frame: int,
) -> list[dict[str, Any]]:
    if not observations:
        return []
    if len(observations) <= maximum:
        selected = list(observations)
    else:
        indices = np.linspace(
            0, len(observations) - 1, maximum, dtype=np.int64
        )
        selected = [observations[int(index)] for index in sorted(set(indices))]
    if (
        all(item["frame_index"] != evidence_frame for item in selected)
        and any(item["frame_index"] == evidence_frame for item in observations)
    ):
        replacement = next(
            item
            for item in observations
            if item["frame_index"] == evidence_frame
        )
        selected[-1] = replacement
        selected.sort(key=lambda item: item["frame_index"])
    return selected


def project_lidar_map(
    points: np.ndarray,
    map_T_tf: np.ndarray,
    tf_R_image: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    shape: tuple[int, int],
    projection_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    fx, fy, cx, cy = intrinsics
    if projection_radius_m > 0.0:
        distance = np.linalg.norm(
            points[:, :2] - map_T_tf[:2, 3], axis=1
        )
        global_indices = np.flatnonzero(distance <= projection_radius_m)
        map_points = points[global_indices]
    else:
        global_indices = np.arange(points.shape[0], dtype=np.int64)
        map_points = points
    map_R_image = map_T_tf[:3, :3] @ tf_R_image
    image_points = map_R_image.T @ (
        map_points - map_T_tf[:3, 3]
    ).T
    depth = image_points[2]
    valid = np.isfinite(depth) & (depth > 0.20)
    image_points = image_points[:, valid]
    depth = depth[valid]
    global_indices = global_indices[valid]
    u = np.rint(fx * image_points[0] / depth + cx).astype(np.int32)
    v = np.rint(fy * image_points[1] / depth + cy).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[inside]
    v = v[inside]
    depth = depth[inside]
    global_indices = global_indices[inside]
    if u.size == 0:
        return global_indices, u, v, depth
    pixel = v.astype(np.int64) * width + u
    order = np.lexsort((depth, pixel))
    ordered_pixel = pixel[order]
    nearest = np.r_[True, ordered_pixel[1:] != ordered_pixel[:-1]]
    selected = order[nearest]
    return (
        global_indices[selected],
        u[selected],
        v[selected],
        depth[selected],
    )


def cluster_geometry(
    points: np.ndarray,
    point_indices: np.ndarray,
    support: np.ndarray,
    camera_positions: np.ndarray,
    *,
    eps_m: float,
    min_samples: int,
    minimum_cluster_points: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if points.shape[0] < minimum_cluster_points:
        return None, []
    if points.shape[0] > 50000:
        selection = np.linspace(
            0, points.shape[0] - 1, 50000, dtype=np.int64
        )
        points = points[selection]
        point_indices = point_indices[selection]
        support = support[selection]
    cluster_labels = DBSCAN(
        eps=eps_m,
        min_samples=min_samples,
        n_jobs=-1,
    ).fit_predict(points)
    identifiers, counts = np.unique(
        cluster_labels[cluster_labels >= 0], return_counts=True
    )
    if identifiers.size == 0:
        return None, []
    largest = int(np.max(counts))
    significance = max(
        minimum_cluster_points,
        int(math.ceil(largest * 0.03)),
    )
    candidates: list[dict[str, Any]] = []
    for identifier, count in zip(identifiers, counts):
        if int(count) < significance:
            continue
        mask = cluster_labels == identifier
        cluster_points = points[mask]
        distances = np.linalg.norm(
            cluster_points[:, None, :] - camera_positions[None, :, :],
            axis=2,
        )
        nearest_camera_distance = float(
            np.median(np.min(distances, axis=1))
        )
        candidates.append(
            {
                "cluster_id": int(identifier),
                "point_count": int(count),
                "nearest_camera_distance_median_m": nearest_camera_distance,
                "support_median_frames": float(np.median(support[mask])),
                "_mask": mask,
            }
        )
    if not candidates:
        return None, []
    # A mask ray terminates at its first substantial coherent surface.  This
    # rejects large background/floor clusters visible through sparse foliage
    # without using a class-specific position prior.
    chosen = min(
        candidates,
        key=lambda item: (
            item["nearest_camera_distance_median_m"],
            -item["support_median_frames"],
            -item["point_count"],
        ),
    )
    mask = chosen.pop("_mask")
    for item in candidates:
        item.pop("_mask", None)
    selected_points = points[mask]
    selected_indices = point_indices[mask]
    selected_support = support[mask]
    lower, upper = np.quantile(selected_points, [0.05, 0.95], axis=0)
    dimensions = np.maximum(upper - lower, 0.02)
    position = (lower + upper) * 0.5
    result = {
        **chosen,
        "points": selected_points,
        "point_indices": selected_indices,
        "support": selected_support,
        "geometry_position_m": position.tolist(),
        "geometry_dimensions_m": dimensions.tolist(),
        "robust_bounds_percentiles": [0.05, 0.95],
        "significant_cluster_minimum_points": significance,
    }
    return result, candidates


def strip_geometry_fields(item: dict[str, Any]) -> None:
    for key in (
        "point_cloud",
        "point_cloud_sha256",
        "point_count",
        "geometry_position_m",
        "geometry_dimensions_m",
        "geometry_source",
        "source_depth_sha256",
        "source_lidar_sha256",
        "calibration_report_sha256",
        "valid_depth_pixels",
        "valid_depth_ratio",
        "geometry_observation_frames",
        "geometry_support_threshold_frames",
    ):
        item.pop(key, None)


def main() -> None:
    args = parse_args()
    if args.maximum_observation_frames < 2:
        raise ValueError("maximum-observation-frames must be at least 2")
    if args.minimum_cluster_points < 4:
        raise ValueError("minimum-cluster-points must be at least 4")
    if args.cluster_eps_m <= 0.0 or args.cluster_min_samples < 2:
        raise ValueError("Invalid cluster configuration")
    if args.projection_radius_m < 0.0:
        raise ValueError("projection-radius-m cannot be negative")

    source_map = args.source_query_map.expanduser().resolve()
    source_dataset = args.source_dataset.expanduser().resolve()
    prepared = args.prepared_dataset.expanduser().resolve()
    label_dir = args.label_dir.expanduser().resolve()
    lidar_map = args.lidar_map.expanduser().resolve()
    calibration_path = args.calibration_report.expanduser().resolve()
    output_map = args.output_query_map.expanduser().resolve()
    if output_map.exists():
        raise FileExistsError(
            f"Output query map already exists; refusing to overwrite: {output_map}"
        )
    for directory in (source_map, source_dataset, prepared, label_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    for path in (lidar_map, calibration_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    dsg_name = "dsg_updated.json"
    semantic_name = "dsg_updated.semantic.json"
    evidence_name = "dsg_updated.evidence.json"
    manifest_name = "dsg_updated.manifest.json"
    for name in (dsg_name, semantic_name, evidence_name, manifest_name):
        if not (source_map / name).is_file():
            raise FileNotFoundError(source_map / name)

    raw_manifest_path = source_dataset / "manifest.jsonl"
    tick_path = prepared / "tick_index.json"
    calibration = load_json(calibration_path)
    correction = verify_calibration(
        calibration,
        calibration_path,
        raw_manifest_path,
        tick_path,
        lidar_map,
    )
    raw_records = load_jsonl(raw_manifest_path)
    tick_index = load_json(tick_path)
    if tick_index.get("pose_frame") != "map":
        raise ValueError("Prepared dataset does not declare map-frame poses")
    frames = tick_index.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Prepared dataset has no indexed frames")
    shape = (int(tick_index["height"]), int(tick_index["width"]))
    intrinsics = tuple(
        float(tick_index[field]) for field in ("fx", "fy", "cx", "cy")
    )

    semantic = load_json(source_map / semantic_name)
    evidence = load_json(source_map / evidence_name)
    manifest = load_json(source_map / manifest_name)
    if semantic.get("schema") != SEMANTIC_SCHEMA:
        raise ValueError("Unsupported semantic sidecar")
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("Unsupported evidence sidecar")
    dsg_sha = sha256_file(source_map / dsg_name)
    if (
        semantic.get("dsg_sha256") != dsg_sha
        or evidence.get("dsg_sha256") != dsg_sha
        or manifest.get("dsg_sha256") != dsg_sha
    ):
        raise ValueError("Source query-map sidecars are not bound to the DSG")

    semantic_records = semantic.get("records")
    evidence_objects = evidence.get("objects")
    if not isinstance(semantic_records, list) or not isinstance(
        evidence_objects, list
    ):
        raise ValueError("Source query-map records are malformed")
    records_by_id = {
        str(record["record_id"]): record for record in semantic_records
    }
    evidence_by_node = {
        str(item["node_id"]): item for item in evidence_objects
    }
    if set(records_by_id) - set(evidence_by_node):
        raise ValueError("Spatial semantic records are missing image evidence")
    label_to_node = {
        int(record["semantic_label"]): node_id
        for node_id, record in records_by_id.items()
    }
    if len(label_to_node) != len(records_by_id):
        raise ValueError("Spatial semantic labels are not unique")

    print(
        f"Indexing {len(label_to_node)} semantic labels across "
        f"{len(frames)} frames...",
        flush=True,
    )
    observations = scan_label_observations(
        label_dir, len(frames), set(label_to_node), shape
    )
    selected_by_node: dict[str, list[dict[str, Any]]] = {}
    frame_to_nodes: dict[int, list[str]] = defaultdict(list)
    for label, node_id in label_to_node.items():
        evidence_frame = int(evidence_by_node[node_id]["frame_index"])
        selected = uniformly_select_observations(
            observations[label],
            args.maximum_observation_frames,
            evidence_frame,
        )
        selected_by_node[node_id] = selected
        for item in selected:
            frame_to_nodes[int(item["frame_index"])].append(node_id)

    cloud = o3d.io.read_point_cloud(str(lidar_map))
    points = np.asarray(cloud.points, dtype=np.float64)
    if points.shape[0] < 1000 or not np.isfinite(points).all():
        raise ValueError("LiDAR map is empty or non-finite")
    counts_by_node: dict[str, dict[int, int]] = {
        node_id: defaultdict(int) for node_id in records_by_id
    }
    hits_by_node: dict[str, dict[int, int]] = {
        node_id: {} for node_id in records_by_id
    }
    camera_by_frame: dict[int, np.ndarray] = {}
    union_frames = sorted(frame_to_nodes)
    for progress, frame_index in enumerate(union_frames, start=1):
        frame_record = frames[frame_index]
        source_index = int(frame_record["source_idx"])
        if source_index < 0 or source_index >= len(raw_records):
            raise ValueError("Prepared source_idx is outside raw manifest")
        map_T_tf = map_T_tf_camera(raw_records[source_index])
        camera_by_frame[frame_index] = map_T_tf[:3, 3].copy()
        global_indices, u, v, _depth = project_lidar_map(
            points,
            map_T_tf,
            correction,
            intrinsics,
            shape,
            args.projection_radius_m,
        )
        label_image = cv2.imread(
            str(label_dir / f"{frame_index:08d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        projected_labels = label_image[v, u]
        for node_id in frame_to_nodes[frame_index]:
            label = int(records_by_id[node_id]["semantic_label"])
            selected_indices = np.unique(
                global_indices[projected_labels == label]
            )
            hits_by_node[node_id][frame_index] = int(
                selected_indices.size
            )
            counts = counts_by_node[node_id]
            for point_index in selected_indices:
                counts[int(point_index)] += 1
        if progress == 1 or progress % 25 == 0 or progress == len(union_frames):
            print(
                f"Projected LiDAR into {progress}/{len(union_frames)} "
                "selected frames",
                flush=True,
            )

    output_map.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_map, output_map)
    point_cloud_dir = output_map / "query_evidence" / "point_clouds"
    if point_cloud_dir.exists():
        shutil.rmtree(point_cloud_dir)
    point_cloud_dir.mkdir(parents=True)
    lidar_sha = sha256_file(lidar_map)
    calibration_sha = sha256_file(calibration_path)
    object_reports: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    # Mesh-bound DSG objects retain their LiDAR-validated authoritative mesh.
    # Remove the old RGB-D point summaries so the v003 package cannot expose
    # stale geometry through evidence.
    for item in evidence_objects:
        if str(item["node_id"]) not in records_by_id:
            strip_geometry_fields(item)

    for node_id, record in records_by_id.items():
        selected_frames = selected_by_node[node_id]
        counts = counts_by_node[node_id]
        valid_frame_count = sum(
            hits > 0 for hits in hits_by_node[node_id].values()
        )
        support_threshold = max(
            1,
            min(3, int(math.ceil(max(valid_frame_count, 1) * 0.10))),
        )
        supported_indices = np.asarray(
            sorted(
                point_index
                for point_index, count in counts.items()
                if count >= support_threshold
            ),
            dtype=np.int64,
        )
        support = np.asarray(
            [counts[int(index)] for index in supported_indices],
            dtype=np.int32,
        )
        camera_positions = np.asarray(
            [
                camera_by_frame[int(item["frame_index"])]
                for item in selected_frames
                if int(item["frame_index"]) in camera_by_frame
            ],
            dtype=np.float64,
        )
        result = None
        candidates: list[dict[str, Any]] = []
        if (
            supported_indices.size >= args.minimum_cluster_points
            and camera_positions.size
        ):
            result, candidates = cluster_geometry(
                points[supported_indices],
                supported_indices,
                support,
                camera_positions,
                eps_m=args.cluster_eps_m,
                min_samples=args.cluster_min_samples,
                minimum_cluster_points=args.minimum_cluster_points,
            )
        evidence_item = evidence_by_node[node_id]
        strip_geometry_fields(evidence_item)
        report: dict[str, Any] = {
            "node_id": node_id,
            "semantic_label": int(record["semantic_label"]),
            "observation_frames_available": len(
                observations[int(record["semantic_label"])]
            ),
            "observation_frames_selected": [
                int(item["frame_index"]) for item in selected_frames
            ],
            "frames_with_projected_mask_points": valid_frame_count,
            "per_frame_projected_mask_points": hits_by_node[node_id],
            "support_threshold_frames": support_threshold,
            "supported_unique_lidar_points": int(supported_indices.size),
            "candidate_clusters": candidates,
        }
        if result is None:
            failure_count += 1
            record["geometry_status"] = "image_only"
            record["geometry_confidence"] = 0.0
            record["source"] = "fastsam_image_only_no_lidar_support"
            record.pop("position_m", None)
            record.pop("dimensions_m", None)
            report.update(
                {
                    "status": "image_only",
                    "reason": "no_significant_visible_lidar_cluster",
                }
            )
        else:
            success_count += 1
            geometry_position = np.asarray(
                result["geometry_position_m"], dtype=np.float64
            )
            geometry_dimensions = np.asarray(
                result["geometry_dimensions_m"], dtype=np.float64
            )
            cluster_points = np.asarray(result.pop("points"))
            cluster_indices = np.asarray(result.pop("point_indices"))
            cluster_support = np.asarray(result.pop("support"))
            if cluster_points.shape[0] > 5000:
                order = np.lexsort(
                    (
                        cluster_indices,
                        -cluster_support.astype(np.int64),
                    )
                )
                chosen = np.sort(order[:5000])
                saved_points = cluster_points[chosen]
                saved_support = cluster_support[chosen]
            else:
                saved_points = cluster_points
                saved_support = cluster_support
            safe_id = node_id.replace("(", "_").replace(")", "")
            relative_cloud = (
                Path("query_evidence") / "point_clouds" / f"{safe_id}.npz"
            )
            cloud_path = output_map / relative_cloud
            np.savez_compressed(
                cloud_path,
                points_map_m=saved_points.astype(np.float32),
                observation_support=saved_support.astype(np.uint16),
                coordinate_frame=np.asarray("map"),
            )
            cloud_sha = sha256_file(cloud_path)
            frame_coverage = valid_frame_count / max(1, len(selected_frames))
            point_confidence = min(
                1.0,
                math.log1p(cluster_points.shape[0])
                / math.log1p(500),
            )
            confidence = float(frame_coverage * point_confidence)
            record.update(
                {
                    "geometry_status": "spatial_only",
                    "geometry_confidence": confidence,
                    "source": LIDAR_GEOMETRY_SOURCE,
                    "position_m": geometry_position.tolist(),
                    "dimensions_m": geometry_dimensions.tolist(),
                }
            )
            evidence_item.update(
                {
                    "point_cloud": str(relative_cloud),
                    "point_cloud_sha256": cloud_sha,
                    "point_count": int(saved_points.shape[0]),
                    "geometry_position_m": geometry_position.tolist(),
                    "geometry_dimensions_m": geometry_dimensions.tolist(),
                    "geometry_source": LIDAR_GEOMETRY_SOURCE,
                    "source_lidar_sha256": lidar_sha,
                    "calibration_report_sha256": calibration_sha,
                    "geometry_observation_frames": [
                        int(item["frame_index"]) for item in selected_frames
                    ],
                    "geometry_support_threshold_frames": support_threshold,
                }
            )
            evidence_frame = int(evidence_item["frame_index"])
            if evidence_frame in camera_by_frame:
                evidence_item["camera_position_m"] = camera_by_frame[
                    evidence_frame
                ].tolist()
            report.update(
                {
                    "status": "spatial_only",
                    "geometry_position_m": geometry_position.tolist(),
                    "geometry_dimensions_m": geometry_dimensions.tolist(),
                    "geometry_confidence": confidence,
                    "selected_cluster": result,
                    "saved_point_count": int(saved_points.shape[0]),
                    "point_cloud": str(relative_cloud),
                    "point_cloud_sha256": cloud_sha,
                }
            )
        object_reports.append(report)

    semantic["records"] = sorted(
        semantic_records,
        key=lambda item: (
            int(item["semantic_label"]),
            str(item["entity_id"]),
        ),
    )
    semantic["record_count"] = len(semantic_records)
    semantic["geometry_counts"] = {
        "image_only": failure_count,
        "spatial_only": success_count,
    }
    semantic["source"] = {
        **dict(semantic.get("source") or {}),
        "geometry_reconstruction": LIDAR_GEOMETRY_SOURCE,
        "calibration_report_sha256": calibration_sha,
        "source_lidar_sha256": lidar_sha,
    }
    semantic_path = output_map / semantic_name
    write_json(semantic_path, semantic)
    semantic_sha = sha256_file(semantic_path)

    evidence["objects"] = evidence_objects
    evidence["object_count"] = len(evidence_objects)
    evidence["missing_node_ids"] = []
    evidence["geometry_reconstruction"] = {
        "source": LIDAR_GEOMETRY_SOURCE,
        "successful_spatial_objects": success_count,
        "image_only_objects": failure_count,
        "source_lidar_sha256": lidar_sha,
        "calibration_report_sha256": calibration_sha,
    }
    evidence_path = output_map / evidence_name
    write_json(evidence_path, evidence)
    evidence_sha = sha256_file(evidence_path)

    report_payload = {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "method": LIDAR_GEOMETRY_SOURCE,
        "coordinate_frame": "map",
        "query_coordinate_compensation": False,
        "nearest_point_snapping": False,
        "source_query_map": str(source_map),
        "output_query_map": str(output_map),
        "source_dataset": str(source_dataset),
        "prepared_dataset": str(prepared),
        "label_dir": str(label_dir),
        "lidar_map": str(lidar_map),
        "source_lidar_sha256": lidar_sha,
        "calibration_report": str(calibration_path),
        "calibration_report_sha256": calibration_sha,
        "tf_camera_R_image_camera": correction.tolist(),
        "camera_quaternion_order": "xyzw",
        "pose_composition": (
            "map_T_base_link @ base_link_T_head_camera_xyzw "
            "@ tf_camera_T_image_camera"
        ),
        "projection": {
            "model": "pinhole",
            "intrinsics": {
                "fx": intrinsics[0],
                "fy": intrinsics[1],
                "cx": intrinsics[2],
                "cy": intrinsics[3],
                "width": shape[1],
                "height": shape[0],
            },
            "maximum_depth_m": None,
            "projection_radius_m": (
                None
                if args.projection_radius_m == 0.0
                else args.projection_radius_m
            ),
            "z_buffer": "nearest_map_point_per_pixel",
        },
        "clustering": {
            "eps_m": args.cluster_eps_m,
            "min_samples": args.cluster_min_samples,
            "minimum_cluster_points": args.minimum_cluster_points,
            "significance_fraction_of_largest": 0.03,
            "selection": (
                "nearest_substantial_coherent_surface_to_observing_cameras"
            ),
            "geometry_bounds_percentiles": [0.05, 0.95],
        },
        "spatial_records": len(semantic_records),
        "successful_spatial_objects": success_count,
        "image_only_objects": failure_count,
        "selected_projection_frames": len(union_frames),
        "semantic_index_sha256": semantic_sha,
        "evidence_sha256": evidence_sha,
        "objects": object_reports,
    }
    report_path = output_map / "lidar_semantic_geometry_report.json"
    write_json(report_path, report_payload)
    report_sha = sha256_file(report_path)

    mesh_bound = int(manifest["dsg_queryable_objects"])
    manifest["geometry_counts"] = {
        "mesh_bound": mesh_bound,
        "spatial_only": success_count,
        "image_only": failure_count,
    }
    manifest["semantic_index"] = {
        **dict(manifest.get("semantic_index") or {}),
        "file": semantic_name,
        "records": len(semantic_records),
        "schema": SEMANTIC_SCHEMA,
        "sha256": semantic_sha,
    }
    manifest["query_evidence"] = {
        "file": evidence_name,
        "schema": EVIDENCE_SCHEMA,
        "sha256": evidence_sha,
    }
    manifest["geometry_reconstruction"] = {
        "file": report_path.name,
        "schema": REPORT_SCHEMA,
        "sha256": report_sha,
        "source": LIDAR_GEOMETRY_SOURCE,
        "coordinate_frame": "map",
        "source_lidar_sha256": lidar_sha,
        "calibration_report_sha256": calibration_sha,
    }
    write_json(output_map / manifest_name, manifest)

    print(
        json.dumps(
            {
                "output_query_map": str(output_map),
                "spatial_only": success_count,
                "image_only": failure_count,
                "mesh_bound": mesh_bound,
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
