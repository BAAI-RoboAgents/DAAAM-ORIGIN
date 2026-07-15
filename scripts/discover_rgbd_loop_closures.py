#!/usr/bin/env python3
"""Discover depth-verified nonlocal RGB-D links for a sequence.

Visual-word retrieval only proposes candidates.  Every reported link must also
pass epipolar filtering and two-sided depth rigid fitting, so repeated office
textures cannot become a pose-graph constraint by themselves.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.transform import Rotation

from build_rgbd_pose_graph_dataset import (
    DenseCloudCache,
    create_intrinsic,
    multiscale_icp,
)
from refine_rgbd_trajectory import (
    FrameCache,
    estimate_3d_constraint,
    load_poses,
    select_keyframes,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve and geometrically verify nonlocal RGB-D links."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--keyframe-distance-m", type=float, default=0.10)
    parser.add_argument("--max-keyframe-gap", type=int, default=10)
    parser.add_argument("--feature-count", type=int, default=2000)
    parser.add_argument("--vocabulary-size", type=int, default=64)
    parser.add_argument("--vocabulary-samples", type=int, default=100000)
    parser.add_argument("--samples-per-frame", type=int, default=900)
    parser.add_argument("--min-keyframe-separation", type=int, default=25)
    parser.add_argument("--candidates-per-keyframe", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.12)
    parser.add_argument("--ratio-test", type=float, default=0.65)
    parser.add_argument("--min-inliers", type=int, default=80)
    parser.add_argument("--min-loop-inliers", type=int, default=140)
    parser.add_argument("--max-loop-3d-error-m", type=float, default=0.025)
    parser.add_argument("--max-loop-reprojection-error-px", type=float, default=2.0)
    parser.add_argument("--min-loop-inlier-ratio", type=float, default=0.20)
    parser.add_argument("--max-depth-m", type=float, default=None)
    parser.add_argument("--contact-sheet-count", type=int, default=24)
    parser.add_argument(
        "--dense-global-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use repeatable FPFH/ICP plus dense depth/color verification when "
            "sparse feature geometry is insufficient."
        ),
    )
    parser.add_argument("--dense-candidate-count", type=int, default=80)
    parser.add_argument("--dense-image-scale", type=float, default=0.25)
    parser.add_argument("--dense-voxel-size-m", type=float, default=0.07)
    parser.add_argument("--dense-hypotheses", type=int, default=3)
    parser.add_argument("--dense-ransac-iterations", type=int, default=100000)
    parser.add_argument("--dense-repeat-rotation-deg", type=float, default=2.0)
    parser.add_argument("--dense-repeat-translation-m", type=float, default=0.05)
    parser.add_argument("--dense-min-forward-fitness", type=float, default=0.30)
    parser.add_argument("--dense-min-reverse-fitness", type=float, default=0.20)
    parser.add_argument("--dense-max-icp-rmse-m", type=float, default=0.045)
    parser.add_argument("--dense-max-translation-m", type=float, default=4.5)
    parser.add_argument(
        "--dense-max-rotation-from-prior-deg", type=float, default=45.0
    )
    parser.add_argument("--dense-pixel-step", type=int, default=4)
    parser.add_argument("--dense-min-comparable-ratio", type=float, default=0.25)
    parser.add_argument("--dense-min-depth-agreement", type=float, default=0.25)
    parser.add_argument("--dense-max-median-lab-delta", type=float, default=25.0)
    parser.add_argument("--dense-min-color-agreement", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_histograms(
    cache: FrameCache,
    keyframes: list[int],
    vocabulary_size: int,
    vocabulary_samples: int,
    samples_per_frame: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    descriptor_samples = []
    descriptors = []
    for ordinal, frame in enumerate(keyframes, start=1):
        _, descriptor = cache.feature(frame)
        if descriptor is None or len(descriptor) == 0:
            descriptor = np.empty((0, 128), dtype=np.float32)
        descriptor = descriptor.astype(np.float32, copy=False)
        descriptors.append(descriptor)
        if len(descriptor):
            indices = rng.choice(
                len(descriptor), min(samples_per_frame, len(descriptor)), replace=False
            )
            descriptor_samples.append(descriptor[indices])
        if ordinal % 25 == 0 or ordinal == len(keyframes):
            print(f"Loaded features {ordinal}/{len(keyframes)}", flush=True)

    training = np.concatenate(descriptor_samples)
    if len(training) < vocabulary_size:
        raise RuntimeError("Too few image descriptors for visual-word retrieval")
    if len(training) > vocabulary_samples:
        training = training[
            rng.choice(len(training), vocabulary_samples, replace=False)
        ]
    vocabulary = MiniBatchKMeans(
        n_clusters=vocabulary_size,
        batch_size=4096,
        n_init=3,
        max_iter=100,
        random_state=seed,
    ).fit(training)

    histograms = np.zeros((len(keyframes), vocabulary_size), dtype=np.float64)
    for index, descriptor in enumerate(descriptors):
        if len(descriptor):
            words = vocabulary.predict(descriptor)
            histograms[index] = np.bincount(words, minlength=vocabulary_size)
    document_frequency = (histograms > 0).sum(axis=0)
    inverse_frequency = np.log((len(histograms) + 1.0) / (document_frequency + 1.0)) + 1.0
    histograms *= inverse_frequency
    norms = np.linalg.norm(histograms, axis=1, keepdims=True)
    histograms /= np.maximum(norms, 1.0e-12)
    return histograms


def retrieve_pairs(
    histograms: np.ndarray,
    min_separation: int,
    candidates_per_keyframe: int,
    min_similarity: float,
) -> list[tuple[int, int, float]]:
    similarity = histograms @ histograms.T
    pairs: dict[tuple[int, int], float] = {}
    for first in range(len(histograms)):
        order = np.argsort(similarity[first])[::-1]
        selected = 0
        for second in order:
            if abs(first - second) < min_separation:
                continue
            score = float(similarity[first, second])
            if score < min_similarity:
                break
            pair = (int(min(first, second)), int(max(first, second)))
            pairs[pair] = max(score, pairs.get(pair, -np.inf))
            selected += 1
            if selected == candidates_per_keyframe:
                break
    return [
        (first, second, score)
        for (first, second), score in sorted(
            pairs.items(), key=lambda item: item[1], reverse=True
        )
    ]


class DenseLoopCache:
    """Cache quarter-resolution clouds and global geometric descriptors."""

    def __init__(
        self,
        dataset: Path,
        camera: dict,
        image_scale: float,
        voxel_size_m: float,
        max_depth_m: float,
    ):
        intrinsic, width, height = create_intrinsic(camera, image_scale)
        self.cloud_cache = DenseCloudCache(
            dataset, intrinsic, width, height, max_depth_m
        )
        self.voxel_size_m = voxel_size_m
        self.global_features: dict[
            int, tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]
        ] = {}

    def cloud(self, frame: int) -> o3d.geometry.PointCloud:
        return self.cloud_cache.cloud(frame)

    def global_feature(
        self, frame: int
    ) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        if frame not in self.global_features:
            cloud = self.cloud(frame).voxel_down_sample(self.voxel_size_m)
            cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size_m * 2.0, max_nn=30
                )
            )
            feature = o3d.pipelines.registration.compute_fpfh_feature(
                cloud,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size_m * 5.0, max_nn=100
                ),
            )
            self.global_features[frame] = (cloud, feature)
        return self.global_features[frame]


def transform_disagreement(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    rotation_deg = float(
        np.rad2deg(
            Rotation.from_matrix(first[:3, :3] @ second[:3, :3].T).magnitude()
        )
    )
    translation_m = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    return rotation_deg, translation_m


def dense_registration_hypothesis(
    cache: DenseLoopCache,
    first: int,
    second: int,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    source_global, source_feature = cache.global_feature(first)
    target_global, target_feature = cache.global_feature(second)
    o3d.utility.random.seed(seed)
    global_result = (
        o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_global,
            target_global,
            source_feature,
            target_feature,
            True,
            args.dense_voxel_size_m * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    args.dense_voxel_size_m * 1.5
                ),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                args.dense_ransac_iterations, 0.999
            ),
        )
    )
    source = cache.cloud(first)
    target = cache.cloud(second)
    transform, icp_levels = multiscale_icp(
        source, target, global_result.transformation
    )
    forward = o3d.pipelines.registration.evaluate_registration(
        source, target, 0.05, transform
    )
    reverse = o3d.pipelines.registration.evaluate_registration(
        target, source, 0.05, np.linalg.inv(transform)
    )
    return {
        "seed": seed,
        "transform": transform,
        "global_fitness": float(global_result.fitness),
        "global_rmse_m": float(global_result.inlier_rmse),
        "icp_levels": [[float(fitness), float(rmse)] for fitness, rmse in icp_levels],
        "forward_fitness_5cm": float(forward.fitness),
        "forward_rmse_5cm_m": float(forward.inlier_rmse),
        "reverse_fitness_5cm": float(reverse.fitness),
        "reverse_rmse_5cm_m": float(reverse.inlier_rmse),
    }


def select_repeatable_hypothesis(
    hypotheses: list[dict], args: argparse.Namespace
) -> tuple[dict | None, int]:
    best = None
    best_repeat_count = 0
    for hypothesis in hypotheses:
        repeat_count = 0
        for alternate in hypotheses:
            rotation_deg, translation_m = transform_disagreement(
                hypothesis["transform"], alternate["transform"]
            )
            if (
                rotation_deg <= args.dense_repeat_rotation_deg
                and translation_m <= args.dense_repeat_translation_m
            ):
                repeat_count += 1
        score = (
            repeat_count,
            min(
                hypothesis["forward_fitness_5cm"],
                hypothesis["reverse_fitness_5cm"],
            ),
            -hypothesis["icp_levels"][-1][1],
        )
        if best is None or score > best[0]:
            best = (score, hypothesis)
            best_repeat_count = repeat_count
    return (best[1] if best is not None else None), best_repeat_count


def dense_pixel_verification(
    dataset: Path,
    first: int,
    second: int,
    transform: np.ndarray,
    camera: dict,
    max_depth_m: float,
    pixel_step: int,
) -> dict:
    first_depth = cv2.imread(
        str(dataset / "depth" / f"{first:08d}.png"), cv2.IMREAD_UNCHANGED
    )
    second_depth = cv2.imread(
        str(dataset / "depth" / f"{second:08d}.png"), cv2.IMREAD_UNCHANGED
    )
    first_rgb = cv2.imread(str(dataset / "rgb" / f"{first:08d}.png"))
    second_rgb = cv2.imread(str(dataset / "rgb" / f"{second:08d}.png"))
    if (
        first_depth is None
        or second_depth is None
        or first_rgb is None
        or second_rgb is None
        or first_depth.dtype != np.uint16
        or second_depth.dtype != np.uint16
    ):
        raise ValueError(f"Invalid dense loop input for frames {first}, {second}")
    first_depth = first_depth.astype(np.float32) / 1000.0
    second_depth = second_depth.astype(np.float32) / 1000.0
    fx, fy, cx, cy = (float(camera[key]) for key in ("fx", "fy", "cx", "cy"))
    height, width = first_depth.shape
    v, u = np.mgrid[0:height:pixel_step, 0:width:pixel_step]
    z = first_depth[::pixel_step, ::pixel_step]
    valid_source = (z >= 0.25) & (z <= max_depth_m)
    points = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=-1)
    transformed = points @ transform[:3, :3].T + transform[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_u = fx * transformed[..., 0] / transformed[..., 2] + cx
        projected_v = fy * transformed[..., 1] / transformed[..., 2] + cy
    in_bounds = (
        valid_source
        & (transformed[..., 2] > 0.0)
        & (projected_u >= 0.0)
        & (projected_u <= width - 1.0)
        & (projected_v >= 0.0)
        & (projected_v <= height - 1.0)
    )
    target_z = cv2.remap(
        second_depth,
        projected_u.astype(np.float32),
        projected_v.astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    comparable = in_bounds & (target_z >= 0.25) & (target_z <= max_depth_m)
    absolute_error = np.abs(target_z - transformed[..., 2])
    tolerance = 0.04 + 0.03 * transformed[..., 2]
    depth_agreement = comparable & (absolute_error <= tolerance)

    first_lab = cv2.cvtColor(first_rgb, cv2.COLOR_BGR2LAB)[
        ::pixel_step, ::pixel_step
    ].astype(np.float32)
    second_lab = cv2.cvtColor(second_rgb, cv2.COLOR_BGR2LAB)
    projected_lab = cv2.remap(
        second_lab,
        projected_u.astype(np.float32),
        projected_v.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)
    lab_delta = np.linalg.norm(first_lab - projected_lab, axis=-1)
    valid_count = int(valid_source.sum())
    comparable_count = int(comparable.sum())
    agreement_count = int(depth_agreement.sum())
    return {
        "valid_source_samples": valid_count,
        "comparable_samples": comparable_count,
        "comparable_ratio": (
            float(comparable_count / valid_count) if valid_count else 0.0
        ),
        "depth_agreement_samples": agreement_count,
        "depth_agreement_rate": (
            float(agreement_count / comparable_count) if comparable_count else 0.0
        ),
        "median_absolute_depth_error_m": (
            float(np.median(absolute_error[comparable])) if comparable_count else None
        ),
        "median_lab_delta_on_depth_agreement": (
            float(np.median(lab_delta[depth_agreement])) if agreement_count else None
        ),
        "color_agreement_rate_on_depth_agreement": (
            float((lab_delta[depth_agreement] <= 30.0).mean())
            if agreement_count
            else 0.0
        ),
    }


def estimate_dense_loop(
    dataset: Path,
    first: int,
    second: int,
    poses: np.ndarray,
    camera: dict,
    max_depth_m: float,
    cache: DenseLoopCache,
    args: argparse.Namespace,
) -> dict:
    hypotheses = [
        dense_registration_hypothesis(
            cache, first, second, args.seed + hypothesis_index, args
        )
        for hypothesis_index in range(args.dense_hypotheses)
    ]
    selected, repeat_count = select_repeatable_hypothesis(hypotheses, args)
    required_repeat_count = max(2, args.dense_hypotheses - 1)
    if selected is None:
        return {"accepted": False, "reasons": ["no_dense_hypothesis"]}
    transform = selected["transform"]
    source_prior = np.linalg.inv(poses[second]) @ poses[first]
    prior_rotation_difference_deg, _ = transform_disagreement(
        transform, source_prior
    )
    pixel = dense_pixel_verification(
        dataset,
        first,
        second,
        transform,
        camera,
        max_depth_m,
        args.dense_pixel_step,
    )
    checks = {
        "repeatable_hypotheses": repeat_count >= required_repeat_count,
        "forward_fitness": (
            selected["forward_fitness_5cm"] >= args.dense_min_forward_fitness
        ),
        "reverse_fitness": (
            selected["reverse_fitness_5cm"] >= args.dense_min_reverse_fitness
        ),
        "icp_rmse": selected["icp_levels"][-1][1] <= args.dense_max_icp_rmse_m,
        "translation": (
            np.linalg.norm(transform[:3, 3]) <= args.dense_max_translation_m
        ),
        "rotation_from_prior": (
            prior_rotation_difference_deg
            <= args.dense_max_rotation_from_prior_deg
        ),
        "comparable_ratio": (
            pixel["comparable_ratio"] >= args.dense_min_comparable_ratio
        ),
        "depth_agreement": (
            pixel["depth_agreement_rate"] >= args.dense_min_depth_agreement
        ),
        "median_lab_delta": (
            pixel["median_lab_delta_on_depth_agreement"] is not None
            and pixel["median_lab_delta_on_depth_agreement"]
            <= args.dense_max_median_lab_delta
        ),
        "color_agreement": (
            pixel["color_agreement_rate_on_depth_agreement"]
            >= args.dense_min_color_agreement
        ),
    }
    serializable_hypotheses = []
    for hypothesis in hypotheses:
        serializable = {key: value for key, value in hypothesis.items() if key != "transform"}
        serializable["transform"] = hypothesis["transform"].tolist()
        serializable_hypotheses.append(serializable)
    return {
        "accepted": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "repeat_count": repeat_count,
        "required_repeat_count": required_repeat_count,
        "rotation_difference_from_input_prior_deg": prior_rotation_difference_deg,
        "estimated_translation_m": float(np.linalg.norm(transform[:3, 3])),
        "pixel_verification": pixel,
        "selected_hypothesis": {
            key: value for key, value in selected.items() if key != "transform"
        },
        "hypotheses": serializable_hypotheses,
        "transform": transform.tolist(),
    }


def make_contact_sheet(
    dataset: Path,
    keyframes: list[int],
    candidates: list[dict],
    path: Path,
    count: int,
) -> None:
    candidates = candidates[:count]
    if not candidates:
        return
    tile_width, tile_height = 280, 210
    columns = 3
    rows = int(np.ceil(len(candidates) / columns))
    canvas = np.full((rows * tile_height, columns * tile_width * 2, 3), 245, dtype=np.uint8)
    for ordinal, candidate in enumerate(candidates):
        row, column = divmod(ordinal, columns)
        x0 = column * tile_width * 2
        y0 = row * tile_height
        for offset, key in enumerate(("first", "second")):
            frame = keyframes[candidate[key]]
            image = cv2.imread(str(dataset / "rgb" / f"{frame:08d}.png"))
            if image is None:
                continue
            image = cv2.resize(image, (tile_width, tile_height))
            canvas[y0 : y0 + tile_height, x0 + offset * tile_width : x0 + (offset + 1) * tile_width] = image
        label = (
            f"{keyframes[candidate['first']]} / {keyframes[candidate['second']]} "
            f"score={candidate['similarity']:.3f}"
        )
        if "constraint" in candidate:
            label += f" inliers={candidate['constraint']['inliers']}"
        cv2.putText(
            canvas,
            label,
            (x0 + 8, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x0 + 8, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), canvas)


def main():
    args = parse_args()
    if args.vocabulary_size < 8 or args.candidates_per_keyframe < 1:
        raise ValueError("Vocabulary size and candidates per keyframe must be positive")
    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setRNGSeed(args.seed)
    poses = load_poses(dataset / "pose" / "poses.txt")
    camera = json.loads((dataset / "camera_info.json").read_text())
    camera_matrix = np.asarray(camera["intrinsics"], dtype=np.float64)
    tick_index = json.loads((dataset / "tick_index.json").read_text())
    max_depth_m = args.max_depth_m or float(
        tick_index.get("recommended_max_depth_m", 3.0)
    )
    keyframes = select_keyframes(
        poses, args.keyframe_distance_m, args.max_keyframe_gap
    )
    cache = FrameCache(dataset)
    cache.sift = cv2.SIFT_create(nfeatures=args.feature_count)
    histograms = make_histograms(
        cache,
        keyframes,
        args.vocabulary_size,
        args.vocabulary_samples,
        args.samples_per_frame,
        args.seed,
    )
    pairs = retrieve_pairs(
        histograms,
        args.min_keyframe_separation,
        args.candidates_per_keyframe,
        args.min_similarity,
    )
    retrieved = [
        {"first": first, "second": second, "similarity": score}
        for first, second, score in pairs
    ]
    make_contact_sheet(
        dataset,
        keyframes,
        retrieved,
        output_dir / "retrieved_candidates.png",
        args.contact_sheet_count,
    )

    geometric_candidates = []
    verified = []
    for ordinal, (first_key, second_key, score) in enumerate(pairs, start=1):
        constraint = estimate_3d_constraint(
            cache,
            keyframes[first_key],
            keyframes[second_key],
            poses,
            camera_matrix,
            max_depth_m,
            args.ratio_test,
            args.min_inliers,
            180.0,
            True,
        )
        if constraint is not None:
            quality_ok = (
                constraint.inliers >= args.min_loop_inliers
                and constraint.median_3d_error_m <= args.max_loop_3d_error_m
                and constraint.median_reprojection_error_px
                <= args.max_loop_reprojection_error_px
                and constraint.inlier_ratio >= args.min_loop_inlier_ratio
            )
            candidate = {
                "first": int(first_key),
                "second": int(second_key),
                "similarity": float(score),
                "method": "sift_depth_3d3d",
                "constraint": asdict(constraint),
                "quality_ok": bool(quality_ok),
            }
            geometric_candidates.append(candidate)
            if quality_ok:
                verified.append(candidate)
        if ordinal % 25 == 0 or ordinal == len(pairs):
            print(
                f"Verified {ordinal}/{len(pairs)} candidates; accepted {len(verified)}",
                flush=True,
            )

    dense_candidates = []
    if args.dense_global_fallback:
        if not 0.0 < args.dense_image_scale <= 1.0:
            raise ValueError("--dense-image-scale must be in (0, 1]")
        if args.dense_hypotheses < 2 or args.dense_candidate_count < 1:
            raise ValueError("Dense fallback requires at least two hypotheses")
        dense_cache = DenseLoopCache(
            dataset,
            camera,
            args.dense_image_scale,
            args.dense_voxel_size_m,
            max_depth_m,
        )
        sift_verified_pairs = {
            (candidate["first"], candidate["second"]) for candidate in verified
        }
        dense_pairs = [
            pair
            for pair in pairs
            if (pair[0], pair[1]) not in sift_verified_pairs
        ][: args.dense_candidate_count]
        for ordinal, (first_key, second_key, score) in enumerate(
            dense_pairs, start=1
        ):
            first = keyframes[first_key]
            second = keyframes[second_key]
            dense = estimate_dense_loop(
                dataset,
                first,
                second,
                poses,
                camera,
                max_depth_m,
                dense_cache,
                args,
            )
            candidate = {
                "first": int(first_key),
                "second": int(second_key),
                "similarity": float(score),
                "method": "fpfh_icp_dense_depth_color",
                "quality_ok": bool(dense["accepted"]),
                "dense_verification": dense,
            }
            if dense["accepted"]:
                transform = np.asarray(dense["transform"], dtype=np.float64)
                pixel = dense["pixel_verification"]
                delta_world = -poses[second, :3, :3] @ transform[:3, 3]
                candidate["constraint"] = {
                    "first": first,
                    "second": second,
                    "delta_world_xy": delta_world[:2].tolist(),
                    "sigma_m": float(
                        np.clip(
                            pixel["median_absolute_depth_error_m"], 0.025, 0.08
                        )
                    ),
                    "inliers": pixel["depth_agreement_samples"],
                    "median_reprojection_error_px": None,
                    "rotation_error_deg": dense[
                        "rotation_difference_from_input_prior_deg"
                    ],
                    "relative_camera_rotation": transform[:3, :3].tolist(),
                    "relative_camera_translation_m": transform[:3, 3].tolist(),
                    "constraint_type": "fpfh_icp_dense_depth_color",
                    "median_3d_error_m": pixel["median_absolute_depth_error_m"],
                    "inlier_ratio": pixel["depth_agreement_rate"],
                    "loop": True,
                }
                verified.append(candidate)
            dense_candidates.append(candidate)
            if ordinal % 10 == 0 or ordinal == len(dense_pairs):
                dense_verified = sum(
                    candidate["quality_ok"] for candidate in dense_candidates
                )
                print(
                    f"Dense-verified {ordinal}/{len(dense_pairs)} candidates; "
                    f"accepted {dense_verified}",
                    flush=True,
                )

    verified.sort(key=lambda item: item["similarity"], reverse=True)
    make_contact_sheet(
        dataset,
        keyframes,
        geometric_candidates,
        output_dir / "geometric_candidates.png",
        args.contact_sheet_count,
    )
    make_contact_sheet(
        dataset,
        keyframes,
        dense_candidates,
        output_dir / "dense_candidates.png",
        args.contact_sheet_count,
    )
    make_contact_sheet(
        dataset,
        keyframes,
        verified,
        output_dir / "verified_links.png",
        args.contact_sheet_count,
    )
    report = {
        "dataset": str(dataset),
        "keyframes": keyframes,
        "keyframe_count": len(keyframes),
        "max_depth_m": max_depth_m,
        "retrieved_candidates": retrieved,
        "geometric_candidates": geometric_candidates,
        "dense_candidates": dense_candidates,
        "verified_links": verified,
        "retrieved_count": len(retrieved),
        "geometric_count": len(geometric_candidates),
        "dense_tested_count": len(dense_candidates),
        "dense_verified_count": sum(
            candidate["quality_ok"] for candidate in dense_candidates
        ),
        "verified_count": len(verified),
        "settings": vars(args),
    }
    report["settings"]["dataset"] = str(dataset)
    report["settings"]["output_dir"] = str(output_dir)
    (output_dir / "loop_closure_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        f"Found {len(verified)}/{len(retrieved)} depth-verified nonlocal links\n"
        f"  report: {output_dir / 'loop_closure_report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
