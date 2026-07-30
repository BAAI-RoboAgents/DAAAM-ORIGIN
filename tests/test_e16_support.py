from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from daaam.experiments.e16_support import (
    choose_adaptive_object_source,
    parse_mesh_extraction_decisions,
    required_observations_for_allocation,
    summarize_semantic_support,
    track_confidence,
)


def test_strict_khronos_allocation_gate_requires_one_extra_observation():
    assert track_confidence(8, 8) == 0.5
    assert required_observations_for_allocation(4) == 5
    assert required_observations_for_allocation(6) == 7
    assert required_observations_for_allocation(8) == 9
    with pytest.raises(ValueError):
        required_observations_for_allocation(8, 1.0)


def test_support_ledger_applies_range_and_cluster_pixel_gates(tmp_path: Path):
    labels = np.array(
        [
            [1, 1, 1, 2],
            [1, 1, 1, 2],
        ],
        dtype=np.uint16,
    )
    depth = np.array(
        [
            [1000, 1000, 1000, 7000],
            [1000, 1000, 1000, 7000],
        ],
        dtype=np.uint16,
    )
    label_paths = []
    depth_paths = []
    frames = []
    for index in range(3):
        label_path = tmp_path / f"labels_{index}.png"
        depth_path = tmp_path / f"depth_{index}.png"
        assert cv2.imwrite(str(label_path), labels)
        assert cv2.imwrite(str(depth_path), depth)
        label_paths.append(label_path)
        depth_paths.append(depth_path)
        frames.append(
            {
                "intrinsics": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1]],
                "world_T_camera": np.eye(4).tolist(),
            }
        )

    rows = summarize_semantic_support(
        label_paths=label_paths,
        depth_paths=depth_paths,
        frames=frames,
        maximum_range_m=5.0,
        minimum_cluster_pixels=5,
        observation_thresholds=(2,),
    )
    by_label = {row["semantic_label"]: row for row in rows}
    assert by_label[1]["cluster_observation_count"] == 3
    assert by_label[1]["total_in_range_depth_pixels"] == 18
    assert by_label[1]["allocation_gate_by_minimum_observations"]["2"][
        "passes_allocation_confidence_strictly_above_0p5"
    ]
    assert by_label[2]["cluster_observation_count"] == 0
    assert by_label[2]["total_in_range_depth_pixels"] == 0


def test_extractor_decision_parser_preserves_terminal_reason():
    rows = parse_mesh_extraction_decisions(
        "I [MeshObjectExtractor] Dropping track 107 (machine): "
        "low confidence (0.5 < 0.5).\n"
        "I [MeshObjectExtractor] Extracted track 42 (cabinet) with volume "
        "0.2, confidence 1, mesh size 20.\n"
    )
    assert [(row["semantic_label"], row["decision"]) for row in rows] == [
        (107, "dropped"),
        (42, "extracted"),
    ]
    assert "low confidence" in rows[0]["detail"]


def test_adaptive_object_selection_prefers_near_except_compact_supported_far():
    def node(volume_scale: float, points: int):
        return {
            "dimensions_json": f"[{volume_scale}, 1.0, 1.0]",
            "mesh_points": points,
        }

    assert choose_adaptive_object_source(node(1.0, 100), None) == (
        "near",
        "near_only_preservation",
    )
    assert choose_adaptive_object_source(None, node(1.0, 100)) == (
        "far",
        "far_only_recovery",
    )
    assert choose_adaptive_object_source(node(1.0, 100), node(0.2, 60))[0] == "far"
    assert choose_adaptive_object_source(node(1.0, 100), node(0.2, 40))[0] == "near"
