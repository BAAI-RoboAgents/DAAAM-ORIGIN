"""Contracts for the small-object tabletop semantic-mapping profiles."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.config import PipelineConfig  # noqa: E402


def test_tabletop_pipeline_profile_loads_and_keeps_small_masks():
    path = REPOSITORY_ROOT / "config" / "pipeline_config_tabletop.yaml"
    config = PipelineConfig.from_yaml(str(path))

    assert config.segmentation.min_mask_region_area == 150
    assert config.segmentation.model_config_path == (
        "fastsam/fastsam_tabletop_config.yaml"
    )
    assert config.workers.assignment_config.min_obs_per_track == 5
    assert config.depth.depth_lb == 0.20
    assert config.depth.depth_ub == 2.0

    fastsam = yaml.safe_load(
        (
            REPOSITORY_ROOT
            / "config"
            / "fastsam"
            / "fastsam_tabletop_config.yaml"
        ).read_text()
    )
    assert fastsam == {
        "fastsam_conf": 0.25,
        "fastsam_iou": 0.60,
        "fastsam_retina_masks": True,
    }


def test_tabletop_hydra_profile_retains_global_tsdf_and_small_objects():
    tabletop = yaml.safe_load(
        (REPOSITORY_ROOT / "config" / "hydra_g1_tabletop.yaml").read_text()
    )
    baseline = yaml.safe_load(
        (REPOSITORY_ROOT / "config" / "hydra_g1_high_quality.yaml").read_text()
    )
    active = tabletop["active_window"]
    extractor = active["object_extractor"]

    assert active["volumetric_map"]["voxel_size"] == baseline["active_window"][
        "volumetric_map"
    ]["voxel_size"] == 0.05
    assert active["volumetric_map"]["truncation_distance"] == 0.15
    assert active["object_detector"]["max_range"] == 2.0
    assert active["tracker"]["min_num_observations"] == 4
    assert extractor["min_object_allocation_confidence"] == 0.5
    # ExternalTracker confidence is n / (2k): four observations equal the
    # rejected 0.5 boundary, while the fifth is the first to pass it.
    assert 4 / (2 * active["tracker"]["min_num_observations"]) == 0.5
    assert 5 / (2 * active["tracker"]["min_num_observations"]) > 0.5
    assert extractor["min_object_volume"] == 0.0001
    assert extractor["object_reconstruction_resolution"] == -0.02
    assert extractor["min_reconstruction_resolution"] == 0.003
    assert tabletop["frontend"]["enable_mesh_objects"] is False
