"""Regression tests for absolute-time, content-safe stereo keyframe selection."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
CAPTURE_OFFSETS_NS = [0, 80_000_000, 230_000_000, 440_000_000, 540_000_000, 900_000_000]


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


selector = load_script_module(
    "select_mapping_keyframes_test",
    REPOSITORY_ROOT / "scripts" / "select_mapping_keyframes.py",
)
mapping_runner = load_script_module(
    "run_stereo_mapping_test",
    REPOSITORY_ROOT / "scripts" / "run_stereo_mapping.py",
)


class FakeLogger:
    def debug(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class FakeHydraPipeline:
    def __init__(self):
        self.step_args = None

    def step(self, *args):
        self.step_args = args
        return True


def pose_matrix(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return pose


def create_synthetic_dataset(
    root: Path, small_content: bool = False
) -> tuple[Path, int]:
    dataset = root / "source"
    (dataset / "rgb").mkdir(parents=True)
    (dataset / "stereo_right").mkdir()
    (dataset / "pose").mkdir()

    rng = np.random.default_rng(17)
    base = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    cv2.putText(base, "stable background", (35, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    content_change = base.copy()
    if small_content:
        cv2.rectangle(content_change, (185, 45), (197, 57), (0, 0, 255), -1)
    else:
        cv2.rectangle(content_change, (185, 45), (270, 145), (0, 0, 255), -1)
        cv2.putText(content_change, "new", (195, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    second_content_change = base.copy()
    if small_content:
        second_content_change = content_change.copy()
    else:
        cv2.rectangle(second_content_change, (185, 45), (270, 145), (255, 0, 0), -1)
        cv2.putText(second_content_change, "next", (190, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    images = [
        base,
        base,
        content_change,
        second_content_change,
        second_content_change,
        second_content_change,
    ]
    positions = [0.0, 0.01, 0.01, 0.01, 0.01, 0.20]
    origin_ns = 1_783_933_507_759_540_877
    frames = []
    for index, image in enumerate(images):
        rgb = dataset / "rgb" / f"{index:08d}.png"
        right = dataset / "stereo_right" / f"{index:08d}.png"
        assert cv2.imwrite(str(rgb), image)
        assert cv2.imwrite(str(right), image)
        timestamp_ns = origin_ns + CAPTURE_OFFSETS_NS[index]
        frames.append(
            {
                "idx": index,
                "source_idx": index,
                "cam0_source_idx": index,
                "cam1_source_idx": index,
                "pose_row": index,
                "cam0": str(rgb),
                "cam1": str(right),
                "timestamp": (timestamp_ns - origin_ns) / 1.0e9,
                "cam0_sensor_time_ns": timestamp_ns,
                "cam1_sensor_time_ns": timestamp_ns,
                "sensor_time_ns": timestamp_ns,
                "pose_sensor_time_ns": timestamp_ns,
                "stereo_delta_ms": 0.0,
            }
        )
    poses = "".join(
        " ".join(f"{value:.12g}" for value in pose_matrix(position).reshape(-1)) + "\n"
        for position in positions
    )
    (dataset / "pose" / "poses.txt").write_text(poses)
    (dataset / "pose" / "pose_timestamps_ns.txt").write_text(
        "".join(f"{origin_ns + offset_ns}\n" for offset_ns in CAPTURE_OFFSETS_NS)
    )
    (dataset / "camera_info.json").write_text(
        json.dumps(
            {
                "width": 320,
                "height": 240,
                "model": "pinhole",
                "intrinsics": [[250.0, 0.0, 160.0], [0.0, 250.0, 120.0], [0.0, 0.0, 1.0]],
            }
        )
    )
    (dataset / "tick_index.json").write_text(
        json.dumps(
            {
                "time_origin_ns": origin_ns,
                "fx": 250.0,
                "baseline": 0.07,
                "projection_model": "pinhole",
                "frames": frames,
            }
        )
    )
    return dataset, origin_ns


class TimeSafeKeyframeTests(unittest.TestCase):
    def test_timestamp_ns_extension_preserves_legacy_positional_fields(self):
        from daaam.datasets.interfaces import DatasetFrame
        from daaam.pipeline.models import Frame

        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        depth = np.ones((2, 2), dtype=np.float32)
        transform = np.arange(7, dtype=np.float64)
        dataset_frame = DatasetFrame(4, 0.4, rgb, depth, transform)
        pipeline_frame = Frame(4, 0.4, rgb, depth, transform)
        self.assertIs(dataset_frame.depth_image, depth)
        self.assertIs(dataset_frame.transform, transform)
        self.assertIsNone(dataset_frame.timestamp_ns)
        self.assertIs(pipeline_frame.depth_image, depth)
        self.assertIs(pipeline_frame.transform, transform)
        self.assertIsNone(pipeline_frame.timestamp_ns)

    def test_stationary_content_change_is_preserved_with_absolute_times(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, origin_ns = create_synthetic_dataset(root)
            output = root / "selected"
            report = selector.run_selection(
                dataset, output, selector.SelectionConfig(), overwrite=False
            )

            self.assertEqual(report["selected_frame_count"], 4)
            decisions = {item["source_frame_idx"]: item for item in report["decisions"]}
            self.assertEqual(decisions[1]["reason"], "strict_duplicate")
            self.assertEqual(decisions[2]["reason"], "image_event_at_static_pose")
            self.assertEqual(decisions[3]["reason"], "image_event_at_static_pose")
            self.assertEqual(decisions[4]["reason"], "strict_duplicate")
            self.assertEqual(decisions[5]["reason"], "pose_motion")

            output_index = json.loads((output / "tick_index.json").read_text())
            self.assertEqual(
                [frame["source_frame_idx"] for frame in output_index["frames"]], [0, 2, 3, 5]
            )
            self.assertEqual(
                [frame["sensor_time_ns"] for frame in output_index["frames"]],
                [
                    origin_ns,
                    origin_ns + 230_000_000,
                    origin_ns + 440_000_000,
                    origin_ns + 900_000_000,
                ],
            )
            self.assertEqual(
                (output / "pose" / "pose_timestamps_ns.txt").read_text().splitlines(),
                [
                    str(origin_ns),
                    str(origin_ns + 230_000_000),
                    str(origin_ns + 440_000_000),
                    str(origin_ns + 900_000_000),
                ],
            )

            from daaam.datasets.loaders.image_sequence import ImageSequenceDataset

            loaded = ImageSequenceDataset(output, compute_velocities=False)
            self.assertEqual(loaded[1].timestamp_ns, origin_ns + 230_000_000)
            self.assertAlmostEqual(loaded[1].timestamp, 0.23)

    def test_misaligned_pose_time_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, _ = create_synthetic_dataset(root)
            metadata_path = dataset / "tick_index.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["frames"][2]["pose_sensor_time_ns"] += 1
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "image/pose timestamps"):
                selector.run_selection(
                    dataset,
                    root / "selected",
                    selector.SelectionConfig(),
                    dry_run=True,
                )
            with self.assertRaisesRegex(RuntimeError, "camera and pose absolute times"):
                mapping_runner.validate_time_contract(dataset)

    def test_mapping_manifest_contract_reports_nonuniform_capture_times(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, origin_ns = create_synthetic_dataset(root)
            contract = mapping_runner.validate_time_contract(dataset)
            self.assertTrue(contract["valid"])
            self.assertEqual(contract["frame_count"], len(CAPTURE_OFFSETS_NS))
            self.assertEqual(contract["time_origin_ns"], origin_ns)

    def test_small_salient_stationary_content_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, _ = create_synthetic_dataset(root, small_content=True)
            report = selector.run_selection(
                dataset, root / "selected", selector.SelectionConfig()
            )
            decisions = {item["source_frame_idx"]: item for item in report["decisions"]}
            self.assertEqual(decisions[2]["reason"], "image_event_at_static_pose")
            self.assertEqual(decisions[3]["reason"], "strict_duplicate")

    def test_hydra_prefers_absolute_timestamp(self):
        from daaam.hydra.integration import HydraIntegration

        integration = object.__new__(HydraIntegration)
        integration.pipeline = FakeHydraPipeline()
        integration.logger = FakeLogger()
        integration.stats = {"frames_processed": 0, "processing_times": []}
        absolute_timestamp = 1_783_933_507_959_540_877
        success = integration.process_frame(
            timestamp=0.2,
            timestamp_ns=absolute_timestamp,
            rgb_image=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_image=np.ones((2, 2), dtype=np.float32),
            semantic_labels=np.zeros((2, 2), dtype=np.int32),
            transform=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )
        self.assertTrue(success)
        self.assertEqual(integration.pipeline.step_args[0], absolute_timestamp)


if __name__ == "__main__":
    unittest.main()
