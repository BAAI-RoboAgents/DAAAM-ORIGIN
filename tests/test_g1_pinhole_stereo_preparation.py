"""Regression tests for fixed G1 stereo calibration preparation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preparation = load_script_module(
    "prepare_g1_pinhole_stereo_dataset_test",
    REPOSITORY_ROOT / "scripts" / "prepare_g1_pinhole_stereo_dataset.py",
)


class G1PinholeStereoPreparationTests(unittest.TestCase):
    def test_recover_pose_cannot_rewrite_the_recorded_ransac_count(self):
        points = np.zeros((1200, 2), dtype=np.float64)
        ransac_mask = np.ones((1200, 1), dtype=np.uint8)

        def mutate_pose_mask(*_args, **kwargs):
            kwargs["mask"][:600] = 0
            return 1100, np.eye(3), np.array([[-1.0], [0.0], [0.0]]), kwargs["mask"]

        with mock.patch.object(
            preparation.cv2,
            "findEssentialMat",
            return_value=(np.eye(3), ransac_mask),
        ), mock.patch.object(
            preparation.cv2, "recoverPose", side_effect=mutate_pose_mask
        ):
            _, _, evidence = preparation.recover_stereo_pose(
                points, points, baseline=0.06
            )

        self.assertEqual(evidence["ransac_inliers"], 1200)
        self.assertEqual(evidence["ransac_inlier_ratio"], 1.0)
        self.assertEqual(int(np.count_nonzero(ransac_mask)), 1200)

    def test_fixed_report_requires_matching_geometry_and_records_provenance(self):
        K = np.array(
            [[417.0, 0.0, 691.0], [0.0, 417.0, 438.0], [0.0, 0.0, 1.0]]
        )
        distortion = np.zeros((4, 1), dtype=np.float64)
        translation = [-0.06, 0.0, 0.0]
        report = {
            "source_intrinsics": K.tolist(),
            "source_distortion": distortion.reshape(-1).tolist(),
            "estimated_stereo_calibration": {
                "camera0_R_camera1": np.eye(3).tolist(),
                "camera0_t_camera1_m": translation,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "pinhole_preparation_report.json"
            report_path.write_text(json.dumps(report))
            _, _, evidence = preparation.load_fixed_stereo_calibration(
                report_path, K, distortion, baseline=0.06
            )
            self.assertEqual(evidence["mode"], "fixed_report")
            self.assertEqual(len(evidence["report_sha256"]), 64)
            self.assertTrue(
                evidence["source_geometry_validation"]["intrinsics_match"]
            )

            mismatched_K = K.copy()
            mismatched_K[0, 0] += 1.0
            with self.assertRaisesRegex(ValueError, "intrinsics do not match"):
                preparation.load_fixed_stereo_calibration(
                    report_path, mismatched_K, distortion, baseline=0.06
                )


if __name__ == "__main__":
    unittest.main()
