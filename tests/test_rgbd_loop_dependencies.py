"""Regression tests for dense RGB-D loop-closure dependencies."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
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


geometry = load_script_module(
    "build_rgbd_pose_graph_dataset_test",
    REPOSITORY_ROOT / "scripts" / "build_rgbd_pose_graph_dataset.py",
)


class RgbdLoopDependencyTests(unittest.TestCase):
    def test_scaled_intrinsic_and_dense_cloud_contract(self):
        camera = {
            "width": 64,
            "height": 48,
            "fx": 50.0,
            "fy": 52.0,
            "cx": 31.5,
            "cy": 23.5,
        }
        intrinsic, width, height = geometry.create_intrinsic(camera, 0.5)
        self.assertEqual((width, height), (32, 24))
        self.assertEqual((intrinsic.width, intrinsic.height), (32, 24))
        self.assertTrue(
            np.allclose(
                intrinsic.intrinsic_matrix,
                [[25.0, 0.0, 15.75], [0.0, 26.0, 11.75], [0.0, 0.0, 1.0]],
            )
        )

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset / "rgb").mkdir()
            (dataset / "depth").mkdir()
            rgb = np.zeros((48, 64, 3), dtype=np.uint8)
            rgb[..., 1] = 127
            depth = np.full((48, 64), 1000, dtype=np.uint16)
            self.assertTrue(cv2.imwrite(str(dataset / "rgb/00000000.png"), rgb))
            self.assertTrue(
                cv2.imwrite(str(dataset / "depth/00000000.png"), depth)
            )
            cache = geometry.DenseCloudCache(
                dataset, intrinsic, width, height, max_depth_m=3.0
            )
            cloud = cache.cloud(0)
            self.assertGreater(len(cloud.points), 700)
            self.assertIs(cache.cloud(0), cloud)

            transform, metrics = geometry.multiscale_icp(
                cloud, cloud, np.eye(4)
            )
            self.assertTrue(np.allclose(transform, np.eye(4), atol=1.0e-8))
            self.assertEqual(len(metrics), 3)
            self.assertTrue(all(fitness > 0.99 for fitness, _ in metrics))

    def test_loop_discovery_script_imports_all_dense_dependencies(self):
        module = load_script_module(
            "discover_rgbd_loop_closures_dependency_test",
            REPOSITORY_ROOT / "scripts" / "discover_rgbd_loop_closures.py",
        )
        self.assertTrue(callable(module.multiscale_icp))
        self.assertTrue(callable(module.create_intrinsic))
        self.assertTrue(hasattr(module, "DenseCloudCache"))


if __name__ == "__main__":
    unittest.main()
