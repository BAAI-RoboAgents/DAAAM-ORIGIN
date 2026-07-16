from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from daaam.config import PipelineConfig
from daaam.tracking import services


class _FakeBotSort:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cmc = SimpleNamespace(termination_criteria=(3, 100, 1.0e-5))


def test_realtime_profile_enables_bounded_ecc_and_batched_reid() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = PipelineConfig.from_yaml(
        str(repository_root / "config" / "pipeline_config_realtime.yaml")
    )

    assert config.tracking.cmc_method == "ecc"
    assert config.tracking.cmc_ecc_max_iterations == 20
    assert config.tracking.batch_reid_crops is True


def test_botsort_adapter_bounds_realtime_ecc_iterations(monkeypatch) -> None:
    monkeypatch.setattr(services, "BotSort", _FakeBotSort)

    adapter = services.BotSortAdapter(
        device="cuda",
        cmc_method="ecc",
        cmc_ecc_max_iterations=20,
    )

    assert adapter.tracker.kwargs["cmc_method"] == "ecc"
    assert adapter.tracker.cmc.termination_criteria == (3, 20, 1.0e-5)


def test_botsort_adapter_rejects_nonpositive_ecc_iteration_limit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(services, "BotSort", _FakeBotSort)

    with pytest.raises(
        ValueError,
        match="cmc_ecc_max_iterations must be positive",
    ):
        services.BotSortAdapter(cmc_ecc_max_iterations=0)


def test_non_ecc_tracker_does_not_require_ecc_termination(monkeypatch) -> None:
    monkeypatch.setattr(services, "BotSort", _FakeBotSort)

    adapter = services.BotSortAdapter(
        cmc_method="orb",
        cmc_ecc_max_iterations=0,
    )

    assert adapter.tracker.kwargs["cmc_method"] == "orb"


def test_batched_tensorrt_crops_preserve_boxmot_preprocessing() -> None:
    backend = SimpleNamespace(
        bindings={
            "images": SimpleNamespace(
                shape=(64, 3, 256, 128),
                dtype=np.float32,
            )
        },
        device=torch.device("cpu"),
        half=False,
    )
    image = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    boxes = np.asarray([[1.8, 2.2, 31.9, 42.1], [12.0, 7.0, 60.0, 45.0]])

    actual = services._BatchedTensorRTCrops(backend)(boxes, image)
    expected = torch.empty((2, 3, 256, 128), dtype=torch.float32)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype("int")
        crop = cv2.resize(
            image[y1:y2, x1:x2],
            (128, 256),
            interpolation=cv2.INTER_LINEAR,
        )
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        expected[index] = torch.from_numpy(crop).permute(2, 0, 1).float()
    expected = expected / 255.0
    expected = (
        expected - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    assert torch.equal(actual, expected)


def test_batched_tensorrt_crops_reject_degenerate_box() -> None:
    backend = SimpleNamespace(
        bindings={
            "images": SimpleNamespace(
                shape=(64, 3, 256, 128),
                dtype=np.float32,
            )
        },
        device=torch.device("cpu"),
        half=False,
    )

    with pytest.raises(ValueError, match="degenerate ReID bounding box"):
        services._BatchedTensorRTCrops(backend)(
            np.asarray([[4.0, 4.0, 4.0, 8.0]]),
            np.zeros((16, 16, 3), dtype=np.uint8),
        )
