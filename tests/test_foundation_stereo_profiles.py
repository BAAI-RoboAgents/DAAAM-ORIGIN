"""Configuration tests that do not require loading FoundationStereo or CUDA."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


spec = importlib.util.spec_from_file_location(
    "run_foundation_stereo_depth_test",
    REPOSITORY_ROOT / "scripts" / "run_foundation_stereo_depth.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def args(profile, **updates):
    values = {
        "profile": profile,
        "valid_iters": None,
        "scale": None,
        "precision": None,
        "torch_compile": False,
        "confidence_mode": "left-right",
        "lr_absolute_tolerance_px": 0.75,
        "lr_relative_tolerance": 0.03,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_online_and_refine_profiles_have_distinct_reproducible_settings():
    online = module.resolve_inference_profile(args("online"))
    refine = module.resolve_inference_profile(args("refine"))
    assert online["valid_iters"] == 8
    assert online["scale"] == 0.15
    assert refine["valid_iters"] == 32
    assert refine["scale"] == 1.0
    assert online["left_right_inferences_per_frame"] == 2


def test_explicit_profile_overrides_are_recorded():
    profile = module.resolve_inference_profile(
        args(
            "online",
            valid_iters=8,
            scale=0.75,
            precision="bf16",
            torch_compile=True,
            confidence_mode="validity",
        )
    )
    assert profile["valid_iters"] == 8
    assert profile["scale"] == 0.75
    assert profile["precision"] == "bf16"
    assert profile["torch_compile"]
    assert profile["left_right_inferences_per_frame"] == 1


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"valid_iters": 7}, "valid_iters"),
        ({"scale": 0.0}, "scale"),
        ({"scale": 1.1}, "scale"),
        ({"lr_absolute_tolerance_px": 0.0}, "tolerances"),
    ],
)
def test_invalid_profile_fails_before_model_loading(updates, message):
    with pytest.raises(ValueError, match=message):
        module.resolve_inference_profile(args("custom", **updates))
