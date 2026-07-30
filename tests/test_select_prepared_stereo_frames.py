from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_selected_view_rewrites_poses_and_links_frame_artifacts(tmp_path):
    dataset = tmp_path / "prepared"
    (dataset / "pose").mkdir(parents=True)
    (dataset / "depth").mkdir()
    (dataset / "depth_metadata").mkdir()
    (dataset / "camera_info.json").write_text('{"fx": 100.0}\n')
    frames = [
        {
            "idx": index,
            "source_idx": 100 + index,
            "pose_row": index,
            "cam0": f"/rgb/{index}.png",
            "cam1": f"/right/{index}.png",
            "stereo_delta_ms": float(index),
        }
        for index in range(3)
    ]
    (dataset / "tick_index.json").write_text(
        json.dumps({"projection_model": "pinhole", "frames": frames})
    )
    poses = [
        " ".join(str(value) for value in range(index * 16, index * 16 + 16))
        for index in range(3)
    ]
    (dataset / "pose/poses.txt").write_text("\n".join(poses) + "\n")
    for index in range(3):
        (dataset / "depth" / f"{index:08d}.png").write_bytes(b"depth")
        (dataset / "depth_metadata" / f"{index:08d}.json").write_text("{}")

    output = tmp_path / "selected"
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/select_prepared_stereo_frames.py"),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--source-indices",
            "102",
            "100",
        ],
        check=True,
    )
    selected = json.loads((output / "tick_index.json").read_text())
    assert [frame["source_idx"] for frame in selected["frames"]] == [102, 100]
    assert [frame["pose_row"] for frame in selected["frames"]] == [0, 1]
    assert (output / "pose/poses.txt").read_text().splitlines() == [
        poses[2],
        poses[0],
    ]
    assert (output / "depth/00000000.png").is_symlink()
    assert (output / "depth/00000000.png").resolve() == (
        dataset / "depth/00000002.png"
    )
    assert (output / "depth_metadata/00000001.json").resolve() == (
        dataset / "depth_metadata/00000000.json"
    )
