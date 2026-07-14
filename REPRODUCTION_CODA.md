# CODa Sequence 0 Reproduction

This repository is fixed at commit `ebd6e3b89763849eb7bce7f75e8d08550ea7344a`.
The local reproduction uses ROS 2 Jazzy and system Python 3.12 in an isolated
workspace under `.repro/`; it does not modify the existing
`/home/user/ros2_ws` workspace.

## Reproduction plan

1. Clone DAAAM and all 17 ROS/native dependencies.
2. Record exact repository commits and build the ROS 2 workspace.
3. Install the Python/CUDA runtime in a Python 3.12 virtual environment.
4. Supply the upstream-omitted pseudo labelspace and GPU-specific FastSAM/ReID
   TensorRT engines.
5. Validate the complete 8213-frame CODa sequence 0 RGB-D ROS bag.
6. Run a short smoke test, then the full bag.
7. Validate the DSG, semantic corrections, performance log, sentence-embedding
   post-processing, and place clustering outputs.

## Fixed inputs

- CODa root: `/home/user/Code/coda-devkit/data`
- ROS bag:
  `/home/user/Code/coda-devkit/data/rosbags/coda_0_with_depth_full_20260610_192234.bag`
- Frames: 8213
- Duration: 821.270744064 seconds
- Topics: `/cam0/rgb_image`, `/cam0/depth_image`, `/cam0/camera_info`,
  `/tf`, `/tf_static`, and `/clock`
- RGB/depth size: 640 x 480
- Depth encoding and scale: `16UC1`, millimetres, `depth_scale=1000`

The existing bag already contains FoundationStereo depth and exactly matches
the topic names used by the current DAAAM-ROS CODa launch file. No data
conversion or topic remapping is required.

The resolved Hugging Face snapshot revisions are:

- `nvidia/DAM-3B`: `0797bedd98d645cd021379a4661ee233da279bba`
- `facebook/PE-Core-L14-336`: `bafb0f76541d399057e980a25947f67acec76575`
- `sentence-transformers/sentence-t5-xl`:
  `92e07434e0b0e93b36dd780c47c9b48d32fbcdaa`

The local, GPU-specific TensorRT engines have SHA-256 hashes
`416a68734b584ac3b3794e19944ce78f91323212222295673e0d69eb714a63c7`
(FastSAM) and
`929d90e4a096a0486229ae74503c13ff2972d91c23cd50a07f73b1df73a78c8d`
(ReID). They are real files in this checkout rather than links to another
workspace.

## Runtime

Check the environment and input bag:

```bash
scripts/reproduce_coda_demo.sh check
```

The wrapper uses ROS domain 42 by default; set `ROS_DOMAIN_ID` explicitly to
override it.

Start DAAAM and Hydra in terminal 1:

```bash
scripts/reproduce_coda_demo.sh launch
```

After FastSAM, ReID, DAM-3B, and PE-Core-L14-336 workers report ready, run a
60-second smoke test in terminal 2:

```bash
scripts/reproduce_coda_demo.sh play --duration 60
```

For the full run, restart the pipeline with a fresh scene/output directory and
play all 8213 frames:

```bash
SCENE_NAME=coda_sequence_0_full \
  scripts/reproduce_coda_demo.sh launch \
  log_path:=output/coda/hydra_$(date +%Y%m%d_%H%M%S)
# In terminal 2, after all workers are ready:
scripts/reproduce_coda_demo.sh play
```

Post-process the completed run:

```bash
scripts/reproduce_coda_demo.sh postprocess output/coda/out_YYYYMMDD_HHMMSS
```

Expected primary files are `dsg.json`, `corrections.yaml`,
`background_objects.yaml`, `keyframe_annotations.yaml`, `pipeline_config.yaml`,
and `performance_statistics.csv`. Post-processing adds `dsg_updated.json` and
`clustered_dsg.json`.

Validate the final graph through the Rerun visualization path without opening
a GUI:

```bash
source .repro/venv/bin/activate
python scripts/run_static_visualizer.py \
  --dsg output/coda/out_YYYYMMDD_HHMMSS/clustered_dsg.json \
  --color-map config/labels_pseudo.csv \
  --no-spawn --interlayer-edge-subsample 50
```

## Completed run (2026-07-13)

- DAAAM output: `output/coda/out_20260713_103618`
- Hydra output: `output/coda/hydra_full_20260713_1035`
- The complete 821.270744-second bag played to EOF and both ROS nodes exited
  cleanly. The pipeline's performance log records 7175 processed frames; the
  current real-time ROS configuration drops input frames when queues are busy.
- DAAAM produced 2031 semantic corrections (zero pending), 331 keyframe
  annotations, 1520 background objects, and 2054 tracked 3D positions.
- The final clustered graph reloads successfully with `spark_dsg` and contains
  5310 nodes, 8313 edges, 8 rooms, 700 objects, 839 background objects, 1035
  agents, and 2728 traversability nodes. Its mesh has 823495 vertices and
  1083240 faces.
- Sentence embeddings, place merging and room clustering completed. Place
  clustering merged 4099 traversability nodes to 2728.
- The headless Rerun path loaded and logged the complete clustered graph.

The optional `scripts/summarize_regions.py` stage calls the OpenAI API. It was
not run because it requires the user to provide an API key and incurs external
API usage; it is not required to produce the clustered DSG.

## Reproducibility records

- `.repro/ros2_ws/repos.lock.yaml`: exact native/ROS repository commits
- `.repro/requirements-runtime.txt`: Python requirements with Git commits and
  TensorRT CUDA generation fixed
- `.repro/patches/khronos-small-gicp.patch`: missing native dependency metadata
  needed for deterministic first-pass colcon build ordering
- `.repro/patches/daaam-ros-safe-shutdown.patch`: avoid calling
  `rclpy.shutdown()` twice after Hydra ends bag playback
- `.repro/pip-freeze.txt`: exact installed Python environment
- `.repro/tf_overrides.yaml`: `/tf_static` playback durability override
- `.repro/runtime_logs/`: full-run and post-processing logs

The `.repro` directory is intentionally excluded through `.git/info/exclude`
because it contains a complete build, virtual environment, and large logs.

## Upstream caveats reproduced as-is

- The upstream installer assumes DAAAM is located at `<workspace>/src/daaam`;
  running it directly from this checkout would infer the wrong workspace.
- `requirements.txt` lists both CUDA-12 and unqualified TensorRT packages. The
  local runtime uses only `tensorrt-cu12==11.0.0.114` to avoid cu12/cu13 module
  collisions.
- `labels_pseudo.yaml/csv` and model checkpoints are required but gitignored by
  the upstream repositories.
- CODa sequence 0 has no `*_undist_intrinsics.yaml`. The current upstream
  loader/bag uses the distorted camera matrix with rectified images. This
  reproduces the published code path but may introduce systematic 3D geometry
  error. A calibration-corrected experiment should regenerate the bag using
  the projection matrix and zero distortion.
- The standalone CODa loader composes the OS1-to-camera transform in the wrong
  direction. The ROS loader and the supplied bag use the correct inverse, which
  is another reason the ROS route is used here.
