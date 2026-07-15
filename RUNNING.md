# Running DAAAM

This guide walks through the full CODa dataset workflow: downloading data, creating ROS bags, running the pipeline, post-processing, and visualization.

## 1. Downloading the CODa Dataset

Clone the [CODa-devkit](https://github.com/ut-amrl/coda-devkit) and download a sequence:

```bash
git clone git@github.com:ut-amrl/coda-devkit.git
cd coda-devkit

# Download sequence 0 (≈15 GB)
python scripts/download_split.py -d /path/to/CODa -t sequence -se 0
```

Pre-defined splits (`tiny`, `small`, `medium`, `full`) are also available — see the devkit README.

Set the dataset root for convenience:

```bash
export CODA_ROOT_DIR=/path/to/CODa
```

Expected directory structure after download:

```
CODa/
├── 2d_rect/          # Rectified camera images
│   ├── cam0/
│   │   └── <seq>/    # Numbered frames: 000000.png, 000001.png, ...
│   └── cam1/
├── 3d_raw/           # DO NOT USE, THIS DEPTH IS IN A DIFFERENT REFERENCE FRAME AND NOT USABLE WITH OUR PIPELINE
│   ├── cam0/
│   └── cam0_undist/
├── calibrations/     # Per-sequence camera intrinsics & extrinsics
│   └── <seq>/
├── poses/
│   └── dense_global/ # Global pose estimates (used for TF)
└── timestamps/       # Per-sequence timestamp files
```

## 1.5 Depth Estimation (Optional)

For the CODa dataset, depth can be estimated via stereo-disparity between `cam0` and `cam1`.
We estimated depth a-priori in our experiments using [FoundationStereo](https://github.com/NVlabs/FoundationStereo).
Below we provide instructions on running depth estimation with that model.
The script can be found at: [run_coda_depth_estimation.py](https://github.com/nicogorlo/FoundationStereo/blob/master/scripts/run_coda_depth_estimation.py)

```bash
cd /path/to/FoundationStereo
pip install -r requirements.txt --no-build-isolation --no-deps
pip install xformers==0.0.33 --index-url https://download.pytorch.org/whl/cu128
python scripts/run_coda_depth_estimation.py \
  --dataset_folder $CODA_ROOT_DIR \
  --sequence_id 0
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset_folder` | *(required)* | Root path to CODa dataset |
| `--sequence_id` | *(required)* | CODa sequence number |
| `--save_format` | `png` | Output format (`png` or `npz`) |
| `--depth_skip` | `1` | Process every N-th frame |
| `--batch_size` | `1` | Parallel rectification batch size |

This writes depth maps to `CODa/3d_raw_estimated/` (or the configured output directory), which the dataloader picks up via `depth_source:=3d_raw_estimated` in the next step.

## 1.6 Validated Stereo Mapping

For prepared stereo sequences, DAAAM can remove only strict duplicate frames before
depth inference. The selector keeps the original camera nanosecond timestamps and
the exact pose row associated with every retained image. In particular, an image
with new visible content is retained even when the robot pose is unchanged.

FoundationStereo is an NVIDIA research/non-commercial dependency. Initialize its
pinned submodule and create its separate environment before using the one-click
workflow:

```bash
git submodule update --init --recursive
bash scripts/setup_foundation_stereo_env.sh
export FOUNDATION_STEREO_CHECKPOINT=/path/to/model_best_bp2.pth
```

Inspect the complete command graph without changing data:

```bash
python scripts/run_stereo_mapping.py \
  --adapter prepared-stereo \
  --src /path/to/prepared_stereo_dataset \
  --run-dir output/my_stereo_run \
  --checkpoint "$FOUNDATION_STEREO_CHECKPOINT" \
  --floor-calibration-report /path/to/validated_floor_geometry_calibration.json \
  --hydra-config-path /path/to/hydra_config.yaml \
  --dry-run
```

The G1 rig needs a previously validated fixed floor/image-frame calibration.
Generate that report once from a clean nominal-depth calibration dataset as described
in `REPRODUCTION_G1_20260713_170500_FISHEYE_FIX.md`; do not estimate it independently
for every capture.

Source ROS and the DAAAM environment, then run through the direct RGB-D preview gate:

```bash
source /opt/ros/jazzy/setup.bash
source .repro/ros2_ws/install/setup.bash
source .repro/venv/bin/activate

python scripts/run_stereo_mapping.py \
  --adapter g1-fisheye \
  --src /path/to/g1_capture \
  --run-dir output/g1_map \
  --checkpoint "$FOUNDATION_STEREO_CHECKPOINT" \
  --floor-calibration-report /path/to/validated_floor_geometry_calibration.json \
  --accept-foundation-stereo-noncommercial-license \
  --hydra-config-path /path/to/hydra_config.yaml \
  --pipeline-config config/pipeline_config.yaml \
  --labelspace-path config/labels_pseudo.yaml \
  --labelspace-colors config/labels_pseudo.csv \
  --stop-after fuse \
  --resume
```

Inspect `10_direct_rgbd_fusion/direct_rgbd_fusion_preview.png`. Only after the floor,
walls, and repeated objects are coherent should Hydra be allowed to run:

```bash
python scripts/run_stereo_mapping.py \
  --adapter g1-fisheye \
  --src /path/to/g1_capture \
  --run-dir output/g1_map \
  --checkpoint "$FOUNDATION_STEREO_CHECKPOINT" \
  --floor-calibration-report /path/to/validated_floor_geometry_calibration.json \
  --accept-foundation-stereo-noncommercial-license \
  --hydra-config-path /path/to/hydra_config.yaml \
  --pipeline-config config/pipeline_config.yaml \
  --labelspace-path config/labels_pseudo.yaml \
  --labelspace-colors config/labels_pseudo.csv \
  --accept-direct-fusion-preview \
  --resume
```

The run directory contains eleven ordered stages from pinhole rectification through
Hydra. Mapping is blocked when no geometrically verified loop exists, the final
temporal depth gate fails, or the direct-fusion preview has not been acknowledged.
`mapping_run.json` records the FoundationStereo commit and checkpoint, thresholds,
stage reports, and time-contract checks.

Every dataset stage retains `sensor_time_ns`, `pose_sensor_time_ns`, source indices,
and `pose/pose_timestamps_ns.txt`. `keyframe_selection_report.json` explains every
retained or rejected input frame; output filenames may be renumbered, but capture
times and pose associations are never reconstructed from those filenames.

Render an estimated metric depth PNG as a pseudo-color depth image:

```bash
python scripts/visualize_depth.py \
  --dataset /path/to/prepared_stereo_dataset \
  --frame 600 \
  --min-depth-m 0.25 \
  --max-depth-m 3.0
```

The default `turbo` color map uses the same fixed range for every frame; black
pixels are invalid or outside the requested range. Add `--with-context` to
include RGB, an RGB/depth overlay, and a color legend. FoundationStereo depth
PNGs are read as millimeters and converted to meters automatically.

## 3. Creating ROS Bags

The dataloader node reads raw CODa files and publishes them as ROS topics, optionally recording to a bag (preferred option):

```bash
source ~/ros2_ws/install/setup.bash

ros2 launch daaam_ros dataloader_coda_with_depth.launch.yaml \
  sequence:=0 \
  dataset_path:=${CODA_ROOT_DIR} \
  bag_path:=${CODA_ROOT_DIR}/coda_0_with_depth.bag \
  depth_source:=3d_raw_estimated
```

| Argument | Default | Description |
|---|---|---|
| `sequence` | `0` | CODa sequence number |
| `dataset_path` | `/path/to/coda/data/` | Root of the CODa dataset |
| `bag_path` | auto-generated | Output `.bag` path |
| `depth_source` | `3d_raw_estimated` | Depth source directory name |
| `framerate` | `1000.0` | Playback rate (Hz) |
| `start_frame` | `0` | First frame index |
| `end_frame` | `-1` | Last frame index (`-1` = all) |
| `stride` | `1` | Frame stride |

Published topics:

| Topic | Type |
|---|---|
| `/coda/cam0/rgb_image` | `sensor_msgs/Image` |
| `/coda/cam0/depth_image` | `sensor_msgs/Image` |
| `/coda/cam0/camera_info` | `sensor_msgs/CameraInfo` |
| `/tf` | Pose transforms |

Images are resized to 640×480 (crop-resize, preserving aspect ratio) by default.

### TF Override

The CODa bag publishes `/tf_static` with `TRANSIENT_LOCAL` durability, which can be missed by nodes that start after the bag. Create a QoS override file at `~/.tf_overrides.yaml`:

```yaml
/tf_static: {depth: 1, durability: transient_local}
```

Then set the environment variable before launching:

```bash
export ROS_STATIC_TRANSFORM_BROADCASTER_QOS_OVERRIDES=~/.tf_overrides.yaml
```

## 4. Running with ROS 2

Launch DAAAM + Hydra together, then play the bag in a second terminal:

```bash
# Terminal 1: launch pipeline
source ~/ros2_ws/install/setup.bash
ros2 launch daaam_ros coda_daaam_hydra.launch.yaml scene:=coda_sequence_0

# Terminal 2: play bag
ros2 bag play /path/to/CODa/coda_0_with_depth.bag --clock -p --qos-profile-overrides-path ~/.tf_overrides.yaml
```

Once everything is loaded press play (space) on the bag.
Attention! It is important to pass `--clock` and `--qos-profile-overrides-path ~/.tf_overrides.yaml`.

Key launch arguments:

| Argument | Default | Description |
|---|---|---|
| `scene` | `coda_sequence_0` | Scene identifier (used for output directory naming) |
| `depth_scale` | `1000.0` | Depth image scale factor (mm → m) |
| `sam_model` | `fastsam/FastSAM-x-640x480.engine` | SAM model path (`.engine` or `.pt`) |
| `sam_imgsz` | `480,640` | SAM input resolution `H,W` |
| `grounding_worker` | `dam_multi_image` | Grounding backend |
| `assignment_worker` | `min_frames_max_size` | Frame selection strategy |
| `num_grounding_workers` | `1` | Parallel grounding workers |
| `query_interval_frames` | `120` | Frames between grounding queries |
| `min_obs_per_track` | `8` | Minimum observations before grounding a track |
| `depth_lb` | `0.05` | Minimum valid depth (m) |
| `depth_ub` | `20.0` | Maximum valid depth (m) |
| `exit_after_clock` | `true` | Shut down when bag finishes |

## 5. Running without ROS 2

!WARNING! This can be brittle and is not the preferred option of running the pipeline as the hydra python bindings are less actively maintained. If there are errors, please try to use the ROS2 [DAAAM-ROS](https://github.com/MIT-SPARK/DAAAM-ROS) interface.

The standalone pipeline script reads image sequences or rosbags directly:

```bash
python scripts/run_pipeline.py /path/to/dataset \
  --hydra-config coda_dataset_khronos \
  --dataset-type ImageSequenceDataset \
  --target-fps 10 \
  --output-dir output/my_run
```

Key CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `data_path` | *(required)* | Dataset path (folder or rosbag) |
| `--config` | `config/pipeline_config.yaml` | Pipeline config file |
| `--config-overrides` | | Key=value overrides (e.g. `workers.num_grounding_workers=8`) |
| `--dataset-type` | `ImageSequenceDataset` | Dataset loader class |
| `--hydra-config-path` | `coda_dataset_khronos.yaml` | Hydra integration config |
| `--sam-model` | `fastsam/FastSAM-s.pt` | SAM model path |
| `--sentence-embedding-model` | `sentence-transformers/sentence-t5-large` | Embedding model for post-processing |
| `--target-fps` | | Target processing framerate |
| `--max-frames` | | Maximum frames to process |
| `--depth-scale` | `1.0` | Depth scale factor |
| `--depth-lb` / `--depth-ub` | `0.25` / `5.0` | Valid depth range (m) |
| `--num-grounding-workers` | `4` | Parallel grounding workers |
| `--grounding-worker` | `dam_multi_image` | Grounding backend |
| `--assignment-worker` | `min_frames_max_size` | Frame selection strategy |
| `--output-dir` | auto-generated | Output directory |
| `--save-images` | `false` | Save segmentation visualization frames |
| `--dry-run` | `false` | Load config/dataset but don't run |

## 6. Post-Processing

### Sentence Embeddings

For smaller GPUs it can be difficult to load the sentence embedding model (e.g., T5-xl) directly while running the DAAAM pipeline due to limited VRAM. By default we therefore compute the sentence embedding vectors post-hoc. This step can be skipped if a `sentence_embedding_model` is set in the ROS2 launch file.

After the pipeline finishes, update the DSG with sentence embeddings for downstream tasks (e.g. NaVQA):

```bash
python scripts/postprocess_scene_graph.py \
  --data-dir output/my_run \
  --sentence-model-name sentence-transformers/sentence-t5-xl
```

Reads `dsg.json`, `corrections.yaml`, and `background_objects.yaml` from `--data-dir`. Writes `dsg_updated.json` with sentence embeddings added to all object nodes.

### Cluster Places

Clustering places into semantically meaningful regions is currently run post-hoc. This will be integrated into the flow of the pipeline in the future.
To cluster the places, run:

```bash
ros2 launch daaam_ros cluster_places.launch.yaml data_dir:=/path/to/output/data/dir/
```
This will read from the `dsg_updated.json` file and save `clustered_dsg.json` .

### Region Summaries

Generate LLM-based natural language summaries for room/region nodes:

```bash
python scripts/summarize_regions.py \
  --data-dir output/my_run \
  --openai-api-key $OPENAI_API_KEY \
  --model-name gpt-5-nano \
  --n-samples 20
```

Uses farthest-first sampling for semantic diversity within each region. Writes `region_summaries.yaml` and `clustered_dsg_with_summaries.json`.

## 7. Visualization

View the final scene graph with the Rerun-based static visualizer (requires rerun-sdk ):

```bash
python scripts/run_static_visualizer.py \
  --dsg output/my_run/dsg_updated.json \
  --color-map config/labels_pseudo.csv \
  --spawn
```

| Argument | Default | Description |
|---|---|---|
| `--dsg` | | Path to DSG JSON file |
| `--color-map` | `config/labels_pseudo.csv` | Label color map |
| `--gt-dsgs` | | Ground truth DSG(s) for comparison |
| `--log-object-meshes` | `false` | Log individual object meshes |
| `--spawn` / `--no-spawn` | `--spawn` | Open Rerun viewer automatically |
| `--z-offset-objects` | `0.0` | Z offset for object layer |
| `--z-offset-places` | `10.0` | Z offset for places layer |
| `--z-offset-rooms` | `20.0` | Z offset for rooms layer |

## 8. Pipeline Outputs

The pipeline writes results to the output directory (default: `output/out_<timestamp>/`):

| File | Description |
|---|---|
| `dsg.json` | Final Dynamic Scene Graph (Hydra format) |
| `corrections.yaml` | All semantic corrections with temporal history |
| `background_objects.yaml` | Background object layer (depth-filtered objects not in Hydra DSG) |
| `keyframe_annotations.yaml` | Per-keyframe annotation data |
| `pipeline_config.yaml` | Snapshot of the pipeline configuration used |
| `performance_statistics.csv` | Per-frame timing statistics |
| `grounding_images/` | Saved grounding visualizations (if `save_grounding_images: true`) |
| `logs/` | Pipeline log files |

## 9. Agentic Scene Understanding

Query the scene graph interactively using the agentic scene understanding pipeline. Requires a completed DSG (from §8) and an OpenAI API key.

```bash
cd scripts

python demo_query.py \
  --dsg-path /path/to/output/dsg_updated.json \
  --seq-id 0 \
  --model-name gpt-5-mini
```

This starts a REPL where you can ask free-form questions about the scene (e.g. "What objects are near the door?"). The agent uses tool-calling over the scene graph to answer.
