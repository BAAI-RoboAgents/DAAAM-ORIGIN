# ZED 260713 真机数据复现记录

本记录对应一次已经完成的真实运行，而不是 dry-run。输入为
`/home/user/datasets/zed260713-160732`，FoundationStereo 仓库为
`/home/user/Code/FoundationStereo`，DAAAM 仓库为
`/home/user/Code/DAAAM_Origin`。运行日期为 2026-07-13。

## 1. 数据完整性门禁

在启动 GPU 推理前完成了以下检查：

- 数据格式为 `zed_sequence_v1`，序列号为 `000000`。
- 左目、右目、全局位姿和时间戳均为 964 条，帧号一一对应。
- 图像为 640 × 480，采样频率为 15 Hz，轨迹时长约 65.64 s。
- 标定文件完整：`fx=465.971611465`、`fy=465.661789914`、
  `cx=317.473802395`、`cy=236.899595869`、双目基线
  `0.062972612679 m`。
- 全部帧的 ZED 跟踪状态正常，置信度为 100；`pose_reset_count=0`，
  全局位姿有实际运动，位姿步长中位数约为 0.0293 m。
- 原始数据虽然包含 ZED NEURAL 深度，但本次没有复用该深度，实际重新运行了
  FoundationStereo。

检查通过后，使用软链接建立隔离的数据目录：

```bash
cd /home/user/Code/DAAAM_Origin
python scripts/prepare_zed_foundation_dataset.py \
  --src /home/user/datasets/zed260713-160732 \
  --dst .repro/datasets/zed260713-160732-foundation
```

该步骤不会修改原始 ZED 数据。

## 2. FoundationStereo 实际推理

使用 checkpoint：

```text
/home/user/Code/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth
```

执行命令：

```bash
cd /home/user/Code/DAAAM_Origin
conda run -n foundation_stereo \
  python scripts/run_foundation_stereo_depth.py \
  --dataset .repro/datasets/zed260713-160732-foundation \
  --fs-root /home/user/Code/FoundationStereo \
  --checkpoint /home/user/Code/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth \
  --valid-iters 32 \
  --max-depth-m 20
```

先运行了 10 帧 smoke test，再利用脚本的断点续跑功能完成剩余 954 帧：

- 总计生成 964/964 张 16-bit PNG 毫米深度图，失败 0 张。
- smoke test 用时约 4.0 s；剩余帧用时 324.74 s，总推理时间约 329 s。
- 输出目录为 `.repro/datasets/zed260713-160732-foundation/depth/`，约 189 MiB。
- 0–5 m 范围内的每帧有效像素比例最小值/中位数/最大值约为
  0.6302/0.9365/1.0000。
- 每帧深度中位数的 P5/P50/P95 为 1.760/2.324/2.907 m。
- 抽查 5 帧与 ZED NEURAL 深度对比，FoundationStereo/ZED 的尺度中位数为
  0.991–1.005，绝对相对误差中位数为 1.3%–2.8%。该比较只用于合理性检查，
  后续 Hydra 使用的是 FoundationStereo 输出。

运行日志：

- `.repro/runtime_logs/zed260713_foundation_smoke.log`
- `.repro/runtime_logs/zed260713_foundation_full.log`
- `.repro/datasets/zed260713-160732-foundation/foundation_stereo_run.json`

RGB/深度诊断图位于
`.repro/diagnostics/zed_foundation_000482_rgb_depth.png`。

## 3. DAAAM + Hydra 全量运行

本次使用 FastSAM TensorRT 分割、CLIP TensorRT ReID、NVIDIA DAM-3B 描述模型、
FoundationStereo 深度和 ZED 全局位姿。完整命令为：

```bash
cd /home/user/Code/DAAAM_Origin
source /opt/ros/jazzy/setup.bash
source .repro/ros2_ws/install/setup.bash
source .repro/venv/bin/activate

python scripts/run_pipeline.py \
  .repro/datasets/zed260713-160732-foundation \
  --config config/pipeline_config.yaml \
  --dataset-type ImageSequenceDataset \
  --dataset-name zed260713_foundation_full \
  --depth-scale 1000 \
  --fps 15 \
  --hydra-config-path .repro/ros2_ws/src/daaam_ros/config/hydra_config/clio_dataset_khronos.yaml \
  --labelspace-path config/labels_pseudo.yaml \
  --labelspace-colors config/labels_pseudo.csv \
  --zmq-url none \
  --output-dir output/zed260713_foundation_full \
  --query-interval-frames 90 \
  --no-throttle \
  --no-progress \
  --verbose
```

实际输出目录为：

```text
output/zed260713_foundation_full/out_20260713_180947
```

全量结果：

- 处理 964/964 帧，进程正常退出；墙钟时间约 192 s。
- CV 平均耗时 0.089 s/帧，Hydra 平均耗时 0.052 s/帧。
- DAAAM 保存 485 个标签记录，其中 443 个具有有效自然语言描述。
- Hydra 后端图实际可加载，含 590 个节点、1,141 条边和网格。
- 分层节点为：259 个物体、138 个 places、1 个 room、1 个 building；此外包含
  113 个 traversability 节点和 78 个附加物体层节点。
- 网格含 90,123 个顶点和 114,831 个三角面。

`processing_stats.json` 中的 `DSG nodes: 0` 不是空图。Standalone runner 没有从
Hydra 回填节点计数，Hydra 自己保存的
`hydra_output/backend/dsg_with_mesh.json` 已通过 `spark_dsg` 反序列化验证，以上
590/1,141 统计来自实际图文件。

完整日志为 `.repro/runtime_logs/zed260713_daaam_full.log`。

## 4. 文本查询语义后处理

将 DAAAM 描述用 `sentence-transformers/sentence-t5-large` 编码为 768 维向量，并
写回 Hydra 图：

```bash
source /opt/ros/jazzy/setup.bash
source .repro/ros2_ws/install/setup.bash
source .repro/venv/bin/activate

python scripts/prepare_zed_query_dsg.py \
  --run-dir output/zed260713_foundation_full/out_20260713_180947
```

输出 `dsg_updated.json`。259/259 个 Hydra 物体节点都匹配到了 DAAAM 描述，并
写入自然语言、时间历史和 768 维句向量；图级 feature 表保存了 443 个描述。

## 5. 房间聚类与可视化验收

非交互聚类命令：

```bash
source /opt/ros/jazzy/setup.bash
source .repro/ros2_ws/install/setup.bash
ROS_DOMAIN_ID=73 ros2 launch daaam_ros cluster_places.launch.yaml \
  data_dir:=/home/user/Code/DAAAM_Origin/output/zed260713_foundation_full/out_20260713_180947 \
  interactive:=false
```

该命令正常生成 `clustered_dsg.json`，并将 traversability 节点由 113 合并为 83。
但这段短轨迹中的 traversability 节点没有有效语义 feature，聚类器没有形成新的
room，并移除了输入中的单个 room。因此，本次推荐的最终图是保留单房间结构的
`dsg_updated.json`；`clustered_dsg.json` 仅保留为聚类实验产物，不作为更优结果。

无界面 Rerun 验收命令：

```bash
source /opt/ros/jazzy/setup.bash
source .repro/ros2_ws/install/setup.bash
source .repro/venv/bin/activate
python scripts/run_static_visualizer.py \
  --dsg output/zed260713_foundation_full/out_20260713_180947/dsg_updated.json \
  --color-map config/labels_pseudo.csv \
  --no-spawn \
  --interlayer-edge-subsample 50
```

入口正常退出并成功记录全部 590 个节点、1,141 条边和主网格。需要打开 GUI 时将
`--no-spawn` 改为 `--spawn`。

## 6. 最终产物

- 推荐查询图：`output/zed260713_foundation_full/out_20260713_180947/dsg_updated.json`
- Hydra 原始带网格图：
  `output/zed260713_foundation_full/out_20260713_180947/hydra_output/backend/dsg_with_mesh.json`
- Hydra 网格：
  `output/zed260713_foundation_full/out_20260713_180947/hydra_output/backend/mesh.ply`
- DAAAM 描述：
  `output/zed260713_foundation_full/out_20260713_180947/out_20260713_180949/corrections.yaml`
- 聚类实验图：
  `output/zed260713_foundation_full/out_20260713_180947/clustered_dsg.json`

本次完整输出约 712 MiB，FoundationStereo 隔离深度目录约 193 MiB。

## 7. 已知非致命信息

- 启动日志中存在 Hydra 类型的重复注册提示，以及缺少
  `TraversabilityVisualizer`、`RosMetaDataListener` 的提示。这两个插件只影响当前
  standalone 进程内的可选可视化/元数据 sink；964 帧处理、Hydra 后端图保存和
  `spark_dsg` 加载均成功。
- 642 条短时跟踪中有 157 条在关机诊断时仍未标注，但最终进入 Hydra 物体层的
  259 个节点均有有效描述和句向量。
