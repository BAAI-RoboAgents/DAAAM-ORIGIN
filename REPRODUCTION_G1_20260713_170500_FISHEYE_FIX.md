# G1 20260713_170500 鱼眼双目修正与地图重建

本记录对应 2026-07-15 完成的修复。原始数据位于
`/home/user/datasets/20260713_170500`，修正数据位于
`.repro/datasets/20260713_170500-pinhole-sync10ms`。

## 根因

旧地图效果差由多个问题叠加造成：

- 原图声明为 Kannala-Brandt 鱼眼，旧流程却直接使用针孔公式
  `depth = fx * baseline / disparity`。
- 标定文件省略了真实双目相对旋转；原图的水平视差存在正负混杂，且独立去畸变会
  放大垂直视差，不满足 FoundationStereo 的水平正视差假设。
- `head_camera.orientation_xyzw` 字段实际存储顺序为 `wxyz`。按字段名读取会把机头
  偏航误解释为相机俯仰，导致图像帧与全局 pose 不一致。
- 标称 60.193 mm 基线与该图像流的有效立体尺度不一致。利用已知相机高度和首批
  地面深度校正后，有效基线为 68.962 mm。
- 旧流程保留到 20 m；6.9 cm 小基线在该距离产生明显长尾噪声。修正流程将可靠
  深度范围限制为 3 m。
- 1579 对图像中只有 1502 对满足 10 ms 同步门限，其余 77 对不参与建图。

## 1. 鱼眼双目转针孔

脚本从 24 对同步图像估计遗漏的双目旋转和基线方向，使用统一极线校正生成针孔
左右图；同时自动识别相机四元数顺序并组合里程计全局位姿。

```bash
cd /home/user/Code/DAAAM_Origin
source .repro/venv/bin/activate

python scripts/prepare_g1_pinhole_stereo_dataset.py \
  --src /home/user/datasets/20260713_170500 \
  --output .repro/datasets/20260713_170500-pinhole-sync10ms \
  --max-delta-ms 10 \
  --horizontal-fov-deg 100 \
  --down-fov-deg 28 \
  --recommended-max-depth-m 3 \
  --overwrite
```

关键结果：

- 1502/1579 对图像通过同步门限，匹配时间差均不超过 10 ms。
- 输出针孔内参为 `fx=fy=537.023764`、`cx=640`、`cy=673.459400`。
- 左右重映射有效率均为 100%。
- 自动选择 `wxyz` 相机四元数解释；相机向下轴的世界 Z 中位数为 -0.9973。
- 全局轨迹长度为 14.050 m。

## 2. 名义深度小批测试

先生成 5 帧名义深度，供地面尺度和图像帧姿态标定：

```bash
env -u PYTHONPATH -u VIRTUAL_ENV \
  /home/user/miniconda3/bin/conda run --no-capture-output \
  -n foundation_stereo \
  python scripts/run_foundation_stereo_depth.py \
  --dataset .repro/datasets/20260713_170500-pinhole-sync10ms \
  --fs-root /home/user/Code/FoundationStereo \
  --checkpoint /home/user/Code/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth \
  --valid-iters 32 \
  --max-frames 5 \
  --overwrite
```

`run_foundation_stereo_depth.py` 现在会拒绝未经针孔矫正的 G1 输入，并默认读取数据集
中的 `recommended_max_depth_m`。

## 3. 地面几何标定

标定脚本在左下地面 ROI 中对逆深度做 RANSAC 平面拟合，使用世界地面 `z=0` 和
相机高度同时求解固定图像帧旋转与有效基线。完成后会删除名义深度，防止不同尺度
混用。

```bash
python scripts/calibrate_g1_floor_geometry.py \
  --dataset .repro/datasets/20260713_170500-pinhole-sync10ms \
  --frame-count 5
```

本次标定结果：

- 地面 RANSAC 内点率 0.818。
- 名义地面距离 1.320 m，相机高度 1.512 m。
- 图像帧固定旋转修正 26.680°。
- 深度尺度 1.14568，有效基线从 0.060193 m 修正为 0.068962 m。

## 4. FoundationStereo 全量深度

使用第 2 节相同命令，去掉 `--max-frames 5` 和 `--overwrite` 即可补齐全量深度。

全量结果：

- 1502/1502 张深度完整，失败 0。
- 推理 1491 张，跳过前面验证过的 11 张，总耗时 1826.15 s。
- 所有文件均为 1280 x 960 的 `uint16` PNG，单位为毫米。
- 单帧有效率最小/P5/中位/P95/最大为
  0.678/0.728/0.922/1.000/1.000。
- 单帧中位深度 P5/P50/P95 为 1.031/1.911/2.242 m。
- 单帧 P95 深度最大为 2.948 m，不再出现旧流程约 10.76 m 的中位长尾。
- 全帧稀疏反投影中，低于地面 0.2 m 的点仅占 0.012%，地面 +/-15 cm 内占
  13.6%。

## 5. DAAAM/Hydra 地图重建

```bash
source /opt/ros/jazzy/setup.bash
source .repro/ros2_ws/install/setup.bash
source .repro/venv/bin/activate

python scripts/run_pipeline.py \
  .repro/datasets/20260713_170500-pinhole-sync10ms \
  --config config/pipeline_config.yaml \
  --dataset-type ImageSequenceDataset \
  --dataset-name g1_20260713_170500_pinhole_sync10ms_full \
  --depth-scale 1000 \
  --fps 10 \
  --hydra-config-path .repro/ros2_ws/src/daaam_ros/config/hydra_config/clio_dataset_khronos.yaml \
  --labelspace-path config/labels_pseudo.yaml \
  --labelspace-colors config/labels_pseudo.csv \
  --zmq-url none \
  --output-dir output/g1_20260713_170500_pinhole_sync10ms_full \
  --query-interval-frames 90 \
  --no-throttle \
  --no-progress \
  --verbose
```

实际输出目录：

```text
output/g1_20260713_170500_pinhole_sync10ms_full/out_20260715_123238
```

地图结果：

- 处理 1502/1502 帧，CV 平均 0.236 s/帧，Hydra 平均 0.101 s/帧。
- 后端图包含 656 个节点、1301 条边。
- 主层包含 294 个 objects、144 个 places、2 个 rooms、1 个 building。
- 网格包含 53,095 个顶点、71,360 个面。
- 旧网格高度范围为 0.55–6.35 m，地面带顶点为 0；新网格高度范围为
  -0.90–3.375 m，地面 +/-15 cm 顶点占 14.7%，天花板回到约 3 m。
- 新网格连通分量从旧图的 4080 降至 2028。仍有 4.66% 未标注 TSDF 背景顶点
  低于 -0.2 m，集中在独立碎片，不影响对象和房间层；可在展示时隐藏背景小分量。

## 6. 查询图和可视化

将 DAAAM 描述和 768 维 Sentence-T5 向量写回 Hydra 图：

```bash
python scripts/prepare_zed_query_dsg.py \
  --run-dir output/g1_20260713_170500_pinhole_sync10ms_full/out_20260715_123238 \
  --allow-unmatched
```

本次匹配 290/294 个对象节点、462 条描述，并生成 `dsg_updated.json`。

可视化时不要使用旧的 10/20/40 m 层偏移；所有层保持同一世界坐标，并减少跨层边：

```bash
python scripts/run_static_visualizer.py \
  --dsg output/g1_20260713_170500_pinhole_sync10ms_full/out_20260715_123238/dsg_updated.json \
  --color-map config/labels_pseudo.csv \
  --z-offset-objects 0 \
  --z-offset-places 0 \
  --z-offset-rooms 0 \
  --z-offset-buildings 0 \
  --interlayer-edge-subsample 50
```

旧/新地图顶视与侧视对比图位于：

```text
.repro/diagnostics/g1_20260713_170500_map_geometry_comparison.png
```

## 7. 绝对时间关键帧与最终几何基线

第 5 节的 1502 帧结果只验证了鱼眼和深度尺度修正。最终基线在深度推理前增加
内容安全关键帧选择，并在 Hydra 前完成 RGB-D 局部约束、闭环和全局优化。

- 1502 帧筛选为 789 帧；第一帧、最后一帧、pose 运动、静止位姿下的新内容和
  watchdog 帧均保留。
- 每帧保留 `cam0_sensor_time_ns`、`cam1_sensor_time_ns`、
  `pose_sensor_time_ns`、源图像索引和源 pose 行。
- FoundationStereo 固定到 submodule commit
  `6e8806816b533e4d13ddbb95ffa907b797060a62`，789/789 帧成功，耗时
  978.12 s，即 1.240 s/帧。
- RGB-D 窗口图使用 174 个关键帧、1023 个候选局部连接，接受 900 个约束；
  视觉残差中位数从 0.200 m 降至 0.0327 m。
- 检索和双向稠密几何验证得到一个独立重访簇；重力一致性过滤后使用
  frame 90 到 673 的闭环。
- gravity-SE3 全局优化正常收敛。闭环误差从 2.971 m / 41.849° 降至
  0.078 m / 2.541°，优化轨迹长度为 11.003 m。

最终轨迹数据集：

```text
.repro/datasets/20260713_170500-pinhole-sync10ms-timealigned-selected-floor-calibrated-rgbd-window-loop-gravity-se3
```

## 8. 时序深度门禁与直接融合

深度过滤只移除被多个时间相邻视图明确否定的深度像素，不删除 RGB、pose 或
时间戳。最终仍为 789 帧，使用相邻偏移 1/2/3、最少 3 个可判断邻居和 0.5
支持率。

最终全量相邻帧检查结果：

- 加权深度一致率 0.9102，门限 0.85。
- 相邻帧误差中位数的中位数 0.01535 m，门限 0.035 m。
- 每 100 帧局部窗口的最低一致率 0.8531，门限 0.80。
- `pose/pose_timestamps_ns.txt` 与 789 个 `cam0_sensor_time_ns` 逐行精确相等。

最终 Hydra 输入和诊断：

```text
.repro/datasets/20260713_170500-pinhole-sync10ms-timealigned-selected-floor-calibrated-final
.repro/diagnostics/g1_20260713_170500_final_temporal/temporal_depth_consistency_report.json
.repro/diagnostics/g1_20260713_170500_final_direct_fusion/direct_rgbd_fusion_preview.png
```

直接 RGB-D 融合使用 80 帧生成 488,128 个下采样点。正视、侧视中地面、墙面、
天花板和货架能够重合，因此才继续运行 Hydra。

## 9. 最终 Hydra 结果

最终运行目录：

```text
output/g1_20260713_170500_final/out_20260715_192646
```

- 处理 789/789 帧并正常退出。
- CV 平均 0.2285 s/帧，Hydra 平均 0.1040 s/帧，当前串行部分约
  0.3325 s/帧。
- 后端 DSG 实际包含 594 个节点、1106 条边；运行时 `dsg_nodes=0` 是统计接口
  未读取后端图，并非输出为空。
- Mesh 包含 81,607 个顶点和 106,047 个三角面，空间范围约
  11.2 x 8.8 x 4.55 m。
- Mesh 真实空间轮廓可辨认，但仍有 4,013 个连通分量，最大分量占 1.84%。
  当前剩余问题主要是深度空洞、TSDF 融合权重和小分量清理，不再是整条 pose
  轨迹的明显时空错位。

Mesh 正交预览：

```text
.repro/diagnostics/g1_20260713_170500_final_hydra/hydra_mesh_preview.png
```

后续实时动态语义地图的性能目标和改造顺序见
`REALTIME_DYNAMIC_MAPPING_ROADMAP.md`。
