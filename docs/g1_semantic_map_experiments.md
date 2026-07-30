# G1 语义地图模块实验

> 本文档只适用于 `/home/user/datasets/test_ros2_all_tf_map` 的 815–915 帧。
> `g1_20260724` 的 473–573 帧不得直接复用本文配置、共享产物或结论；请使用
> [G1 语义地图逐模块诊断实验方案 V1.1](g1_semantic_map_experiments_v1_1.md)。

## 固定数据范围

- 原始数据：`/home/user/datasets/test_ros2_all_tf_map`
- 原始帧范围：815–915（含端点，共 101 帧）
- 基线双目同步门限：10 ms
- 在该门限下可进入针孔双目基线的帧：96 帧
- 因同步门限未进入基线的帧：820、850、854、855、905
- 上述 5 帧仍保留在原始输入清单、LiDAR/相机真值、人工标注任务和输入同步消融中。

连续场景块的默认切分如下：

- calibration：815–829
- development：830–869
- stress：870–899
- held_out：900–915

这些切分避免相邻帧跨集合泄漏。正式统计前仍须根据场景内容审核并冻结
`ground_truth/split_manifest.json`；不得根据 held-out 结果反向调参。

## 一键管理

```bash
python scripts/run_g1_semantic_map_experiments.py \
  --config config/g1_semantic_map_experiment.yaml all

python scripts/run_g1_semantic_map_experiments.py \
  --config config/g1_semantic_map_experiment.yaml preflight

python scripts/run_g1_semantic_map_experiments.py \
  --config config/g1_semantic_map_experiment.yaml \
  run --experiment E0 --repeat 1
```

最后一条命令默认只写不可变运行规格。确认预检通过后追加 `--execute` 才会执行。
可用 `--variant` 和 `--repeat` 只运行矩阵中的一个单元。
硬门失败后，管理器会拒绝正式执行；仅为定位问题时可显式追加
`--execute --diagnostic`，其运行规格会保留失败门和“非正式”标记。

## 数据保留契约

每个运行目录固定包含：

```text
manifest/         运行规格、命令、状态和失败原因
configs/          本次运行实际使用的 FastSAM、语义和 Hydra 配置副本
logs/             标准输出、错误和服务日志
telemetry/        逐帧、队列、资源、语义、跟踪、绑定和 DAM JSONL
frame_artifacts/  逐帧中间数值产物
stage_reports/    阶段汇总、门限和诊断报告
map_artifacts/    深度、位姿、静态深度、DSG、mesh、MapMemory 等
visualizations/   逐帧过程图、叠加图、montage 和轨迹图
analysis/         与真值对齐后的指标和误差样本
```

正式实时运行强制启用：

- `--experiment-telemetry`
- `--experiment-visualizations --visualization-stride 1`
- 静态深度、动态 mask、unknown mask 全量写盘
- 双目图、深度色图、置信度图、运动残差原始 NPY/色图、动静叠加图
- RGB/深度/动静/静态深度四宫格
- 语义标签逐帧彩图
- DAM grounding、plain grounding、object crop 全量图像
- XY 轨迹 PNG 和轨迹点 CSV
- 全运行文件 SHA-256、字节数和相对路径清单

Fast-FoundationStereo 运行同时启用 `--save-raw-products`，保留视差、右目视差、
置信度、一致性、遮挡、深度和各类 overlay。

## 已生成的共享证据

`experiments/g1_semantic_map/` 下已经生成：

- `registry/experiment_manifest.json`：代码、环境和 815–915 输入哈希
- `registry/experiment_registry.jsonl`：E0–E18、Q1、三重复的完整矩阵
- `ground_truth/annotation_tasks.jsonl`：101 帧人工实例/轨迹/DSG 绑定任务
- `ground_truth/manual_annotation_package/`：101 帧左目/LiDAR/右目三联标注图及真值模板
- `ground_truth/lidar_camera_815_915/`：101 帧运动补偿 LiDAR 对应、稀疏深度、
  mask、overlay 和逐点 NPZ
- `shared_inputs/right_rectification_lidar_815_915/`：右目 LiDAR 标定的训练记录、
  原始/校正极线图和 held-out 误差
- `shared_inputs/offline_prepared/01_pinhole/`：96 帧正式针孔输入
- `shared_inputs/offline_prepared/02_selected/`：92 帧默认关键帧结果
- `shared_inputs/prepared_stereo_geometry_audit_visual/`：96 帧逐帧匹配图和几何报告
- `shared_inputs/keyframe_selection_visualizations/`：逐帧选择图、同步曲线和轨迹图
- `shared_inputs/offline_prepared/02_selected/raw_*`：92 帧原始视差、深度、
  左右一致性误差、置信度、遮挡和深度叠加图
- `shared_inputs/depth_lidar_diagnostic/`：92 帧逐帧 LiDAR/双目误差叠加图、
  原始误差样本、大型 gallery 和汇总图
- `shared_inputs/floor_calibration_visualizations/`：92 帧地面 ROI/拟合拒绝图和
  结构化失败图
- `reports/experiment_preflight.json`：模型、环境、真值和几何硬门状态
- `reports/available_experiments/`：已完成输入、标定和关键帧消融的 JSON 与总览图

固定中英双语开放集查询位于 `config/g1_semantic_query_set.yaml`。最终 DSG 准备好后，
`scripts/evaluate_semantic_query_set.py` 会保存每条查询的完整排名、开放集拒绝理由、
延迟、top-1 margin 和校验后的本地证据图副本。

失败的门限结果必须保留，不能被后续成功运行覆盖。新候选应写入新的 run ID。

正式输入已切换到**纯视觉右目**几何通过子集：

- `shared_inputs/offline_prepared/01_pinhole_visual_geometry_pass`：78 帧
- `shared_inputs/prepared_stereo_geometry_audit_visual`：审计 **57/57** 通过
  （21 帧因匹配不足跳过，不计入失败）
- 旧 LiDAR 右目 0/96 失败审计保留为
  `prepared_stereo_geometry_audit_visual_lidar_right_failed`
- 地面标定报告：`shared_inputs/floor_calibration_report.json`
  （在 visual-only selected 上接受 88/92 平面，`depth_scale≈0.937`）
- 失败地面诊断仍保留：`floor_calibration_diagnostic.json`

人工语义真值：标注包、连续窗口、双人复核子集与 LiDAR 连通域提案已就绪；
提案**不是**正式人工真值。`preflight.ready` 可在几何/地面通过后为 true，而
`ready_for_semantic_eval` 仍要求人工 `complete/reviewed`。精确状态以
`reports/experiment_preflight.json` 与 `ground_truth/STATUS.json` 为准。

## 统计约束

- 正式候选和基线至少各运行 3 次，种子为 0、1、2。
- bootstrap 和显著性检验以运行、连续场景块、对象轨迹为统计单位。
- 像素只用于计算帧内误差，不能作为独立样本夸大显著性。
- 几何消融先过输入、时间、深度和 LiDAR 硬门，再进入语义归因。
- 语义消融必须同时报告几何安全性、实例/轨迹质量、DSG 绑定和查询证据效用。
