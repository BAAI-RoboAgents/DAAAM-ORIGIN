# 地图构建核心流程：从双目图像到语义地图

> 当前 G1 实验阶段（2026-07-28）：暂时跳过人工数据标注，使用现有数据对构建链路逐环节
> 做证据优先的细粒度诊断。阶段目标、`exact/proxy/unavailable/not_applicable` 结论等级
> 和全量中间产物保留要求见
> [docs/g1_semantic_map_diagnostic_no_gt_stage.md](docs/g1_semantic_map_diagnostic_no_gt_stage.md)，
> 执行配置见
> [config/g1_semantic_map_diagnostic_no_gt.yaml](config/g1_semantic_map_diagnostic_no_gt.yaml)。
> 本轮现有产物诊断已完成，证据总账见
> [EVIDENCE_LEDGER.md](experiments/g1_20260724_473_573_v1_1/diagnostic_no_gt/EVIDENCE_LEDGER.md)。
> 这不会把自动输出提升为 GT，也不会取消正式 V1.1 的双人标注、裁决和 held-out 条件。

本文依据当前仓库代码和配置，说明从实时或离线双目图像到 Hydra Dynamic
Scene Graph（DSG）语义地图的实际处理链路、每一步必须提供或可以调整的参数、
当前默认值、质量门和输出产物。

> 本文同时区分“当前代码具备的能力”和“已有运行证据验证过的能力”，不是理想架构。
> 尤其需要注意：
>
> 1. `scripts/run_stereo_mapping.py` 是高质量离线几何编排入口，但其默认 `map`
>    输出路由当前有确定性缺陷，且完成检查不是严格语义完成门；
> 2. `scripts/run_realtime_mapping.py` 是带有界队列的准实时数据集回放入口；
> 3. 当前仓库内的准实时入口仍读取已经整理好的针孔双目数据集，不直接订阅相机
>    ROS topic；真机相机、IMU、在线 VIO 的接入属于 DAAAM-ROS/部署层；
> 4. “完成语义地图”和“完成可查询语义地图”不是同一个终点。后者还要生成查询
>    embedding 和 manifest；FastSAM 证据资产是再下一步的可选增强。
> 5. 当前核心流程提供两个可选深度后端：默认使用 **Fast-FoundationStereo**（全分辨率、
>    8 iterations、FP16、左右一致性检查）；FoundationStereo 保留为兼容、复现和 A/B
>    对比方案。新建图任务不再使用 5.0 m 业务截断，`uint16` 毫米深度的格式上限
>    **65.535 m** 仅用于防止存储溢出。准实时 `foundation-worker` 仍只实现后者。
> 6. `2d_rect` 路径、`D=0` 和 `roi.do_rectify=true` **只能证明当前像素已经完成单目
>    去畸变，不能单独证明左右目已经完成联合立体校正**。2026-07-28 对
>    `g1_20260724` 的 653–953 帧实验确认：输入 PNG 仍有系统性垂直极线误差，必须先应用
>    经过留出帧验证的双目校正，才可送入 FoundationStereo。禁止再次应用 fisheye
>    去畸变；是否需要立体校正必须由匹配点实验判定，不能由目录名猜测。
> 7. 后续语义地图构建**不再安排 1 Hz、3 Hz、5 Hz 等回放频率 A/B 或时效测试**。
>    `--rate-hz` 只作为单次正式构建的调度和资源保护参数，沿用任务已确认的稳定值，
>    不得为了比较频率重复运行地图构建。只有用户明确要求性能基准时才允许做频率测试；
>    此类测试必须使用独立 `run-dir`，且不能替代或阻塞正式全量语义地图交付。

## 1. 两条当前可用链路

### 1.1 高质量离线链路

几何编排入口：[scripts/run_stereo_mapping.py](scripts/run_stereo_mapping.py)

默认深度入口：[scripts/run_fast_foundation_stereo_depth.py](scripts/run_fast_foundation_stereo_depth.py)

```text
已单目去畸变的 G1 双目，或已准备好的针孔双目
  -> 判定是否已经水平立体校正
  -> 必要时应用留出验证通过的双目校正（不重复 fisheye 去畸变）
  -> 双目同步、位姿对齐和可追溯的数据整理
  -> 内容安全关键帧选择
  -> Fast-FoundationStereo 深度和左右一致性（取消 5.0 m 业务截断）
  -> 固定地面/图像坐标校准
  -> 输入深度时序诊断
  -> 局部 RGB-D 里程计
  -> 几何闭环发现
  -> 全局位姿图优化
  -> 时序深度过滤
  -> 最终深度质量门
  -> 直接 RGB-D 融合预览和人工验收
  -> FastSAM + BotSort + DAM + Hydra/Khronos
  -> Hydra 几何产物 + DAAAM 语义 side products
```

这两个入口共同构成默认 Fast 离线流程；原版 FoundationStereo 仍可由几何 wrapper 直接
调用。原始离线双目到旧式 DAAAM/Hydra `map` 至少有四类显式硬门：

- 默认 Fast-FoundationStereo 的仓库、checkpoint 和相邻 `cfg.yaml` 必须通过固定 commit 与
  SHA-256 校验，并遵守其研究/非商用许可证；选择 FoundationStereo 时则使用对应的原版
  checkpoint、`cfg.yaml` 和许可证确认；
- G1 在深度之后继续处理时，必须提供已验证的固定地面/相机标定报告；
- 必须发现至少一个经过几何验证的闭环；
- 必须人工检查直接 RGB-D 融合预览，再显式允许 Hydra 建图。

它追求几何质量和可审计性，不是实时路径；原始 DAAAM/Hydra `map` 阶段仍按帧串行
执行 CV 前端和 Hydra，并且不会先剔除运动像素。

当前还有一个确定存在的输出路由缺陷：`run_stereo_mapping.py` 把 map 输出目标设为
`<run-dir>/11_daaam`，但它调用的 `run_pipeline.py` 又固定启用 ROS 风格日志布局，
`HydraPipelineRunner` 在 wrapper 总会提供 `dataset_name` 的情况下，无条件把实际输出改到
仓库 `output/<dataset-name>/out_<t1>/`；DAAAM orchestrator 又在其下创建
`out_<t2>/`。wrapper 随后仍去 `<run-dir>/11_daaam` 查找文件，所以默认 `map` 命令即使
已经生成 Hydra 地图，也会确定性地判定失败，`mapping_run.json` 不会成为
`complete`。修复目录优先级前，直接运行 `run_pipeline.py` 时也只有加
`--no-logging --output-dir <path>` 才会稳定尊重显式目录；否则应以终端打印的
`runner.output_dir` 为准。

此外，离线 wrapper 的 `map_ready` 只检查 `mesh.ply` 和 `dsg_with_mesh.json` 是否存在，
不检查 DAM 是否全部 drain、correction 是否全部 ACK、是否仍有 pending，也不会保证语义
已经同步写回带 mesh 的 DSG。因此它当前只能算“Hydra 几何产物存在门”，不能作为严格的
语义地图完成证明。需要严格语义提交和质量验收时，应使用下述准实时 Hydra + DAM 路径。

### 1.2 准实时动态语义链路

入口：[scripts/run_realtime_mapping.py](scripts/run_realtime_mapping.py)

```text
已准备的针孔双目 + 绝对时间 + 已提供的相机位姿
                    |
                    +-> pose -> depth -> dynamic -> fusion -> global/Hydra
                    |
                    +-> FastSAM -> BotSort -> MapMemory -> DAM（异步）
                                                    |
                                                    -> 逐帧语义标签后处理
                                                    -> 最终静态语义 Hydra DSG
```

主几何链使用有界队列。运动和不确定像素会在进入静态 TSDF 前被清零；FastSAM、
BotSort 和 DAM 是不阻塞几何链的语义旁路。使用 `--static-map-backend hydra`、
`--stop-after global` 和 `--semantic-mode dam` 时，可以在一次命令中从“已准备数据集”
得到动态像素隔离后的静态语义 Hydra DSG。动态实体的状态和轨迹目前只持久化到 checkpoint，
尚未导出为最终 Hydra DSG 的动态节点或轨迹边。

“准实时”在这里指按原始绝对时间进行限速回放预计算深度，或使用可选的原版
FoundationStereo worker；它不是当前仓库内的原始 ROS 相机 topic 接口。默认 Fast 后端
当前采用预计算方式，不由 `foundation-worker` 生成。回放保留
`sensor_time_ns`，墙钟派发则同时受 source delta 和 `--rate-hz` 控制，默认每帧至少间隔
`1/rate_hz`，不等于按采集墙钟 1:1 重放。

这里的 `--rate-hz` 是执行参数，不是语义地图质量参数。正常建图只选择一个已确认值完成
一次全量运行，不再先后试跑 1 Hz、3 Hz 等频率来判断是否能够建图，也不以这类时效试验
作为语义完成门。排障应依据逐 stage 延迟、队列、内存、drop/error、postpass 和质量报告，
不能通过反复降低或切换回放频率替代根因分析。

### 1.3 参数来源和优先级

当前参数分散在三层，运行报告应保存三层的最终组合：

1. 入口脚本 CLI：数据路径、运行模式、几何门、频率和本次覆盖值；
2. DAAAM pipeline YAML：FastSAM、BotSort、assignment、DAM 和对象深度门；
3. Hydra YAML：TSDF、3D 对象、place、room 和 backend。

`run_pipeline.py` 先加载 pipeline YAML，再应用显式 CLI 字段，最后应用
`--config-overrides key=value`，因此 `--config-overrides` 优先级最高。准实时入口中的
`--semantic-minimum-observations`、分割频率和队列参数由实时 adapter 直接使用，并不由
`pipeline_config_realtime.yaml` 中同名或相近的 assignment 字段覆盖。Hydra 参数只来自
`--hydra-config-path` 指向的 YAML，不应在 pipeline YAML 中寻找 TSDF 体素设置。
MapMemory 的跨 track 合并距离和最终 DSG object mesh 绑定距离也由准实时入口 CLI 控制；
桌面配置必须同时传入 `--entity-merge-distance-m` 和两个
`--object-binding-maximum-*-m` 参数，单独替换 pipeline/Hydra YAML 不会覆盖它们。
`run_stereo_mapping.py` 当前不转发 `--config-overrides`；通过该一键入口运行时，应复制并
修改 pipeline YAML，或使用它已经暴露的 CLI 覆盖项。

### 1.4 当前“一键完成”的能力边界

| 起点和终点 | 当前结论 |
| --- | --- |
| 原始离线 G1 双目 -> 几何验收预览 | 对真正已立体校正的输入，默认 Fast 方案仍由两个脚本三段式编排。对 `g1_20260724`，先用 `materialize_g1_v1_v2_rectified_dataset.py` 把已验收组合物化为完整 `prepared-stereo`，再进入 Fast 深度和几何链 |
| 原始离线 G1 双目 -> 旧式 DAAAM/Hydra map | `g1_20260724` 已完成组合校正物化、LiDAR 尺度、几何链、人工预览和一次全量 Hydra + DAM 构建；旧 wrapper 的默认 map 输出路由及弱完成检查仍未改变，严格结论应读取准实时运行的 postpass、durable commit、质量报告和独立验证 |
| 已准备针孔双目、预计算深度、pose/time -> 动态隔离后的静态语义 Hydra DSG | 默认 Fast-FoundationStereo，原版可选；代码路径可由 `run_realtime_mapping.py --depth-backend precomputed --semantic-mode dam --stop-after global` 一次闭合；是否验收通过还要看第 8 节的 postpass、commit 和质量报告 |
| 实时 ROS 双目/IMU topic -> 在线 VIO -> 语义地图 | 当前仓库内尚未由一个脚本闭合，传感器和 VIO 接入仍在部署/DAAAM-ROS 层 |
| 最终语义 DSG -> 中英文可查询 | 视绑定状态执行 rebind，然后生成 embedding + manifest |
| 可查询 DSG -> FastSAM/RGB-D 证据资产 | 可选再执行 evidence 后处理 |

### 1.5 当前验证快照

下列结论只对各自产物记录的代码、模型和配置哈希成立，不能自动外推到当前 dirty
worktree：

| 证据 | 已验证 | 未覆盖/限制 |
| --- | --- | --- |
| `g1_20260724` 653–953 帧 V1/V2 组合校正实验 | 240 帧训练、61 帧独立留出；留出汇总 `abs(dy)` P50/P95 为 `0.195/0.859 px`，正视差 `99.00%`，严格通过 `56/61`，共同有效图像面积 `94.84%`；组合验收 PASS | 证明极线对齐和正视差方向，不等于稠密深度绝对精度证明；物化脚本可以消费 `best_combination.json`，但 V2 的 `60.193 mm` 标称基线仍必须经 LiDAR/已知距离验证 |
| clean 版本 `ab047db` 的 789 帧权威运行 | 789/789 completed，0 drop/error/resume，7/7 hard gates、32/32 validator，真实 FastSAM/BotSort/DAM，`authoritative=true` | 使用旧的预计算深度；早于当前 exact-frame semantic postpass/query 与 Fast-FoundationStereo 默认方案，不作为新默认深度链验收证据 |
| 历史 `output/g1_20260717_1hz_dam_meshbound_opt` 功能性全量运行（仅归档） | 1098/1098、0 drop，预计算的 refine 深度（32 iterations、scale 1.0、FP16），exact-frame label postpass 完成，7 个 hard gate PASS，83 个 DSG correction operations applied | 仍有 178 个 `rejected_no_mesh`；operation 数不等于 mesh-bound 实体数；运行时 worktree dirty，validator 为 `authoritative=false`，且文件哈希不等于当前工作区；不得据此恢复 1 Hz/3 Hz 频率对比流程 |
| 5 cm/3 cm 几何 A/B 和查询资产 | 两组均完成 1098 帧重放；5 cm 结果生成 49 个 mesh-bound、167 个 spatial-only 查询实体，216/216 有图像证据，其中 214 个有 RGB-D 点云 | 3 cm 未增加 mesh-bound 数，网格更大且连通性更差，所以 5 cm 仍是当前基线，3 cm 只是实验配置 |
| `g1_260720_1424_indoor_fast_hq` 的 Fast 深度运行 | 1575/1575、0 failed；全分辨率、8 iterations、FP16、双向左右检查、5.0 m 截断；平均有效率 68.14%，完整运行端到端 964.7 s | 运行前 GPU 已被其他进程占用，计时不是无争用基准；原始采集没有稠密深度 GT，模型间差异不能当作绝对精度 |
| `g1_260720_1424_indoor_fast_semantic_10cm` 的最终语义运行 | 1575/1575、quality/benchmark PASS、durable commit 有效；74 个 described mesh-bound objects，Fast 候选运行约 972.9 s | 当前 worktree/benchmark 非权威；与旧基线还同时存在 Hydra 几何配置差异，不能把全部变化只归因于深度模型 |
| `g1_20260724_v1_v2_semantic_map` 的 V1/V2 + LiDAR 尺度全量运行 | 844/844 完成、0 drop/error；exact-label postpass 844/844、DAM drain PASS；153 个 Hydra objects、50 个 mesh-bound 描述实体、59 个 durable operations；最终 mesh 为 48,891 顶点/65,182 面，DSG commit 哈希验证通过 | 外部 SIGTERM 后从 600 帧检查点恢复，因此独立权威验证的 zero-resume/full-dispatch 门不通过；语义对象重建使 `global` service P95 达到 `305.46 ms`，超过标准 `250 ms` 门，故 `quality_passed=false`。这是可审计的功能性地图，不得误报为权威质量 PASS |

截至当前，尚未完成原始 ROS 双目/IMU 直接接入、真机在线 VIO、Fast-FoundationStereo
在线 request/response worker 接入，以及连续 30 分钟部署稳定性验收。历史验证运行使用
5.0 m 上限；新运行必须保存独立 provenance，并显式使用 65.535 m 存储上限取消该业务
截断，再由准实时链回放。需要在线逐帧深度时可选择原版 FoundationStereo worker，但它
不是默认方案。

## 2. 输入数据和时间契约

### 2.1 高质量离线入口的两种输入

| `--adapter` | 输入 | 后续行为 |
| --- | --- | --- |
| `g1-fisheye` | G1 双目、标定、时间戳和机器人/头部位姿；`g1_20260724/2d_rect` 实际为已单目去畸变但未完成水平立体校正的针孔像素 | 不得依赖 `auto` 或目录名；原始 PNG 诊断时显式使用 `pinhole_unrectified`。正式深度应先物化经过留出验证的组合校正结果，再以 `prepared-stereo` 进入后续流程 |
| `prepared-stereo` | 已经满足下述时间契约的针孔双目数据集 | 跳过 G1 鱼眼准备，但仍执行选择、深度、几何优化和建图 |

进入关键帧选择和 Fast-FoundationStereo 之前，核心结构为：

```text
dataset/
  rgb/                         # 已校正左目原图，也是 Hydra/DAAAM 使用的 RGB
  stereo_right/                # 已校正右目原图
  pose/poses.txt               # 每行一个 4x4 world_T_camera，16 个数
  pose/pose_timestamps_ns.txt  # 每个 pose 行的绝对纳秒时间
  camera_info.json             # 针孔内参
  tick_index.json              # 帧、左右目、pose 和原始来源的绑定
```

Fast-FoundationStereo 步骤随后新增 `depth/`（uint16 毫米）、`depth_confidence/`、
`depth_consistency/`、`depth_occlusion/` 和 `depth_metadata/`。这些目录不是
`prepared-stereo` 入口的先验必需输入。现有脚本若未写 `recommended_max_depth_m` 或未传
`--max-depth-m`，仍会回退到旧默认 `5.0 m`，所以新运行必须显式传
`--recommended-max-depth-m 65.535 --max-depth-m 65.535`，并把所有下游几何上限同步
设为 `65.535 m`。这表示取消 5 m 业务截断；65.535 m 只是 `uint16` 毫米格式可表达的
最大值，不代表模型在该距离仍有可靠精度。

`tick_index.json` 每帧的 `cam0` 和 `cam1` 字段分别指向 `rgb/` 与
`stereo_right/` 中的实际文件；后续 stage 应以这些记录为准，而不是猜测目录或文件名。

### 2.2 必须满足的绝对时间约束

每一帧必须满足：

```text
sensor_time_ns == cam0_sensor_time_ns == pose_sensor_time_ns
pose_sensor_time_ns == pose_timestamps_ns[pose_row]
timestamp == (sensor_time_ns - time_origin_ns) / 1e9
```

另外要求 `sensor_time_ns` 和 pose 时间严格递增，帧号、pose 行和源图索引均可追溯。
文件被关键帧选择器重新编号时，绝对时间不能随新文件名重建。

G1 `prepare` 一定生成
`stereo_delta_ms = abs(cam0_sensor_time_ns - cam1_sensor_time_ns) / 1e6`，并执行默认
`10 ms` 配对门。`prepared-stereo` 输入允许省略 `stereo_delta_ms`，存在时才校验该公式；
该 adapter 也不会重新执行 `--max-delta-ms` 门，因此输入提供方仍要负责证明左右同步质量。

### 2.3 G1 输入整理参数

实现：[scripts/prepare_g1_pinhole_stereo_dataset.py](scripts/prepare_g1_pinhole_stereo_dataset.py)

| 参数 | 当前默认值 | 作用/设置原则 |
| --- | ---: | --- |
| `--sequence` | `000000` | G1 标定和 pose 序列号 |
| `--max-delta-ms` | `10.0 ms` | 左右目最大配对时间差；超过则不配对 |
| `--input-projection-model` | `auto` | `g1_20260724` 原始 PNG 不得使用 `auto`：它会被路径/ROI 误判为 `pinhole_rectified`；诊断或重新物化时显式使用 `pinhole_unrectified`。已经应用验收校正的产物改走 `prepared-stereo` |
| `--horizontal-fov-deg` | `100°` | 只适用于真正的原始 Kannala-Brandt 输入；已单目去畸变的 PNG 忽略 |
| `--down-fov-deg` | `28°` | 只适用于真正的原始 Kannala-Brandt 输入；已单目去畸变的 PNG 忽略 |
| `--rectification-roll-deg` | `0°` | 当前实验保持 `0°`；共同立体校正旋转必须来自固定报告，不得再叠加经验 roll |
| `--camera-quaternion-order` | `auto` | 不得在建图任务中直接依赖 `auto`；必须把候选四元数解释与采集保存的旋转矩阵比较，并显式固定误差较小的 `xyzw` 或 `wxyz`。当前 `orientation_xyzw`/`poses.txt` 格式必须传 `xyzw` |
| `--stereo-calibration-report` | 无 | 同一未改变 rig 才能复用。当前参数解析器尚不能直接读取本次 `best_combination.json`，不得把格式不兼容的文件强行传入 |
| `--right-rectification-report` | 无 | LiDAR 右目单应报告不是本次 V1/V2 联合校正的替代品；最终组合包含左右基础单应和右目 x 不变的 y 残差模型 |
| `--recommended-max-depth-m` | `65.535 m`（新运行显式传入） | 取消 5 m 业务截断；该值是 `uint16` 毫米格式上限 |

用于构建地图的 G1 运行必须把相机 pose 表达到 `map` 坐标系，不能沿用入口默认的
`odom` 坐标系。准备阶段必须显式传 `--base-pose-source map`，并按绝对时间分别插值
`map_T_base_link` 和 `base_link_T_head_camera`，最终组合为：

```text
map_T_camera = map_T_base_link @ base_link_T_head_camera
```

不得把 `odom_T_base_link` 当作 `map_T_base_link`，也不得通过经验性旋转相机来补偿坐标系
使用错误。旋转策略必须由数据支撑：当采集已提供可信的 map pose 时，固定地面标定默认只
应用有证据的深度尺度，并使用 `--floor-rotation-policy identity` 保留源相机旋转；只有同一
未改变 rig 的标定报告能够证明所需旋转、且应用后的重力/重投影/时序几何指标共同改善时，
才允许选择 `report`。选择依据和对比指标必须写入本次运行报告。

四元数存储顺序也属于旋转证据，不能由“相机看起来大致朝前”猜测。准备前应分别按
`xyzw` 和 `wxyz` 解释 `aux_poses.jsonl` 的头部相机四元数，并与采集保存的
`poses.txt` 旋转矩阵计算角误差；只有误差接近数值精度的解释可以使用。例如
`g1_20260724` 按 `xyzw` 的角误差约为 `2.9e-11°`，强制重排成 `wxyz` 的误差约为
`20.95°`，因此该格式必须显式传 `--camera-quaternion-order xyzw`。错误顺序会在转头段
造成大范围时序重投影失败，不能用更强深度过滤掩盖。

当前准备脚本从 `state/<sequence>/map_pose.jsonl` 读取标准
`map_T_base_link` 流。若采集器把同样的数据写在 `manifest.jsonl` 的
`poses.values.map` 中、但没有生成独立文件，先使用仓库脚本创建只含符号链接和规范化 pose
流的 staging 数据集，不修改原始采集：

```bash
python scripts/stage_g1_map_pose_dataset.py \
  --src /path/to/g1_capture \
  --output data/g1_map_pose_stage
```

staging 脚本会硬性核验每条记录的方向为 `target_frame=map`、
`source_frame=base_link`，并要求 pose 时间严格递增。后续 `--src` 必须指向该 staging
目录，同时仍显式传 `--base-pose-source map`。

质量要求：原始 manifest 的 layout 必须为 `capture4daaam_like`，其
`quality_report.alignment.ok` 必须为真；存储图像分辨率必须与输入虚拟内参一致、畸变为
零、左右分辨率一致、baseline 为正，并通过时间配对、相机顺序和极线误差检查。
“原样整理”模式的输出 PNG 才要求与输入逐字节一致；当输入经实验证明尚未立体校正时，
输出哈希必然变化，必须改为保存输入哈希、组合标定报告哈希、左右变换、有效区域和输出
哈希，不能把有依据的像素变换误报为重复校正。

### 2.4 `g1_20260724` V1/V2 组合标定经验

复现实验：
[scripts/optimize_g1_v1_v2_stereo_combination.py](scripts/optimize_g1_v1_v2_stereo_combination.py)

验收报告：
[output/g1_20260724_v1_v2_optimal_combination_653_953_final/best_combination.json](output/g1_20260724_v1_v2_optimal_combination_653_953_final/best_combination.json)

本实验固定 `cam0=左目`，使用 653–953 共 301 帧。按帧位置每第 5 帧留出，240 帧只用于
拟合，61 帧只用于最终验证。结论不是把 V1/V2 的矩阵逐元素平均，而是按职责组合：

原始 PNG 不做任何联合校正时，301 帧的帧中位 `abs(dy)` P50 为 `2.222 px`、帧内 P95
的 P50 为 `7.020 px`、正视差比例 P50 只有 `31.2%`，严格通过为 `0/301`。因此“文件名为
`2d_rect`”与“可直接做水平视差”在本数据上明确不等价。

| 信息来源 | 最终职责 |
| --- | --- |
| 当前 PNG 的 CameraInfo | 输入像素的虚拟针孔 K；左右 `D=0`，不得再次 fisheye 去畸变 |
| V1 | 有效 `cam1_from_cam0` 位姿优化初值；最终旋转与 V1 仅相差约 `0.196°` |
| V2 | `60.1930859728 mm` 公制基线、目标 K、P1/P2/Q 和深度尺度 |
| 当前训练帧匹配 | 在 V1 初值附近优化有效相对旋转和平移方向，并拟合右目 x 不变的 y 残差 |
| 留出帧 | 只验收，不参与位姿、内参候选或 y 残差拟合 |

候选选择必须先满足总体正视差和有效区域硬门，再比较严格通过率和垂直误差。只按
`abs(dy)` 排序会选中错误解：例如 0.75 权重的 V2 原始内参候选虽然垂直残差更小，但训练
总体正视差只有 `85.1%`、严格通过率只有 `52.1%`。最终选择不混入 V1/V2 原始内参，而使用
当前 CameraInfo K；训练总体正视差 `98.81%`、严格通过率 `90.42%`。

最终留出结果为：汇总 `abs(dy)` P50/P95=`0.195/0.859 px`，正视差 `99.00%`，
严格通过 `56/61`，共同有效图像面积 `94.84%`。完整 301 帧严格通过 `273/301`。
5 个留出失败帧均因逐帧正视差比例低于 `95%`，其垂直 P95 仍为 `0.70–1.75 px`；因此
深度阶段仍必须屏蔽 `d<=0`、遮挡、左右一致性失败和低置信度像素。

组合候选的最低验收门固定为：

| 门 | 阈值 |
| --- | ---: |
| 留出集“逐帧 `abs(dy)` 中位数”的 P50 | `<=1 px` |
| 留出集“逐帧 `abs(dy)` P95”的 P50 | `<=3 px` |
| 留出匹配总体正视差比例 | `>=95%` |
| 留出严格通过帧比例 | `>=90%` |
| 左右共同有效图像面积 | `>=75%` |

单帧严格通过同时要求 `abs(dy)` 中位数 `<=1 px`、P95 `<=3 px`、正视差 `>=95%`、
有效匹配 `>=90%`。选型必须先过硬门，不能用更低的垂直残差补偿错误视差方向或过度裁剪。

V2 的 R1/R2 经 OpenCV 重生成和合成三维点实验确认是标准的“相机到校正相机”正向旋转，
并非按逆映射方向保存。对当前 PNG 直接或转置套用 R1/R2 都不是最终方案：文件方向正确，
但它描述的源像素/虚拟 K 与当前已单目去畸变 PNG 不一致。不得为适配当前 PNG 而篡改
R1/R2 语义。

最终像素处理顺序必须固定为：

```text
当前左右 PNG（已单目去畸变，D=0）
  -> 左目 source_to_rectified_left_H
  -> 右目 source_to_rectified_right_base_H
  -> 右目 right_y_projective_x_preserving（x 保持不变）
  -> 正视差 d = x_left - x_right
  -> Z_m = 25.11911653856251 / d_px
```

其中 `fx=417.3090004041 px`、`baseline=0.06019308597284 m`、
`fx*baseline=25.11911653856 px·m`。右目残差模型不改变 x，因此仍可使用报告中的 P1/P2/Q
计算公制深度。该实验只验证极线对齐、视差符号和图像覆盖；绝对深度精度仍必须用 LiDAR
或已知距离独立验证。

本数据集完整物化后的 Fast-FoundationStereo + LiDAR 实验进一步确认，V2 标称 baseline
不能直接作为最终公制尺度。以偶数输出帧折的 61 个跨时段样本拟合、奇数输出帧折的 61 个
互不重叠样本留出，得到固定 `depth_scale=0.8426534188`。在留出折 0.25–5 m、已通过左右
一致性的 LiDAR 投影像素上，中位绝对相对误差由 `21.68%` 降到 `7.59%`，中位绝对误差由
`0.526 m` 降到 `0.164 m`，中位有符号误差为 `0.024 m`。因此本数据最终采用
`effective_baseline=0.05072190968 m`，不追加图像坐标旋转：

```text
Z_raw_m = 25.11911653856 / d_px
Z_final_m = 0.8426534188 * Z_raw_m = 21.16670942782 / d_px
```

拟合和留出证据分别保存在
`output/g1_20260724_v1_v2_lidar_scale_train_fold0_61/` 与
`output/g1_20260724_v1_v2_lidar_scale_holdout_fold1_61/`，固定应用报告为
`output/g1_20260724_v1_v2_lidar_depth_scale_calibration.json`。该结果只绑定当前未改变
rig、组合校正和报告哈希；不得外推到换相机或重新安装后的设备。

组合报告由
[scripts/materialize_g1_v1_v2_rectified_dataset.py](scripts/materialize_g1_v1_v2_rectified_dataset.py)
物化为完整 `prepared-stereo`。该脚本只做一次图像插值，保存源图/报告/输出哈希、有效
区域、P1/P2/Q、绝对时间绑定和校正后的 `map_T_camera`；它不会再次做 fisheye 去畸变。
`run_stereo_mapping.py` 仍不直接消费组合 JSON，必须先显式运行物化脚本，再用
`--adapter prepared-stereo`。不得把原始 `g1_20260724/2d_rect` 直接送入深度后端，也
不得继续用 `--input-projection-model pinhole_rectified` 绕过这一门。

## 3. 高质量离线链路的逐步流程

离线一键入口共有 12 个逻辑 stage：
`prepare -> select -> depth -> calibrate -> temporal -> odometry -> loops ->
optimize -> filter -> validate -> fuse -> map`。

这个列表是执行顺序，不是严格的单链数据依赖。`temporal`、`odometry` 和 `loops` 都直接
读取 `03_geometry/`，随后才在 `optimize` 汇合：

```text
03_geometry
  +-> temporal report ------------------+
  +-> local RGB-D odometry -------------+-> global optimize -> filter -> validate
  +-> geometrically verified loops -----+
```

其中 temporal agreement 只在全局优化时用于调整 RGB-D 边的不确定度，局部 odometry
本身不读取 temporal report。

### 如何阅读下面的模块说明

Dashboard 中每个节点代表一个可独立检查的处理阶段，不一定代表一个独立进程。阅读时应
区分五类信息：

- **作用**：该模块解决什么问题，以及缺少它会造成什么后果；
- **原理**：输入怎样变成输出，哪些数据会被修改，哪些只用于诊断；
- **参数**：参数控制的是精度、召回率、运行时间还是硬门，调大和调小的方向是什么；
- **状态**：`等待中/运行中/完成/失败` 是执行状态，不等于质量结论；质量要看报告字段；
- **产物**：下游真正读取的目录或报告。只有日志显示成功但产物不完整时，不能视为完成。

全局运行参数不属于某一个算法模块，但决定 Dashboard 怎样执行整条链：

| 参数 | 含义 | 使用建议 |
| --- | --- | --- |
| `--stop-after` | 执行到指定 stage 后停止 | 单独验证深度时选 `depth`；验证几何时选 `fuse`；完整旧式链选 `map` |
| `--dry-run` | 只打印命令和计划，不运行模型、不生成运行目录 | 第一次配置时开启；确认命令后必须关闭才能真正运行 |
| `--resume` | 已有阶段产物满足完成条件时跳过该阶段 | 只适合参数未变化的断点续跑；改变深度上限后不能复用旧深度 |
| `--overwrite` | 允许阶段脚本覆盖已有输出 | 可能混合新旧证据，优先使用新的运行目录 |
| `--run-dir` | 本次运行的产物根目录 | 每组重要参数使用独立目录，便于比较和追溯 |

Dashboard 的进度来自阶段报告、已提交帧数和受管子进程状态。`dry-run` 在进程层也可能
显示“成功”，但它只表示计划生成成功；应同时检查命令中是否仍有 `--dry-run`、运行目录
是否存在，以及目标 stage 是否产生了对应报告。

### 步骤 1：双目同步、立体校正物化和位姿绑定（`prepare`）

**作用。** 把 G1 双目、相机标定、机器人/头部位姿和绝对时间整理成所有后续模块都能
使用的、已经通过水平极线验收的针孔双目数据集。它解决的是“同一帧到底是哪两张图、
对应哪一个 pose、发生在什么时间、应使用什么内参，以及是否真的能够做水平视差”的基础
问题。这里绑定或校正错误会系统性污染所有深度和地图，后续优化无法可靠补救。

**核心原理。** 对左右相机时间戳做最近邻配对并应用同步门。必须先区分两种输入：

- 已经由独立报告证明水平立体校正的 `prepared-stereo`：不再重投影，保留像素和 K；
- 只完成单目去畸变的针孔 PNG：`D=0`，不再调用 fisheye undistort，但必须按固定报告对
  左右图做共同立体校正并更新 K/P/Q。

`g1_20260724/2d_rect` 属于第二种，不能再按 `pinhole_rectified` 逐字节复制。随后将机身、
头部和左目校正相机外参组合成 `world_T_camera`，并把图像、pose 行、绝对纳秒时间和标定
provenance 写入 `tick_index.json`。真正未做单目去畸变的 Kannala-Brandt 原始像素才允许
走 fisheye 到虚拟针孔分支。

**输入与输出。** G1 模式读取原始双目、标定、pose/time 和采集质量报告，输出
`01_pinhole/rgb/`、`stereo_right/`、`pose/`、`camera_info.json`、`tick_index.json` 与
`pinhole_preparation_report.json`。Prepared Stereo 模式只记录 `adapter_input`，不会复制
或修复输入数据。

**参数影响。** 关键参数见第 2.3 节。调参时还应理解：

- `max_delta_ms` 越小，同步更严格但可能丢掉更多配对；越大则运动场景中的左右错时风险更高；
- `input_projection_model` 对 `g1_20260724` 原始 PNG 必须显式识别为
  `pinhole_unrectified`；`auto` 基于路径和 ROI 会产生错误结论；
- `horizontal_fov_deg` 和 `down_fov_deg` 只作用于真正的原始 Kannala-Brandt 输入，
  当前已单目去畸变图像不得使用；
- `camera_quaternion_order=auto` 依赖数据证据判断顺序，已知采集格式时固定顺序更容易审计；
- 不得用单个 LiDAR 右目单应代替本次左右共同校正；LiDAR 应作为深度输出的独立验证；
- `recommended_max_depth_m` 只写入下游建议值，不会在本阶段生成或裁剪深度。

**状态与质量判断。** 重点检查 `matched_pairs`、`max_matched_delta_ms`、左右跳过数量、
输入/输出左右图 SHA-256、组合标定报告哈希、校正后 K/P/Q、有效区域以及 pose 插值是否
被 clamp。通过条件是数据结构完整、当前投影模型有显式证据、留出极线验收通过，且所有
保留帧的图像、pose 和绝对时间一一对应。仅仅生成 `01_pinhole/` 目录、看到 `2d_rect`
路径或检测到 `D=0` 都不够。

### 步骤 2：内容安全关键帧选择（`select`）

实现：[scripts/select_mapping_keyframes.py](scripts/select_mapping_keyframes.py)

**作用。** 在尽量不损失几何覆盖和语义事件的前提下减少后续帧数。双目深度推理通常是
全链最昂贵阶段之一，关键帧选择直接影响推理时间、显存使用、轨迹约束密度和语义观测次数。
这里的“关键帧”是离线处理采样，不等于 Hydra 内部 place 节点。

**核心原理。** 选择器同时使用 pose 运动、图像匹配和 watchdog。平移或旋转达到阈值时
保留；相机静止但出现新物体、遮挡或局部颜色结构变化时也保留；长时间没有事件时由
`max_gap_s` 强制保留；只有 pose、全局外观和局部内容都足够相似时才标为
`strict_duplicate`。因此它不是固定 FPS 抽帧。

**输入与输出。** 输入为 Prepared Stereo 数据集，输出 `02_selected/`。输出会重新连续
编号，但在 `tick_index.json` 中保留原始帧索引、选择原因和绝对时间；报告
`keyframe_selection_report.json` 记录每个源帧的决定。

| 参数 | 一键入口默认值 | 作用 |
| --- | ---: | --- |
| `--soft-translation-m` | `0.06 m` | 达到此平移即视为普通 pose 运动 |
| `--soft-rotation-deg` | `5°` | 达到此旋转即视为普通 pose 运动 |
| `--hard-translation-m` | `0.15 m` | 强制保留的较大平移 |
| `--hard-rotation-deg` | `12°` | 强制保留的较大旋转 |
| `--max-gap-s` | `1.5 s` | 即使无明显运动/视觉事件，也由 watchdog 保留一帧 |

选择器内部还使用以下默认视觉参数；一键入口当前未把它们暴露为转发参数：

| 内部参数 | 默认值 | 作用 |
| --- | ---: | --- |
| 分析分辨率 | `320 x 240` | 视觉重复/事件分析 |
| Lowe ratio test | `0.75` | ORB 匹配过滤 |
| 最少 ORB matches/inliers | `50 / 40` | 几何匹配可信度 |
| 最低 inlier ratio | `0.75` | 严格重复判断 |
| 最大重复光流 | `3 px` | 小于此值才可能是严格重复 |
| 最大重复平均 Lab 差 | `8` | 全局颜色变化阈值 |
| 局部变化 Lab 差 | `18` | 局部内容事件阈值 |
| 局部组件/总面积比例 | `0.003 / 0.01` | 小区域变化和聚合变化阈值 |
| 显著变化 Lab 差/组件比例 | `35 / 0.0005` | 小而强的视觉事件 |

输出为 `02_selected/` 和 `keyframe_selection_report.json`。每个输入帧都会记录
`pose_motion`、`image_event_at_static_pose`、`watchdog` 或 `strict_duplicate` 等理由。

**参数影响。** 降低 soft/hard 位姿阈值会保留更多帧、增加重叠和计算量；提高阈值会
加速处理，但可能使相邻关键帧视差过大、局部里程计匹配变差。降低 `max_gap_s` 可提高静止
期间的时间覆盖，但也会重复估计几乎相同的画面。视觉阈值越严格，越不容易把帧判为重复，
对小物体变化更敏感，同时也更容易保留光照噪声或自动曝光变化。

**状态与质量判断。** 查看 `source_frame_count`、`selected_frame_count`、
`reduction_ratio` 和 `selection_reasons`。压缩率没有固定的越高越好：应同时确认没有长时间
空洞，运动段具有连续重叠，静止但内容变化的帧被 `image_event_at_static_pose` 保留，并且
输出的绝对时间契约仍成立。

### 步骤 3：可选双目深度后端（默认 Fast-FoundationStereo，`depth`）

默认实现：[scripts/run_fast_foundation_stereo_depth.py](scripts/run_fast_foundation_stereo_depth.py)

可选实现：[scripts/run_foundation_stereo_depth.py](scripts/run_foundation_stereo_depth.py)

**作用。** 为每个保留帧生成与左目 RGB 像素对齐的度量深度，为后续 RGB-D 位姿约束、
时序过滤、点云融合、FastSAM mask 三维定位和 Hydra TSDF 提供几何观测。该模块只估计
单帧双目几何，不负责跨帧轨迹优化。

**后端选择。** 新运行默认选择 Fast-FoundationStereo；原版 FoundationStereo 只在复现
旧运行、兼容既有模型或显式 A/B 时选择。两个实现都输出同一组目录，所以下游几何阶段
不需要按模型分支，但必须根据运行报告保存 provenance，不能只根据 `depth/*.png` 猜来源。

当前 `run_stereo_mapping.py` 尚未提供离线 `--depth-backend` 开关：当 `02_selected/` 没有
完整深度时，它仍会启动原版 FoundationStereo。因而“Fast 默认”目前通过三段式编排实现：
先 `--stop-after select`，再把 Fast 输出写入同一个 `02_selected/`，最后用 `--resume` 继续。
wrapper 恢复时会优先读取 `fast_foundation_stereo_run.json`。第 7.1 节给出完整命令。

**核心原理。** 两个后端都从校正后的左右图预测视差。针孔双目中，同一个空间点在左右图
上的水平位移与距离成反比，因此使用焦距和 baseline 将视差换成米制深度：

深度换算为：

```text
depth_m = fx * baseline_m / disparity_px
```

对 `g1_20260724` 的验收组合，必须使用报告中的校正后参数：

```text
fx = 417.3090004040853 px
baseline = 0.060193085972838754 m
disparity = x_left - x_right
depth_m = 25.11911653856251 / disparity_px
```

这些参数只适用于已经依次应用左右基础单应和右目 x 不变 y 残差校正的输出，不能用于原始
`2d_rect` PNG。深度后端启动前应抽查匹配点，至少确认垂直误差、正视差比例、共同有效
区域和 P2/Q 的 baseline 编码与固定报告一致。

输出是 `uint16` 毫米深度；无效、非有限和负视差像素置零。新运行必须显式传
`--max-depth-m 65.535`，从而取消原有 5.0 m 业务截断。这里的 65.535 m 是 `uint16`
毫米存储格式上限，不是质量承诺，也不是 Hydra TSDF 的 `truncation_distance`。原始
float 深度超过该格式上限时仍无法写入 PNG；若未来确需更远距离，应改为 float32 深度格式，
而不是继续增大本参数。

默认 `left-right` 置信度模式还会交换左右图再推理一次，把左图像素投到右图并比较正反
视差；差异超过 `max(0.75 px, disparity * 0.03)` 的像素视为不一致。这个检查能去掉遮挡
边界、反光、重复纹理和错误匹配，但会降低有效覆盖率并使推理次数约翻倍。

**输出含义。** 所有输出都与左目 RGB 同尺寸：

| 目录 | 数据类型 | 含义 |
| --- | --- | --- |
| `depth/` | `uint16` PNG，毫米 | `0` 表示无效；普通图片查看器可能把几米深度显示成近黑色，不能据此判断质量 |
| `depth_confidence/` | `uint8` PNG | 视差一致性派生的置信度，数值越大越可靠 |
| `depth_consistency/` | `uint8` 二值 PNG | `255` 表示通过左右一致性，`0` 表示未通过或无证据 |
| `depth_occlusion/` | `uint8` 二值 PNG | 标记推断的遮挡区域；大部分为黑色通常表示未判为遮挡 |
| `depth_metadata/` | 每帧 JSON | 有效率、中值深度、左右一致率、遮挡率和平均置信度 |
| `fast_foundation_stereo_run.json` | 默认 Fast 汇总 JSON | 固定模型资产、设置、帧进度、速度、显存、失败数和全序列统计 |
| `foundation_stereo_run.json` | 可选原版汇总 JSON | 原版模型、profile、帧进度、速度、显存、失败数和全序列统计 |

默认 Fast-FoundationStereo 参数：

| 参数 | 核心流程默认值 | 作用 |
| --- | ---: | --- |
| `--repo` | 必填 | Fast-FoundationStereo 仓库；脚本要求 clean checkout 且固定到 commit `a290ba04...` |
| `--checkpoint` | 必填 | Fast 权重；checkpoint、相邻 `cfg.yaml`、文件大小和参数量都会被固定值校验 |
| `--iters` | `8` | Fast 默认迭代数 |
| 输入比例 | `1.0` | 脚本使用完整针孔图像，不提供缩放参数 |
| 精度 | FP16 | CUDA autocast 固定为 FP16 |
| `--max-disp` | `416 px` | 最大表示视差；必须是 32 的正整数倍且不超过 padding 后图像宽度 |
| `--volume-builder` | `triton` | 可选 `triton` 或 `pytorch1`；默认 Triton |
| `--confidence-mode` | `left-right` | 每帧双向推理并生成一致性证据；`validity` 仅适合显式速度实验 |
| 左右绝对/相对容差 | `0.75 px / 0.03` | 左右一致性阈值 |
| `--max-depth-m` | **`65.535 m`（新运行显式传入）** | 取消 5 m 业务截断；只保留 `uint16` 毫米格式边界 |
| `--warmup` | `1` | 正式计时前的暖机次数 |

可选原版 FoundationStereo 保留以下 profile：

| profile | iterations | 输入比例 | 精度 | 典型用途 |
| --- | ---: | ---: | --- | --- |
| `online` | `8` | `0.15` | FP16 | 低延迟预览/worker |
| `refine` | `32` | `1.0` | FP16 | 离线高质量 |
| `custom` | `32` | `1.0` | FP16 | 显式覆盖 |

选择原版时，`run_stereo_mapping.py` 总是把自身的 `--valid-iters` 默认值 `32` 传给深度
脚本。因此仅写 `--depth-profile online` 仍会得到 32 iterations；要使用完整在线设置，应
同时显式设置 `--valid-iters 8 --depth-scale 0.15 --depth-precision fp16`。原版的
`--max-depth-m` 可以独立设置；新运行同样必须显式设为 `65.535`，不能依赖旧的 5.0 m
默认值。

**参数影响。** Fast 默认不使用原版的 `depth_scale/profile/precision` 旋钮；主要速度与
质量参数是 `iters`、`max_disp`、`volume_builder` 和 `confidence_mode`。增加 iterations
通常更慢；`left-right` 相比 `validity` 约增加一次模型调用，却能提供可审计的遮挡和错误
匹配证据，因此默认必须保留。取消 5 m 截断会暴露更多小视差远距离结果，但不保证其准确；
应依赖左右一致性、时序一致性和雷达独立验证判定质量，不得再按距离先验直接清零。

**状态与质量判断。** 进度以 `processed / frames_requested` 和已提交
`depth_metadata/*.json` 数量为准。默认 Fast 完成后必须检查 `status=complete`、`failed=0`、
`settings.maximum_depth_m=65.535`、`artifacts.verified=true`、
`aggregate.mean_valid_ratio`、`mean_left_right_consistency`、每帧低有效率尾部、推理耗时和
峰值显存。有效率低意味着覆盖不足，不必然意味着保留深度错误；还要结合步骤 5 和步骤 10
的跨帧一致性判断。选择原版时读取 `foundation_stereo_run.json` 中的同类字段。

### 步骤 4：固定地面/图像坐标校准（`calibrate`）

实现：[scripts/apply_g1_floor_calibration.py](scripts/apply_g1_floor_calibration.py)

**作用。** 把双目深度尺度、相机光学坐标和机器人世界坐标统一到经过固定实验验证的几何
约定。它主要防止整张地图出现统一比例误差、地面倾斜、墙体歪斜或相机高度不一致。

**核心原理。** 固定报告提供 `depth_scale`、有效 baseline 和图像相机到校准相机的旋转。
本阶段对所有非零深度乘尺度并按几何深度上限重新裁剪，更新 `camera_info.json` 中的
baseline，并把旋转右乘到每一帧 `world_T_camera`。RGB、帧顺序和绝对时间保持不变。

| 参数 | 默认/要求 | 作用 |
| --- | --- | --- |
| `--floor-calibration-report` | G1 继续建图时必填 | 同一固定 rig 的已验证地面/相机坐标校准 |
| `--floor-rotation-policy` | `report` | `identity` 时只应用深度尺度并保持所有相机旋转不变 |
| `--geometry-max-depth-m` | `65.535 m`（新运行显式传入） | 不在校准和后续几何处理中恢复 5 m 截断 |

G1 不允许每次采集独立拟合一个“看起来平”的地面；必须使用已验证的固定报告。
该步骤会把所有有效深度乘固定 `depth_scale` 并按最大深度裁剪，同时把
`camera_info.json` 的 baseline 更新为 `effective_baseline_m`。只有
`floor_rotation_policy=report` 时才对 `world_T_camera` 右乘报告中的相机旋转；
`identity` 明确保留源相机旋转，报告中的建议角度只作为审计信息。
`prepared-stereo` 可以不提供报告，此时直接使用选择后的数据集，不执行 G1 校正。

**参数影响。** `floor_calibration_report` 不是可随意调节的场景参数，必须来自同一未改变
rig 的验证结果；错误报告会造成全局系统误差。`geometry_max_depth_m` 控制校准后以及步骤
5–11共同使用的深度范围。新运行必须保持为 65.535 m；如果任一下游值仍为旧默认 5.0 m，
已经估计出的远处深度会在这里或后续阶段再次被清零。

**状态与质量判断。** 输出为 `03_geometry/` 和
`floor_calibration_application.json`。检查 `depth_scale`、`effective_baseline_m`、
`tf_camera_R_image_camera`、`clipped_pixels`、校准前后有效率分位数，以及
`absolute_timestamps_preserved=true`。Prepared Stereo 显示
`not_required_for_prepared_stereo` 只表示调用者声明输入已经处于正确坐标系，并不是系统
自动验证了地面方向。

### 步骤 5：输入深度时序诊断（`temporal`）

实现：[scripts/diagnose_temporal_depth_consistency.py](scripts/diagnose_temporal_depth_consistency.py)

**作用。** 在修改深度或轨迹之前，测量相邻帧是否对同一表面给出相近深度。它用于区分
“单帧看起来平滑”与“跨帧几何真的一致”，并为全局优化设置 RGB-D 约束的不确定度提供
证据。本阶段是只读诊断，不会过滤 PNG，也不会改变 pose。

**核心原理。** 从参考帧按 `pixel_step` 采样有效深度，利用内参反投影到参考相机三维，
再通过两帧 `world_T_camera` 变换到邻帧相机坐标并投影到邻帧图像。只有投影在画面内且邻帧
也有有效深度的样本才是 `comparable`；预测 Z 与邻帧采样深度的误差小于
`absolute_tolerance + relative_tolerance * depth` 时记为一致。这个过程同时检验深度、
内参、时间绑定和 pose，任何一项错误都会拉低一致率。

一键入口固定使用：

| 参数 | 值 |
| --- | ---: |
| `frame-step` | `1` |
| `neighbor-offsets` | `1` |
| `pixel-step` | `4` |
| `forward-only` | 开启 |
| `require-time-contract` | 开启 |
| 有效深度范围 | `0.25..geometry-max-depth-m`，新运行为 `0.25..65.535 m` |
| 绝对/相对深度容差 | `0.04 m / 0.03` |

该阶段输出 `04_temporal_input/temporal_depth_consistency_report.json`，不直接修改深度，
也没有通过/失败阈值；报告只在后续全局优化中作为 RGB-D 约束权重证据。

**参数影响。** `frame-step=1` 表示不跳过参考帧；`neighbor-offsets=1` 只比较紧邻帧；
`pixel-step=4` 每 4 像素采样一次，是运行时间与空间覆盖的折中。减小 pixel step 会增加
统计密度和耗时。容差越小越严格，能暴露细微误差但会对噪声更敏感；容差越大则可能掩盖
错误 pose 或深度。`forward-only` 避免同一对帧重复统计，`require-time-contract` 把时间
契约作为前置条件。

**状态与质量判断。** 查看 `overall_agreement_rate_weighted`、
`overall_comparable_samples`、offset 1 汇总、误差中位数、`worst_pairs` 和连续窗口统计。
一致率高但 comparable 很少仍不能说明全图可靠；应同时检查有效覆盖。该阶段一键入口没有
配置失败阈值，因此报告较差也可能显示执行完成，真正的硬门在步骤 10。

### 步骤 6：局部 RGB-D 里程计（`odometry`）

实现：[scripts/refine_rgbd_trajectory.py](scripts/refine_rgbd_trajectory.py)

**作用。** 利用相邻关键帧的图像和深度重新估计短距离相对运动，修正机器人原始 pose 的
局部漂移、抖动和轻微尺度误差。这里的“局部”指时间上相邻的关键帧窗口，不是图像中的
局部区域。它不生成新深度，也不直接融合地图，而是给全局优化提供可靠的 RGB-D 边。

**核心原理。** 先按机器人 XY 运动距离和最大帧间隔选局部关键帧；使用 SIFT 特征和
Lowe ratio test 建立图像对应；借助两帧深度把匹配点反投影为三维点，通过 RANSAC
估计 3D–3D 刚体变换并剔除外点；再把可靠变换组成局部 pose graph。优化以 RGB-D 边为
主要几何证据，同时保留相邻机器人 odometry 和位置先验，得到连续、受约束的局部轨迹。
非关键帧通过关键帧修正量插值，绝对时间不改变。

**输入与输出。** 输入为 `03_geometry/` 的 RGB、深度、内参和初始 pose。输出
`05_rgbd_window_graph/` 中的修正 pose 和 `trajectory_refinement.json`；后者记录关键帧、
候选/接受约束、顺序视觉链接、fallback 链接、优化残差和优化前后路径长度。

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| 优化模式 | `pose-graph-3d` | 建立局部 3D RGB-D 约束 |
| `--local-keyframe-distance-m` | `0.10 m` | 局部里程计关键帧距离 |
| `--local-max-keyframe-gap` | `8` 帧 | 最大关键帧间隔 |
| `--local-neighbor-span` | `6` | 每个关键帧搜索的局部邻居跨度 |
| `--local-min-inliers` | `80` | 接受局部视觉/RGB-D 约束的最低内点数 |
| `--local-visual-max-nfev` | `150` | 局部视觉优化最大函数求值次数 |
| feature ratio test | `0.65` | 局部特征匹配过滤 |
| 最大旋转误差 | `3°` | RGB-D/视觉约束接受门 |
| odometry/position prior sigma | `0.25 / 0.75 m` | 相邻里程计和位置先验权重 |
| 最大深度 | `65.535 m` | 新运行不恢复 5 m 截断；远距离点仍需通过几何质量门 |

输出为 `05_rgbd_window_graph/` 和 `trajectory_refinement.json`。可靠局部约束数量还必须
达到 `max(12, floor(keyframe_count / 4))`，否则阶段失败。

**参数影响。** `local_keyframe_distance_m` 越小，关键帧越密、重叠更好但计算量更大；
过大可能使相邻视角变化超出特征匹配范围。`local_max_keyframe_gap` 是运动很小时的帧数
watchdog，过大可能出现长段缺少约束。`local_neighbor_span` 决定每个关键帧向前连接多少个
邻居，增大能提高冗余但近似按跨度增加匹配成本。`local_min_inliers` 越高越保守，能减少
错误边但会降低约束数量。`local_visual_max_nfev` 只限制优化求值次数；提高它不能补救错误
匹配。最大深度决定参与 3D 约束的点，越远的点通常视差不确定性越大。

表中的 sigma 是优化权重的标准差，不是最大允许误差：sigma 越小表示越信任该类约束，
其残差惩罚越强。随意把 sigma 调得很小可能让少数错误 RGB-D 边支配整条轨迹。

**状态与质量判断。** 重点看 `visual_constraints`、`sequential_visual_links`、
`sequential_fallback_links`、优化是否成功、优化前后视觉残差、路径长度变化和 yaw 修正
分位数。约束刚好达到数量门但大量依靠 fallback，或路径长度发生不合理突变，都应视为
风险；“模块完成”只表示求解成功并达到最低约束数。

### 步骤 7：闭环候选与几何验证（`loops`）

实现：[scripts/discover_rgbd_loop_closures.py](scripts/discover_rgbd_loop_closures.py)

**作用。** 找出相机在较长时间后重新看到同一地点的帧对，为全局优化提供跨时间约束，
消除仅靠局部里程计不断积累的长期漂移。闭环错误的破坏性通常大于没有闭环，因此本模块
宁可少接收，也必须做几何验证。

**核心原理。** 采用“检索只提候选、几何决定是否通过”的两阶段流程。首先用 SIFT
视觉词/图像相似度检索相隔较远的关键帧；随后用 RGB-D 稀疏匹配检查内点数、3D 误差和
重投影误差。为了补充纹理较弱场景，还会对限定数量候选构建点云，使用 FPFH + RANSAC
产生初始变换，再以双向 ICP/稠密深度颜色一致性验证。只有满足门限的非局部链接才写入
`verified_links`。

**输入与输出。** 输入仍直接来自 `03_geometry/`，而不是局部里程计修正后的目录，避免
候选发现被一次局部优化锁死。输出 `06_loop_closures/loop_closure_report.json`，以及检索、
稀疏几何和稠密候选的可视化图；真正传给全局优化的只有 verified links。

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| 关键帧距离 | `0.10 m` | 与局部几何保持一致 |
| 最大关键帧 gap | `10` | 闭环检索使用的固定值 |
| `--loop-dense-candidate-count` | `80` | 稠密候选数量 |
| feature ratio test | `0.65` | 稀疏特征匹配过滤 |
| sparse 最低 loop inliers / inlier ratio | `140 / 0.20` | 稀疏候选几何门 |
| sparse 最大 3D/reprojection error | `0.025 m / 2 px` | 稀疏几何误差门 |
| dense forward/reverse fitness | `0.30 / 0.20` | 双向稠密配准门 |
| dense 最大 ICP RMSE | `0.045 m` | 稠密配准误差门 |
| 最大候选平移/相对先验旋转 | `4.5 m / 45°` | 排除不合理闭环 |
| 最大深度 | `65.535 m` | 新运行的几何验证范围 |

硬门：`loop_closure_report.json` 中 `verified_count` 必须至少为 1；否则全局优化和
Hydra 都被阻断。

如果采集设计已经明确证明整段轨迹没有任何空间回访，则闭环不属于该数据集的可验收
指标，不应浪费计算运行候选检索，也不能把 `verified_count=0` 误写成几何失败。此时必须
同时使用 `--skip-loop-closure-validation --preserve-source-trajectory`：前者生成
`status=skipped`、`skip_reason=no_revisit_by_capture_design` 的审计报告，后者保证后续
输出逐元素保留权威 map pose。禁止在可能存在回访、或需要用闭环修正漂移的数据上使用该
开关。

**参数影响。** `loop_dense_candidate_count` 越大，纹理弱场景找到闭环的机会越高，但
FPFH/RANSAC/ICP 成本明显增加。最低 inlier、fitness 门越高越保守；RMSE、3D error、
reprojection error 上限越低越严格。候选平移和旋转先验用于排除物理上不合理的回访，设得
过小会漏掉真实的大回环，设得过大则增加错误候选。`max_depth_m` 增大时稠密点云覆盖更广，
但远距离点噪声也更高。

**状态与质量判断。** 按 `retrieved_count -> geometric_count ->
dense_tested_count -> verified_count` 阅读漏斗，并检查每条 verified link 的内点、fitness、
RMSE、估计变换和帧间隔。`verified_count >= 1` 是当前硬门，但一条闭环并不自动代表地图
正确；应在步骤 8 查看它与重力和其他边是否相容，并在步骤 11 人工检查回访区域。

### 步骤 8：全局位姿图优化（`optimize`）

实现：[scripts/optimize_rgbd_pose_graph.py](scripts/optimize_rgbd_pose_graph.py)

**作用。** 把原始机器人轨迹、局部 RGB-D 里程计、输入时序诊断和已验证闭环放到同一个
优化问题中，求一条全局自洽的相机轨迹。局部里程计解决短程精度，闭环解决长期漂移，
重力先验防止为了满足平面位置而把相机高度或 roll/pitch 扭坏。

**核心原理。** 每个相机 pose 是图节点；相邻机器人 pose 构成 robot prior 边，局部
RGB-D 估计构成 odometry 边，步骤 7 的 verified links 构成 loop 边。每条边的相对变换
误差按对应 translation/rotation sigma 归一化。`gravity-se3` 优化完整六自由度修正，同时
对相机高度变化和 roll/pitch 修正加入软先验；首帧固定，避免整个图的规范自由度。步骤 5
的一致性会影响 RGB-D 边不确定度，时序证据较弱时降低其权重。

闭环在进入求解器前还会按时间半径聚类、质量排序和重力残差筛选，避免同一回访段的很多
相似闭环重复支配优化。当前 wrapper 最多选 8 条，并把通过严格验证的闭环作为 certain
edge。

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| optimizer mode | `gravity-se3` | 保留重力方向的 SE(3) 优化 |
| robot translation/rotation sigma | `0.08 m / 8°` | 原机器人 pose 先验噪声 |
| RGB-D translation/rotation sigma | `0.04 m / 1.5°` | 局部 RGB-D 约束噪声 |
| loop translation/rotation sigma | `0.04 m / 1.5°` | 闭环约束噪声 |
| `--max-loop-gravity-residual-deg` | `8°` | 闭环重力残差门限 |
| `--global-iterations` | `250` | 全局优化迭代次数 |
| loop cluster radius / max loops | `30` 帧 / `8` | 聚类后保留的闭环数量 |
| loop certainty | `certain` | 当前 wrapper 把已验证闭环作为确定约束 |
| gravity height / roll-pitch sigma | `0.04 m / 2°` | 重力一致性软先验 |

输出为 `07_global_pose_graph/` 和 `global_pose_graph_report.json`。

**参数影响。** sigma 越小，优化器越信任相应边；robot sigma 较大表示允许原始里程计被
视觉几何修正。`max_loop_gravity_residual_deg` 越小，越严格排除会破坏重力方向的闭环。
`global_iterations` 是最大优化迭代/求值预算，增加它只对尚未收敛的合理问题有帮助，不能
修复错误边。gravity height/roll-pitch sigma 越小，越强制保持原始高度和水平姿态；过强
可能压制真实坡面或标定误差，过弱可能出现地图起伏。

**状态与质量判断。** 检查 `optimization.success`、初末残差、各类
`edge_residuals_before/after`、`selected_verified_loops`、重力投影结果、优化前后路径长度和
`position_change_from_source_m`。合理结果通常表现为 RGB-D/loop 残差下降，而 robot prior
残差和轨迹修正保持可解释；如果总残差下降但轨迹长度、相机高度或局部形状突变，仍不能
接受。输出目录保存优化后的全帧 pose，同时保留 RGB、深度和绝对时间契约。

### 步骤 9：多邻帧时序深度过滤（`filter`）

实现：[scripts/filter_temporal_depth_consistency.py](scripts/filter_temporal_depth_consistency.py)

**作用。** 使用已经优化好的轨迹删除跨帧无法得到足够支持的深度像素，降低飞点、运动
物体残留、遮挡边缘错误和单帧双目误匹配进入融合的概率。它采用“只删除、不补洞”的
保守策略，不会凭邻帧插值生成新的深度。

**核心原理。** 在缩放后的当前深度图上，对每个有效像素使用优化 pose 投影到前后
`±1/±2/±3` 邻帧。投影在邻帧画面内且邻帧也有有效深度时计为一次 `judged`；预测深度与
邻帧深度满足绝对/相对容差时计为一次 `support`。只有 judged 数达到最低值、但支持比例
低于门限的像素才被清零；证据不足的像素不会仅因“看不到邻居”而被删除。

拒绝掩码以最近邻方式恢复到原分辨率，应用到 `depth`，并同步清零该像素的 confidence
和 consistency。原始双目后端的左右一致性证据、绝对时间和来源哈希继续保留，
所以过滤后的数据仍可审计。

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--temporal-filter-neighbor-offsets` | `1,2,3` | 使用前后相邻 1/2/3 个保留帧 |
| `--temporal-filter-scale` | `0.5` | 过滤分析分辨率比例 |
| `--temporal-filter-min-judged` | `3` | 一个像素至少要有 3 个可判断邻居 |
| `--temporal-filter-min-support` | `0.5` | 至少一半邻居支持当前深度 |
| 有效深度范围 | `0.25..65.535 m` | 新运行的过滤范围 |
| 绝对/相对容差 | `0.05 m / 0.04` | 跨帧深度支持门 |

该步骤使用全局优化 pose 重投影并过滤原始深度；被拒绝像素对应的 confidence 和
consistency 会同步清零，occlusion 基本复制，并在 metadata 中记录 provenance，不是把
双目后端证据原样复制。输出为 `08_temporal_depth_filtered/`。

**参数影响。** 邻帧 offsets 越多，证据更充分但耗时更高，也更依赖轨迹在较长跨度内
准确。`filter_scale` 越大，边界更细致但计算量上升；过小会把细小物体和深度边缘混在同一
低分辨率块中。`min_judged` 越高越保守，因为更多证据不足像素会被保留；
`min_support_ratio` 越高则越严格，在证据充分时会删除更多像素。绝对容差控制近处噪声，
相对容差随距离放宽；两者过紧会侵蚀真实表面，过松会保留飞点。

**状态与质量判断。** 查看 `rejected_pixels`、输入/输出有效率分位数、每帧
`rejected_valid_ratio`、左右证据覆盖率，以及 `poses_preserved`、
`rgb_frames_preserved`、`absolute_time_contract_validated`。删除比例为零不一定最好，过高
也不一定更可靠；应结合步骤 10 的一致率和步骤 11 的点云连续性判断。

`65.535 m` 是原始 `uint16` 深度的存储上限，不等于所有距离都已经通过目标 rig 的融合
可靠性验收。如果完整范围的步骤 11 预览出现远距放射状飞点，必须按多个距离上限生成诊断
预览并保存点数、边界和图像证据。不得把数据集声明的
`recommended_max_depth_m=65.535` 直接当作 Hydra 相机量程：Hydra 会沿有效深度射线分配
TSDF，完整范围必须先用 1–3 帧做峰值 RSS/体素分配预检。2026-07-24 的 G1 数据在第 3 帧
达到约 126.7 GB 匿名内存并被 Linux OOM killer 终止，证明当前 5 cm TSDF 不能无上限融合。

准实时入口必须区分两个范围。`--maximum-depth-m` 描述运行时深度产品范围；本数据保持
`65.535 m`，因此不会清零或覆盖 5 m 外的原始/过滤深度。`--hydra-maximum-range-m`
则是 Hydra 相机模型的有限视锥/分配范围，必须由几何预览、资源预检和全量质量门共同确定。
它会限制 Hydra 本次融合可使用的观测距离，但不能反向改写深度产品，也不能被表述为新的
数据集深度上限。

Hydra 预检按实际 C++ 分配逻辑计算：

```text
block_size = voxel_size * voxels_per_side
max_steps = ceil(hydra_maximum_range / block_size) + 1
cube_candidate_blocks = (2 * max_steps + 1)^3
```

5 cm、每块 16 voxel 时，把 `65.535 m` 误作 Hydra range 会产生
`max_steps=83`、`4,657,463` 个立方候选块；历史 5 m 只有 `4,913` 个，候选规模约增大
949 倍。这解释了第 3 帧约 126.7 GB RSS，而不是普通逐帧内存泄漏。代码现在默认在
`100,000` 个候选块处硬失败，危险配置会在 Hydra 初始化前被拒绝。

2026-07-24 G1 当前通过的资源配置是：深度产品 `65.535 m`、Hydra range `8 m`、
背景 TSDF `12 cm`、每块 16 voxel；对应 `max_steps=6`、`2,197` 个候选块。819 帧几何
预检完成 819/819、0 drop，global P95 `168.17 ms`、峰值 RSS `4,875,180 kB`，质量门
通过。若要把 Hydra range 扩大到 8 m 以上，必须重新做短序列 RSS 预检和全量质量验证；
不能使用“无限值”，也不能静默恢复历史 5 m。距离 A/B 必须写入独立运行目录和
provenance。

### 步骤 10：最终深度质量门（`validate`）

实现：[scripts/diagnose_temporal_depth_consistency.py](scripts/diagnose_temporal_depth_consistency.py)

**作用。** 对过滤后的最终 RGB-D 数据执行可自动判定的硬门，阻止明显不一致的深度进入
正式 Hydra 融合。它复用步骤 5 的重投影诊断算法，但这一次带有明确退出阈值，并同时检查
全序列平均、典型误差和局部最差时间窗口，防止全局平均掩盖某一段失败。

**核心原理。** 对每一对相邻帧统计 comparable 样本、支持样本和绝对深度误差。全局
一致率按 comparable 样本数加权；中值误差先在帧对内汇总再做全序列中位数；然后把时间
轴划为连续、非重叠窗口，计算每个窗口的一致率并取最差值。任一配置的失败条件成立时脚本
返回非零，wrapper 不得继续到正式 map。

| 参数 | 默认值 | 通过条件 |
| --- | ---: | --- |
| 相邻一致率 | `>= 0.85` | 全序列相邻帧 |
| 相邻中位误差 | `<= 0.035 m` | 全序列相邻帧 |
| 分块相邻一致率 | `>= 0.80` | 每个连续、非重叠分块 |
| 分块长度 | `100` 帧 | 最后一块允许不足 100 帧 |
| frame/pixel step | `1 / 4` | 验证采样 |

任一硬门失败都不能进入正式融合。

**参数影响。** `final_min_adjacent_agreement` 越高，对整体一致性要求越严格；
`final_max_adjacent_median_error_m` 越低，对典型几何误差要求越严格；
`final_min_window_adjacent_agreement` 专门约束局部最差段。`window_size_frames` 太大可能掩盖
短时失败，太小则容易被少量低纹理帧支配。frame/pixel step 越小，检查越密但耗时越高。

**状态与质量判断。** 最终结论读取报告中的 `pre_hydra_gate.passed` 和每个 check，不能
只看报告文件存在。还应同时看 `overall_comparable_samples`：如果有效深度极少，一致率
可能很高但地图覆盖仍不足。本门只证明已有深度在时间上自洽，不证明语义正确、点云完整，
也不代替步骤 11 的人工几何检查。

### 步骤 11：直接 RGB-D 融合预览（`fuse`）

实现：[scripts/diagnose_rgbd_fusion.py](scripts/diagnose_rgbd_fusion.py)

**作用。** 在启动复杂的 Hydra/DAAAM 前，用最少的中间机制把最终 RGB-D 和优化轨迹直接
投到世界坐标，生成独立几何预览。它是定位问题的重要隔离点：如果这里已经出现双墙、
分层地面或回访错位，问题来自深度/标定/轨迹，而不是 Hydra TSDF 参数。

**核心原理。** 每隔 `frame_step` 取一帧，每隔 `pixel_step` 取一个有效 RGB-D 像素；用
针孔内参反投影到相机三维，再用 `world_T_camera` 变换到世界系并附上原 RGB 颜色。所有帧
点云合并后按 voxel 网格保留代表点，写出 PLY，并固定随机种子抽样渲染俯视/侧视预览。
这是采样点云拼接和去重，不是 TSDF、ICP 或网格表面重建。

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--fusion-frame-step` | `10` | 每 10 帧做预览融合 |
| `--fusion-pixel-step` | `8` | 点云采样步长 |
| `--fusion-voxel-size-m` | `0.035 m` | 预览体素尺寸 |
| 有效深度范围 | `0.25..65.535 m` | 新运行的预览采样范围 |
| render sample / seed | `180000 / 0` | 预览渲染采样数和确定性随机种子 |

这里的“融合”是采样 RGB-D 点云后做 voxel 去重，不是 TSDF；即使不落在固定步长上，
最后一帧也会强制加入。必须人工检查
`10_direct_rgbd_fusion/direct_rgbd_fusion_preview.png` 中地面、墙、
家具和回访区域是否一致。确认后通过 `--accept-direct-fusion-preview` 放行；这个参数是
验收声明，不会自动判断预览质量。

**参数影响。** `fusion_frame_step` 越小，使用帧越多、覆盖更密但运行和文件成本更高；
`fusion_pixel_step` 越小，每帧点越密。`fusion_voxel_size_m` 越小，保留细节更多且 PLY
更大；越大则预览更轻但小物体和薄结构可能消失。深度范围应与前面阶段一致。最后一帧无论
是否落在固定 frame step 上都会加入，以覆盖轨迹终点。

**状态与质量判断。** 报告中的 `frames_sampled`、`points_after_voxel_downsample`、空间
bounds 和相机路径长度用于排查空结果或尺度爆炸。人工预览应检查地面是否单层、墙面是否
重影、相邻视角颜色几何是否对齐、回访区域是否闭合，以及孤立飞点是否可接受。
`accept_direct_fusion_preview` 只记录人工签字，不会运行图像质量模型，也不应在未检查时
预先勾选。

### 步骤 12：DAAAM 语义前端与 Hydra 建图（`map`）

**作用。** 将最终 RGB-D、优化 pose 和神经网络语义结合为 Hydra/Khronos Dynamic
Scene Graph。几何侧生成 TSDF/mesh、places、rooms 和对象节点；语义侧用 FastSAM 找
实例区域、BotSort/ReID 跨帧关联、代表帧选择和 DAM-3B 开放词汇描述，把自然语言语义
通过 correction 落到 DSG 对象。它是消费前面所有几何证据的最终模块，不负责修复输入
深度或轨迹。

**核心原理。** 每帧 RGB-D 先经过 DAAAM 分割和跟踪，mask 内深度用于三维定位和有效性
门；达到观测条件的轨迹被分配代表帧并送 DAM。并行地，Hydra 按相机 pose 把深度融合进
TSDF、提取 mesh 和空间图层，并使用外部 instance/semantic labels 建立对象。DAM 描述
返回后通过 semantic correction 更新 SceneGraphService/DSG。第 4 节详细解释语义前端，
第 5 节详细解释 Hydra 几何和对象过滤。

**输入与输出。** 输入是 `08_temporal_depth_filtered/`、pipeline YAML、Hydra YAML、
labelspace 和颜色表。期望输出包括 `mesh.ply`、`dsg.json`、`dsg_with_mesh.json`、语义
corrections、日志和运行报告；但本节开头所述旧式 wrapper 输出路由与严格完成判定缺陷
仍然存在，看到 mesh 文件不能推导 DAM 已全部提交。

一键入口最终调用 [scripts/run_pipeline.py](scripts/run_pipeline.py)，并将最终深度按
`--depth-scale 1000` 从毫米转为米。这里的 `depth-scale=1000` 是深度单位换算，和
原版 FoundationStereo 的“输入图像缩放比例”也不是同一个参数；默认 Fast 后端固定使用
全分辨率输入，没有这个缩放参数。

主要输入参数：

| 参数 | 一键入口默认值 | 作用 |
| --- | ---: | --- |
| `--pipeline-config` | 未强制；推荐显式传 `config/pipeline_config.yaml` | DAAAM 分割、跟踪、grounding 参数 |
| `--hydra-config-path` | `config/hydra_g1_high_quality.yaml` | Hydra TSDF、对象、place、room 和 backend 参数 |
| `--labelspace-path` | 建议 `config/labels_pseudo.yaml` | Hydra labelspace；必须与 pipeline YAML 的 `semantic_config_path` 一致 |
| `--labelspace-colors` | 建议 `config/labels_pseudo.csv` | 语义可视化颜色 |
| `--depth-lb/--depth-ub` | `0.25 / 65.535 m`（新运行显式传入） | DAAAM 对象深度有效范围；不恢复 5 m 截断 |
| `--fps` | `10` | 缺少真实时间时的后备帧率；有绝对时间时优先使用真实时间 |
| `--query-interval-frames` | `90` | 离线 assignment 触发周期 |
| `--target-fps` | 无 | 未设置时一键入口加 `--no-throttle`，尽快处理 |

下面分解这个 `map` stage 内部发生的语义与建图步骤。

第 4 节“通用离线值”是推荐命令显式传入 `config/pipeline_config.yaml` 后的有效值，
不是无配置时的 dataclass 回退值。省略 `--pipeline-config` 会回退到不同设置，例如
FastSAM-S、`min_obs_per_track=6` 和 4 个 grounding workers；不能把它视为与推荐配置
等价。另请注意，CLI `--labelspace-path` 只覆盖 Hydra labelspace，不会同步修改
SceneGraphService 使用的 `semantic_config_path`。

**参数影响。** `pipeline_config` 控制 FastSAM、BotSort、assignment 和 DAM；
`hydra_config_path` 控制 TSDF、mesh、object/place/room/backend，两者不能互相替代。
`depth_lb/depth_ub` 是 DAAAM 对象 mask 的有效深度门，不是毫米换算参数；
`depth-scale=1000` 才表示输入 PNG 为毫米。`query_interval_frames` 越小，离线语义分配触发
更频繁但 grounding 开销更高。`fps` 在缺少可靠时间时提供后备时间尺度，不能随意用来
改变真实运动速度。labelspace 与 pipeline 语义配置必须一致，否则对象 ID、颜色和名称
可能错位。

**状态与质量判断。** 几何侧检查处理帧数、mesh 是否非空、DSG 节点/边和对象数；语义侧
检查 qualifying/grounded/pending 数、correction ACK、对象与 mesh 绑定以及最终提交。
旧式离线 wrapper 当前只用 mesh 与 `dsg_with_mesh.json` 做弱完成判定，不能证明 DAM
drain/ACK 或严格语义落盘。需要 artifact-backed 的严格完成证明时，使用第 6 节准实时
Hydra + DAM 路径，并按第 8.2 节检查 postpass、commit、quality report 和 final report。

## 4. FastSAM、跟踪和语义落实的真实阈值

### 4.1 FastSAM 实例分割

实现：[src/daaam/utils/segmentation.py](src/daaam/utils/segmentation.py)，当前通用配置：
[config/pipeline_config.yaml](config/pipeline_config.yaml)

| 参数 | 推荐离线 YAML | 准实时 YAML | 桌面 YAML | 含义 |
| --- | ---: | ---: | ---: | --- |
| model | `FastSAM-x-640x480.engine` | 同左 | 同左 | TensorRT FastSAM-X |
| `imgsz` | `[480, 640]` | 同左 | 同左 | 高、宽；必须与 TensorRT engine 一致 |
| `fastsam_conf` | `0.3` | `0.3` | `0.25` | FastSAM 检测/掩码置信度下限 |
| `fastsam_iou` | `0.5` | `0.5` | `0.60` | Ultralytics NMS IoU 阈值；提高后减少相邻 mask 被 NMS 抑制，也可能增加重复 mask |
| `fastsam_retina_masks` | `true` | `true` | `true` | 输出原图尺度掩码 |
| `min_mask_region_area` | `300 px` | `300 px` | `150 px` | FastSAM 输出后的 mask 像素面积硬门 |
| `polygon_epsilon_factor` | `0.001` | `0.001` | `0.001` | 掩码转多边形时的轮廓近似强度 |

在 `640 x 480` 图像中，300 像素约为整图的 `0.098%`。它不是 15 cm，也无法在
不知道深度和内参时换算成固定物理尺寸。

当前 `SegmentationConfig.max_mask_region_area` 和 assignment 配置中的
`max_mask_region_area` 虽然存在，但实际 FastSAM 过滤路径没有使用它；不要误以为
`307200 px` 正在过滤过大的 mask。

桌面参数位于 [config/pipeline_config_tabletop.yaml](config/pipeline_config_tabletop.yaml)
和 [config/fastsam/fastsam_tabletop_config.yaml](config/fastsam/fastsam_tabletop_config.yaml)。
`150 px / 0.25 / 0.60` 是偏召回的 A/B 起点，不是已在纸杯、橙子数据集上验收完成的
最优值。当前 TensorRT engine 固定为 `640 x 480` 输入；没有重新构建 engine 时，仅修改
YAML 不能获得更高的推理分辨率。

### 4.2 BotSort + CLIP ReID 跨帧跟踪

实现：[src/daaam/tracking/services.py](src/daaam/tracking/services.py)

| 参数 | 通用离线值 | 准实时值 | 作用 |
| --- | ---: | ---: | --- |
| `track_buffer` | `30` 帧 | `30` 帧 | 检测消失后保留 lost track 的最长帧数 |
| `with_reid` | `true` | `true` | 启用外观 ReID |
| `reid_weights` | `clip_general.engine` | 同左 | 通用物体 CLIP ReID 引擎 |
| `reid_half` | `false` | `false` | CLIP engine 使用 FP32 |
| `cmc_method` | `ecc` | `ecc` | 相机运动补偿 |
| ECC 最大迭代 | `100` | `20` | 准实时配置降低 tracking 尾延迟 |
| `batch_reid_crops` | `false` | `true` | 准实时批量准备 crop 并一次传 GPU |
| temporal history | `true` | `true` | 保存轨迹首次/末次观测和观测次数 |

`track_buffer=30` 不是“物体要出现 30 帧才可信”，而是短时丢失后的存活窗口。

### 4.3 2D mask 的深度有效性与 3D 中心

离线 DAAAM 对每个 track mask 执行：

1. mask 内有效深度像素必须至少占 `25%`；
2. 记录最近 `10` 次 mask 中位深度；
3. 最近历史中位数应在 `depth_lb..depth_ub`；现有代码默认上限仍为 `5.0 m`，新运行
   必须显式覆盖为 `0.25..65.535 m`，否则语义对象阶段会重新引入 5 m 截断；
4. 当前代码有一个背景物体例外：mask 面积达到
   `min_mask_region_area * 30 = 9000 px` 时，即使历史中位深度越界也会视为有效；
5. 对有效 mask，使用 mask 中心像素和中位深度反投影为相机坐标，再用
   `world_T_camera` 转到世界坐标。

第 4 条是代码中的硬编码兼容逻辑，并带有待移除 TODO；调参时应把它当作当前行为，
而不是推荐的几何准则。

准实时语义旁路同样要求 mask 内有效深度比例至少 `25%`，但会进一步使用完整 mask
深度做联合反投影，最多保留 `20,000` 个 3D 点，计算稳定实体的 3D 位置和尺寸。
它当前只检查深度有限且 `>0`，不读取 pipeline YAML 中的 `depth_lb/depth_ub`；桌面
YAML 的 `0.20..2.0 m` 只对旧式离线 DAAAM orchestrator 生效，准实时 Khronos 对象的
有效距离门来自 Hydra `object_detector.max_range=2.0 m`。

### 4.4 “观测 8 帧”到底表示什么

当前有三个独立阈值，不能合并理解：

| 位置 | 默认值 | 真正含义 |
| --- | ---: | --- |
| 离线 DAAAM `assignment_config.min_obs_per_track` | `8` | 使用推荐 YAML 时，同一 BotSort `track_id` 在当前历史窗口内至少被观测 8 次，才有资格送给 DAM |
| 准实时 CLI `--semantic-minimum-observations` | `5` | 同一 MapMemory entity 至少累计 5 次由真实 FastSAM 分割产生、mask 深度有效且成功形成 entity 的观测，才送 DAM；传播帧不计数，代码也不强制来自 5 个不同帧 |
| Hydra `active_window.tracker.min_num_observations` | 通用 `8`；桌面 `4` | 这是 ExternalTracker 置信度公式的分母参数，不是独立的 `>=N` 硬门；配合对象分配置信度 `0.5`，通用配置实际第 9 次、桌面配置实际第 5 次观测才首次通过 |

三者的身份域不同：离线按同一 BotSort track 计数；准实时按同一 MapMemory entity 计数，
一个 entity 可能合并多个 BotSort tracks，甚至同帧的多个有效观测；Hydra 则按同一
ExternalTracker 3D identity 计数。三者都不要求观测绝对连续。ExternalTracker 的对象
置信度为 `min(observations / (2 * min_num_observations), 1)`，MeshObjectExtractor 又要求
该值严格大于 `min_object_allocation_confidence=0.5`。所以通用参数 `8` 在第 8 次恰好
为 `0.5`、仍被拒，第 9 次才通过；桌面参数设为 `4` 后，第 5 次为 `0.625`、首次通过。
`8` 因而不是统一的“可信物体概率阈值”，准实时 DAM 资格门默认也是另一个 `5`。

### 4.5 代表帧选择和 DAM batching

离线配置使用 `min_frames_max_size` assignment worker：先尽量用少数帧覆盖所有合格
track，再在“贪心最少帧数 + 1 帧 slack”的约束下，偏好画面中心、mask 较大的观测。

| 参数 | 推荐离线值 | 准实时 YAML 值 | 作用 |
| --- | ---: | ---: | --- |
| `query_interval_frames` | `90` | YAML 中 `10`，但准实时旁路由自身频率/观测门控制 | 离线触发 assignment 周期；代码还在第 60 帧提前触发一次 |
| `min_obs_per_track` | `8` | YAML 为 `5` | 离线 track 进入 assignment 的最低观测数 |
| `N_masks_per_batch` | `64` | `32` | assignment 单组最大 mask 数 |
| `min_frame_margin_slack` | `1` | `1` | 最少帧解上允许增加的代表帧数 |
| position/size weight | `0.5 / 0.5` | 同左 | 代表帧画面位置与 mask 大小评分权重 |
| `multi_image_min_n_masks` | `32` | `16` | DAM 累积到多少 mask 后执行一次多图 batch |
| `max_batch_age_s` | dataclass 默认 `1 s` | `1 s` | 未达到 batch mask 数时的最长等待时间 |

assignment 中的 `min_mask_region_area=300` 用于“代表帧大小评分”的尺度，不再做第二次
硬过滤；真正的硬过滤已在 FastSAM 输出处完成。

桌面 pipeline YAML 把 assignment 的 `min_obs_per_track` 保持为 `5`，把上述评分尺度改为
`150 px`；但这两项只影响旧式离线 assignment worker。准实时 adapter 仍绕过该 worker，
其资格门只由 CLI `--semantic-minimum-observations=5` 控制。

准实时 adapter 绕过上述 assignment worker，所以实时运行不使用 YAML 中的
`query_interval_frames`、`assignment_config.min_obs_per_track`、`N_masks_per_batch`、
slack 和 position/size weight 来调度实体；有效资格门是 CLI
`--semantic-minimum-observations`。DAM worker 自己的
`multi_image_min_n_masks=16` 和 `max_batch_age_s=1` 仍然生效。实时 adapter 还强制
`defer_dsg_processing=false`、`enable_background_objects=false`，覆盖 YAML 中后一项的
`true`。

### 4.6 DAM-3B 生成开放词汇语义

实现：[src/daaam/grounding/workers/dam_grounding.py](src/daaam/grounding/workers/dam_grounding.py)

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `dam_model_path` | `checkpoints/dam/DAM-3B` | 已完整缓存并校验的本地 DAM-3B 快照；运行期禁止依赖网络下载 |
| `dam_conv_mode` | `v1` | DAM 对话模板 |
| `dam_prompt_mode` | `focal_prompt` | 聚焦 mask 区域 |
| 固定区域问题 | `Describe what you see in this region.` | 每个 mask 的描述请求 |
| sampling | temperature `0.2`，top-p `0.9` | 生成策略 |
| `max_new_tokens` | 通常 `512`；整图描述时 `196` | 输出上限 |
| `compute_full_image_description` | 通用/G1 `false`；CODA `true` | 是否额外描述整帧 |
| sentence embedding | 通用配置为空 | DAM 阶段是否立即生成句子 embedding；为空时需查询后处理 |
| select-frame CLIP | `ViT-L-14/openai/openclip` | 代表帧视觉特征；CODA 使用 PE-Core-L14-336 |

配置里的 `grounding.agent_model_name: gpt-5-mini` 不被当前 `dam_multi_image` worker
用于物体描述；当前实际 grounding 模型是 DAM-3B。

### 4.7 语义 ID、correction 和最终 DSG

#### 4.7.1 “10000 类标签空间”的准确含义

这里的准确值是 `10000`，不是 `1000`。更准确的名称是“容量为 10000 个伪语义 ID 的
开放词汇标签空间”，而不是“预训练的 10000 个物体类别”。Hydra 在第一帧之前必须知道
标签图可使用的整数范围，因为传入 Hydra 的 semantic/instance image 每个像素保存的是
整数 ID，不是 DAM 的自然语言句子。

当前 [`config/labels_pseudo.yaml`](config/labels_pseudo.yaml) 声明：

- `total_semantic_labels: 10000`；
- `object_labels` 覆盖 `0..9999`；
- 初始名称为 `label_0` 到 `label_9999`；
- `dynamic_labels`、`invalid_labels` 和 `surface_places_labels` 初始为空。

配套的 [`config/labels_pseudo.csv`](config/labels_pseudo.csv) 为每个 ID 提供稳定的 RGBA
颜色。两份文件形成可在运行时分配的 ID 池，使 FastSAM/BotSort、逐帧标签图、Hydra
对象、mesh、DAM correction 和最终 DSG 能用同一个整数关联：

```text
FastSAM mask / BotSort track
            ↓
分配或复用稳定 semantic_id
            ↓
Hydra 标签图和 DSG 暂时使用 label_<id>
            ↓
DAM-3B 生成开放词汇描述
            ↓
更新该 ID 的名称，并同步 object 与 mesh labelspace
```

例如某个 MapMemory entity 最终使用 `semantic_id=42`，初始化时其占位名称只是
`label_42`；DAM 返回 `red office chair` 后，SceneGraphService 会把 ID 42 对应名称更新
为该描述。整数 ID 保持稳定，改变的是它对应的自然语言和语义元数据。准实时路径以稳定
MapMemory entity 为权威，同一实体可复用已有 ID；一个临时 BotSort track 被合并到已有
entity 时也会改用该 entity 的 canonical semantic ID。

因此日志中的“10000 个标签已加载”只证明标签范围、占位名称和颜色表就绪，Hydra 可以
安全接收第一帧；它不表示系统内置 10000 个真实类别、DAM 已识别 10000 种物体、场景中
存在 10000 个对象，也不证明语义已经绑定到真实 mesh。实际结果要看最终 DSG 对象节点、
DAM correction、mesh 绑定、pending/unmapped/error 和 `rejected_no_mesh`。10000 还是当前
实现的预留容量边界；超过该范围需要同时扩展 YAML、颜色表和语义 ID 管理逻辑。

所以更准确的状态表述是：

> Hydra 与语义前端已成功初始化：FastSAM、BotSort/ReID、DAM-3B，以及容量为 10000 个
> 伪语义 ID 的开放词汇标签空间均已加载；首帧前就绪门槛已满足。

#### 4.7.2 correction、实体绑定和最终 DSG

FastSAM mask 首先得到临时语义 ID。DAM 返回自然语言描述后，以 semantic correction
的形式写入 SceneGraphService/MapMemory，再与 Hydra 对象节点绑定。当前严格对象绑定门为：

| 参数 | 通用默认值 | 桌面 CLI 值 | 通过逻辑 |
| --- | ---: | ---: | --- |
| MapMemory entity merge | `0.50 m` | `0.075 m` | 新 BotSort local track 只与同名且中心距离不超过此值的实体合并；DAM 前名称通常都是 `unknown` |
| DSG 最大中心距离 | `0.75 m` | `0.10 m` | 候选 Hydra object 中心足够近 |
| DSG 最大 AABB gap | `0.15 m` | `0.025 m` | 或两个 3D AABB 间隙足够小 |

后两个 DSG 门是 OR 逻辑，满足任意一个即可通过。桌面值必须在准实时命令中分别通过
`--entity-merge-distance-m`、`--object-binding-maximum-center-distance-m` 和
`--object-binding-maximum-aabb-gap-m` 显式传入；三个参数的代码默认值保持通用值，
以兼容已有运行。MapMemory 的门只影响一个新的 local track 是否与已有实体合并，不能
修复 BotSort 自身把两个物体错误复用为同一个 track ID 的情况。

对象必须绑定真实 Hydra object mesh 才能成为严格 mesh-bound 对象；未绑定实体可以在后续
查询准备时进入 checksum-bound 语义旁路，但不能声称拥有独立 Hydra object mesh。

## 5. Hydra/Khronos 几何融合和对象过滤

当前 G1 基线配置：
[config/hydra_g1_high_quality.yaml](config/hydra_g1_high_quality.yaml)。桌面小物体配置：
[config/hydra_g1_tabletop.yaml](config/hydra_g1_tabletop.yaml)。

2026-07-24 数据的资源安全配置为
[config/hydra_g1_8m_12cm.yaml](config/hydra_g1_8m_12cm.yaml)：Hydra 相机/对象范围
`8 m`、map window `10 m`、背景 TSDF `12 cm`、truncation `36 cm`，对象重建仍独立使用
`2 cm` 分辨率。该配置来自上述候选块预检、3 帧 A/B 和 819 帧纯几何回放，不是经验性猜测。
需要注意，纯几何回放的 `global` service P95 为 `180.58 ms`，不代表 exact-label
对象重建也必然满足同一 `250 ms` 门；`g1_20260724_v1_v2_semantic_map` 中 153 个对象使该
指标升至 `305.46 ms`。后续若要做权威验收，应优化对象重建开销后从零完成单次全量运行，
不能事后放宽阈值或减少语义对象来把已有报告改成 PASS。

### 5.1 TSDF 和活动窗口

| 参数 | 5 cm 基线 | 3 cm A/B 配置 | 含义 |
| --- | ---: | ---: | --- |
| map spatial radius | `8.0 m` | `8.0 m` | 活动地图半径 |
| voxel size | `0.05 m` | `0.03 m` | TSDF 体素尺寸 |
| truncation distance | `0.15 m` | `0.09 m` | 均为 3 个体素 |
| voxels per side | `16` | `16` | 每个体素块边长 |
| mesh `min_weight` | `0.5` | `0.5` | 过滤低权重 padding voxel |
| extra integration distance | `0.5 m` | `0.5 m` | projective integration 扩展距离 |

3 cm 配置提高采样分辨率并显著增大网格量、显存/内存和计算开销，但不能直接推出“小物体
更多”。当前同一数据 A/B 中，5 cm 和 3 cm 都只有 49 个 mesh-bound 描述实体；3 cm
三角形数从 84,175 增至 195,239，最大连通组件面积占比却从 57.54% 降至 22.62%。
因此 5 cm 仍是当前基线，3 cm 只应作为实验候选。

### 5.2 Hydra 对象检测、跟踪和提取

| 参数 | 5 cm 基线 | 3 cm A/B | 桌面 | 当前实现中的真实含义 |
| --- | ---: | ---: | ---: | --- |
| detector `min_cluster_size` | `20` | `20` | `20` | `max_range` 后同一实例 ID 的二维标签像素数，不是 3D voxel/cluster |
| detector `max_range` | `5.0 m` | `5.0 m` | `2.0 m` | range image 上对象标签的距离门，不缩小全局 TSDF 窗口 |
| tracker `temporal_window` | `10 s` | `10 s` | `10 s` | 未观测超过该窗口后 track 才变为 inactive；结束建图时也会清理 |
| tracker `min_num_observations` | `8` | `8` | `4` | 与 allocation `0.5` 联合作用，首次通过次数分别为 `9 / 9 / 5` |
| allocation confidence | `0.5` | `0.5` | `0.5` | 对象级门，源码要求 tracker confidence 严格大于该值 |
| reconstruction confidence | `0.5` | `0.5` | `0.5` | 对象独立 TSDF 的逐 voxel 语义占比门；不是轨迹观测次数门 |
| `min_object_volume` | `0.005 m³` | `0.005 m³` | `0.0001 m³` | 重建前 observation extent 和重建后 mesh AABB 各检查一次 |
| `max_object_volume` | `10.0 m³` | `10.0 m³` | `10.0 m³` | 对象 AABB 体积上限 |
| object reconstruction resolution | `-0.02` | `-0.02` | `-0.02` | 负数表示物体最大边的 `2%`，不是 `2 cm` |
| minimum reconstruction resolution | 未显式设置 | 未显式设置 | `0.003 m` | 比例模式下对象独立重建体素的下限，桌面配置为 3 mm |
| only reconstructed objects | `true` | `true` | `true` | 独立重建得到空 mesh 时不生成对象 |

基线 YAML 中还有 `use_full_connectivity`、`use_3d` 和 `grid_size`，但当前 detector 类型
`InstanceForwarding` 不读取这些字段。`frontend.objects` 下的 `min_cluster_size`、
`cluster_tolerance` 和 `bounding_box_type=OBB` 也因
`frontend.enable_mesh_objects=false` 不参与当前对象 mesh；当前 Khronos
MeshObjectExtractor 的体积检查使用 AABB，不是这项已禁用的 OBB 配置。

`min_dynamic_displacement=1.0 m` 只用于 dynamic track 分支，而当前 ExternalTracker
创建的 track 全部标为静态，因此在这条链上不生效。不要把它降到 `0.10 m` 后声称已经
改善桌面移动物体；若以后切换 dynamic tracker，应单独验证深度漂移、ID 跳变和 ghost
对象后再建立 dynamic-tabletop profile。

桌面 `0.0001 m³` 的等体积立方体边长约为 `4.6 cm`，可保留常见纸杯和橙子的 AABB，
但不保证每个物体都通过：不完整深度、空 mesh、置信度和重建后 AABB 仍可能将其删除。
对象通过 allocation 门后，还要依次具有语义帧、通过重建前体积门、完成独立 TSDF、
通过 voxel reconstruction confidence、生成非空 mesh，并再次通过重建后体积门。

### 5.3 “FastSAM 会过滤小于 15 cm 的物体”是不准确的

当前真实情况是：

- FastSAM 通用配置只按 `300 px`、桌面配置按 `150 px` mask 面积过滤，完全不知道
  物体的厘米尺寸；
- `0.15 m` 是当前已禁用的 frontend object segmenter 的 `cluster_tolerance`，不是
  活跃对象提取路径中的最小边长门；
- Hydra object extractor 另有 `min_object_volume=0.005 m³`。如果把物体近似成立方体，
  等体积边长约为 `17.1 cm`；桌面配置降为 `0.0001 m³`，等体积边长约 `4.6 cm`，但
  细长、薄片和不完整重建物体不能只用这个边长判断；
- 当前活跃的 `min_cluster_size=20` 是距离门后的二维实例标签像素门；`40` 是禁用
  frontend 的 mesh 顶点聚类门。

因此“小物体为什么消失”必须依次检查：FastSAM 像素面积、深度有效率、BotSort 身份、
DAM 观测次数、距离门后的实例标签像素数、Khronos tracker confidence、重建前后 AABB
体积、对象独立 TSDF、非空 mesh 和重建置信度。

### 5.4 Hydra place、room 和 backend 的主要参数

| 模块 | 关键参数 | 当前值 |
| --- | --- | --- |
| GVD | min/max distance | `0.10 / 4.5 m` |
| GVD | minimum difference | `0.05 m` |
| place graph | min node/edge distance | `0.4 / 0.25 m` |
| place graph | node merge distance | `0.7 m` |
| room finder | dilation range | `0.5..1.2 m` |
| room finder | min component/room size | `10 / 10` |
| backend | `optimize_on_lc` | `false`，当前 G1 配置不由 Hydra 自己在闭环时优化 |
| backend | `enable_node_merging` | `false` |
| PGMO | run mode | `FULL` |
| PGMO | translation/rotation node distance | `1.0 m / 1.2 rad` |
| GNC | inlier probability / mu step / iterations | `0.9 / 1.6 / 100` |

离线高质量链已经在进入 Hydra 前完成自己的 RGB-D 全局 pose graph；不要把
`optimize_on_lc=false` 误解为整条链完全没有做全局优化。

## 6. 准实时动态语义链路的逐步流程

### 步骤 R1：回放输入和调度

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--rate-hz` | `1 Hz` | 最大墙钟派发率，不修改原始 `sensor_time_ns`；默认值不代表需要执行 1 Hz 时效测试 |
| `--queue-capacity` | `8` | 每个主 stage 的有界队列容量 |
| `--stage-deadline-ms` | 无 | 可选消息截止时间 |
| `--drain-timeout-s` | `30 s` | 停机前等待主链排空 |
| `--checkpoint-interval-frames` | `30` | 每 30 个完成帧原子持久化一次状态 |
| `--submap-frames` | `30` | 子地图和路径段长度 |
| `--stop-after` | `fusion` | 生成 Hydra 最终地图必须设为 `global` |

主 stage 的服务能力上限固定为 pose `50 Hz`、depth `30 Hz`、dynamic/fusion/global
各 `10 Hz`，再乘 `--stage-rate-multiplier`。这些是 worker 上限，不是地图发布频率；
真正输入上限由 `--rate-hz` 控制。

**执行约束：**正式语义地图任务不得创建 1 Hz、3 Hz 等多个运行目录做频率试跑或 A/B。
选定一个调度值后直接完成全量建图，并以第 8 节的语义 postpass、durable commit 和质量门
验收。仅当用户明确提出吞吐、延迟或部署容量评测时，才单独开展时效测试。

这里名为 `global` 的 stage 只维护 submap/path bookkeeping，并在启用时调用 Hydra；
它不发现闭环，也不执行第 3 节那种全局位姿图优化。实时链使用的是数据集已经提供的 pose。

### 步骤 R2：pose 验证

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--pose-position-std-m` | `0.05 m` | 为回放 pose 附加的位置标准差 |
| `--pose-rotation-std-deg` | `2°` | 为回放 pose 附加的旋转标准差 |
| 最大 pose 时间 gap | `2 s` | 内部输入验证门 |
| 最大 clock jump | `30 s` | 内部时间跳变门 |
| 验证最大 position std | `max(0.1, 5 x 配置值)`，默认 `0.25 m` | 单帧 pose 接受门 |
| 验证最大 rotation std | `max(5°, 5 x 配置值)`，默认 `10°` | 单帧 pose 接受门 |

全局质量报告还要求最大相邻平移 `<=0.50 m`、最大相邻旋转 `<=20°`、最大位置
标准差 `<=0.50 m`。

该 stage 只校验 `poses.txt` 中的 `world_T_camera` 并附加 CLI 指定的固定协方差，不做
VIO、视觉里程计或 pose estimation。

### 步骤 R3：预计算深度或可选 worker

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--depth-backend` | `precomputed` | 默认读取步骤 3 生成的 Fast 深度；`foundation-worker` 是可选原版后端 |
| precomputed provenance | `fast_foundation_stereo_run.json` | 自动优先识别 Fast 报告；原版报告仍兼容 |
| precomputed 最大深度 | `65.535 m` | 新 Fast PNG 不再清零 5.0 m 以外像素；数据集 `recommended_max_depth_m` 必须为 65.535 |
| `--maximum-depth-m` | 数据集值，本数据 `65.535 m` | 运行时深度产品范围；不等于 Hydra 分配范围 |
| `--hydra-maximum-range-m` | `min(maximum-depth, map_window.max_radius_m)`；本数据显式 `8 m` | Hydra 相机视锥/融合范围；启动前执行候选块安全预检 |
| precomputed 行为 | 直接读取磁盘 | 不执行模型；profile/iterations/scale CLI 不会改变已有深度，模型来源以 provenance 为准 |
| 可选 worker `--depth-profile` | `online` | 原版 FoundationStereo worker 默认 8 iterations、0.15 scale、FP16 |
| worker `--depth-confidence-mode` | `left-right` | 配置周期性左右验证 |
| worker `--depth-lr-interval` | `3` | 帧号能被 3 整除时做双向推理，其余只做 validity |
| worker startup/request timeout | `120 s / 2 s` | worker 启动和单请求超时 |
| worker maximum retries | `1` | 单请求失败后的最大重试次数 |

逐帧质量统计包括深度有效率、左右一致性、左右证据覆盖率、时序一致率、GPU 峰值和
worker RSS。时序一致使用 pose 传播上一帧深度，并以
`abs(error) <= 0.05 m + 0.05 * depth` 判断。

在 `precomputed` 模式下，depth stage P95 只代表读取、转换和质检耗时，CUDA/RSS 指标
通常为 0；它不能证明生成这些深度时 Fast-FoundationStereo 满足在线延迟或资源门。模型
性能应读取 `fast_foundation_stereo_run.json` 的 `timing` 和 GPU 记录。选择原版 worker 时，
worker 指标只对 FoundationStereo 有效，不能外推到默认 Fast 后端。

### 步骤 R4：与类别无关的运动检测

实现：[src/daaam/mapping/motion.py](src/daaam/mapping/motion.py)

处理方式：由上一帧深度、相机内参和两帧 pose 预测静态背景光流，再与 Farneback 实际
光流比较；不依赖“person/car”等类别标签。

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `--motion-analysis-width` | `160 px` | 光流分析宽度；高度等比例且至少 48 px |
| residual threshold | `2.5 px` | 实际光流与背景预测残差超过此值为动态候选 |
| forward/backward threshold | `1.5 px` | 光流前后向一致性门 |
| `--minimum-dynamic-pixels` | CLI 默认 `40 px` | 全帧候选总量为 1–39 时整体转 unknown；覆盖 `MotionConfig` 原始默认 24 |
| morphology kernel | `3 x 3` | 动态 mask 闭运算 |

上述像素阈值都作用在缩放到默认宽度 160 的分析图上，不是 640×480 原图。候选总量达到
40 后，小于 40 像素的单个连通域仍保留在 dynamic mask、仍从静态融合中剔除，只是不送入
DynamicLayer。第一帧或运动估计异常时，整帧标记为 unknown 而不是静态，防止不可信几何
污染永久地图。

### 步骤 R5：动态对象层

实现：[src/daaam/mapping/dynamic_layer.py](src/daaam/mapping/dynamic_layer.py)

每个达到组件门的动态连通域使用中位深度反投影，形成 3D 中心、近似尺寸、速度、
协方差、轨迹和生命周期状态。组件的输入 `track_id` 是逐帧临时值
`frame_index * 10000 + component_index`，不是 BotSort ID；跨帧身份仅由 DynamicLayer
使用默认 `1 m` 空间最近邻关联。近似尺寸的第三维固定为 `0.2 m`，不是完整 3D box 测量。

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| association distance | `1.0 m` | 观测与已有动态实体的最大空间关联距离 |
| moving motion score | `0.55` | 达到即判 moving |
| moving speed | `0.20 m/s` | 或速度达到此值即为动态 |
| stable speed | `0.08 m/s` | 低于此值才可能成为稳定候选 |
| stable duration | `2.0 s` | 晋升静态所需稳定时间 |
| stable observations | `5` | 晋升静态所需稳定观测次数 |
| occluded after | `0.5 s` | 未观测达到此时间转为遮挡 |
| remove after | `5.0 s` | 未观测达到此时间从活动层移到历史层 |
| trajectory limit | `512` | 每个动态实体最多保存的轨迹状态数 |

表中的稳定/晋升阈值属于通用 `DynamicLayer` 状态机，但当前回放 adapter 只提交已超过
`2.5 px` 残差门的连通域，并把其 `motion_score` 计算成饱和的 `1.0`。因此当前端到端路径
实际不会进入 `STATIONARY_CANDIDATE/PROMOTED_STATIC`；对象停止运动后只会因不再被
dynamic mask 观测而转为 occluded/expired。主融合也不存在旧 TSDF 的通用反积分。

动态实体及其轨迹只保存在 `realtime_checkpoint.json`，没有写成最终 Hydra DSG 的动态
object nodes；最终图是动态像素隔离后的静态场景图。

### 步骤 R6：静态深度隔离和 Hydra 融合

进入静态 TSDF 前，以下像素深度全部置 0：

- dynamic mask；
- unknown mask；
- 无效/非正深度；
- 深度置信度 `<0.05`。

输出 `static_depth/`、`dynamic_masks/`、`unknown_masks/`，并统计动态泄漏率。Hydra
只融合 `static_depth`。质量门要求动态污染率 `<=1%`、unknown 比例 `<=60%`。

这里的“动态污染率”只检查已经被标成 dynamic 的像素在 mask 应用后是否仍残留深度，
按当前实现通常构造性地为 0；它不衡量运动检测 false negative，不能证明所有运动物体都
已被发现。

### 步骤 R7：准实时 FastSAM/BotSort/DAM 旁路

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--semantic-mode=disabled` | 默认 | 不启动语义前端 |
| `--semantic-mode=frontend` | 非默认 | FastSAM/BotSort + provisional 逐帧标签；配合 Hydra/global 时仍执行 exact-label postpass 并提交 DSG，但没有 DAM 自然语言 correction、严格 mesh-bound ACK 或 DAM runtime gate |
| `--semantic-mode=dam` | 非默认 | 在 frontend 基础上运行 DAM、MapMemory correction、严格 mesh 绑定和 DAM runtime gate |
| `--semantic-config` | `config/pipeline_config_realtime.yaml` | 模型、跟踪和 DAM 配置 |
| `--segmentation-rate-hz` | `5 Hz` | FastSAM 调用能力上限 |
| `--semantic-frontend-rate-hz` | `10 Hz` | 语义旁路 stage 能力上限 |
| `--semantic-queue-capacity` | `2` | 几何到语义旁路的有界队列 |
| `--semantic-minimum-observations` | `5` | DAM 模式资格门，按真实分割观测计数 |
| `--entity-merge-distance-m` | `0.50 m` | 新 local track 与同名 MapMemory entity 的最大合并距离；桌面推荐显式设为 `0.075 m` |
| `--object-binding-maximum-center-distance-m` | `0.75 m` | MapMemory entity 到 Hydra object mesh 的中心距离门；桌面推荐 `0.10 m` |
| `--object-binding-maximum-aabb-gap-m` | `0.15 m` | DSG 绑定 AABB gap 门；桌面推荐 `0.025 m`，与中心门为 OR 关系 |
| semantic startup/drain timeout | `120 / 60 s` | DAM 模式的 worker 就绪和最终排空期限 |
| `--gpu-sharing-mode` | `staggered` | 单 GPU 时深度、前端和 DAM 用 lock/activity 协调 |
| `--dam-minimum-gpu-idle-s` | `1.0 s` | DAM 开始前要求实时 GPU 空闲时长 |

FastSAM 是否到期按原始 `sensor_time_ns` 判断，不按墙钟回放速度判断。在非 FastSAM 帧，
BotSort 仍更新；上一真实分割 mask 最多根据新 bbox 传播 `2` 帧。
传播帧会生成逐帧 Hydra 语义标签，但不会增加 DAM 的真实分割观测次数。

语义旁路在逐帧运行时不阻塞几何链，但启动前会同步 warmup FastSAM/BotSort 并等待 DAM
worker ready；结束时还会阻塞等待 prompt catch-up、DAM FIFO 和 correction drain。
Hydra 几何 live pass 本身不等待逐帧语义。

运行结束后，系统要求每个已提交帧都有与精确帧号、时间和配置 SHA-256 绑定的语义标签；
随后关闭并丢弃 live Hydra，在隔离进程中用持久化标签和 `static_depth` 全序列重放，生成
最终语义 Hydra 图。标签覆盖不完整会硬失败；最终图不是在线增量发布结果，必须等待第二遍
postpass 完成。语义旁路仍可能描述移动物体，但静态 Hydra 中没有对应 mesh 时会记为
`rejected_no_mesh`。

### 步骤 R8：质量门

配置：[config/realtime_quality_gates.yaml](config/realtime_quality_gates.yaml)

| 类别 | 当前硬门 |
| --- | --- |
| 时间/双目输入 | 双目差 `<=10 ms`，必须针孔；若源图只完成单目去畸变，固定组合标定及其留出极线门必须先通过 |
| 深度 | 有效率 `>=15%`，时序一致 `>=70%`，左右一致 `>=60%`，左右证据覆盖 `>=25%` |
| pose | 相邻平移 `<=0.50 m`，旋转 `<=20°`，位置 std `<=0.50 m` |
| dynamic | 静态图动态污染 `<=1%`，unknown `<=60%` |
| runtime | drop ratio `<=10%`、runtime error `0`；service/queue/E2E P95 不超限；深度 CUDA `<=20 GB`、worker RSS `<=16 GB`、restart `<=2` |
| map | significance 面积阈值必须为 `0.005 m²`，significant components `<=1000`，最大组件面积占比 `>=10%`，tiny 组件面积占比 `<=5%` |
| semantic | pending ratio `<=10%`；DAM 模式还要求 worker ready、有真实提交、至少一次 DSG applied、无 pending/unmapped/error |

主要 service P95 上限：pose `30 ms`、tracking `50 ms`、segmentation/depth/fusion/global
各 `250 ms`、dynamic `100 ms`、semantic frontend `5 s`；global 端到端 P95 上限
`1 s`。queue P95 使用同一组上限，但 global queue 为 `1 s`。有新式 significance 指标时，
原始 mesh component count 不参与最终 map 判定；只有旧产物缺少新指标时，才回退到原始
component count `<=1000`。

DAM 另有不可绕过的 runtime gate：首个语义帧前 worker 必须 ready，prompt catch-up
完成，worker FIFO 和 correction 尾部 drain 完成，不得 timeout/强制终止，worker 必须以
0 退出。

semantic PASS 只证明投递链闭合且至少有 correction 成功写入真实 mesh，并不表示所有 DAM
实体都有 Hydra object mesh。`rejected_no_mesh` 被允许且必须单独审阅；如果业务要求每个
识别实体都进入严格 DSG，应额外要求该值为 0，或明确把其作为 checksum-bound
`spatial_only` 查询实体。

## 7. 推荐运行命令

### 7.1 离线 G1：先运行到人工预览

`g1_20260724` 在默认 Fast-FoundationStereo 三段执行前新增一个不可跳过的标定门。先复现
或核验固定组合报告：

```bash
python scripts/optimize_g1_v1_v2_stereo_combination.py \
  --dataset /path/to/g1_20260724 \
  --v1 "/path/to/g1_20260724/calibrations/000000/New Calibration.yaml" \
  --v2 "/path/to/g1_20260724/calibrations/000000/New Calibration_V2.yaml" \
  --start-frame 653 \
  --end-frame 953 \
  --output output/g1_v1_v2_rectification_audit
```

必须检查 `best_combination.json` 中 `acceptance.passed=true`、留出严格通过率、总体正视差、
共同有效面积和 P2/Q baseline。优化报告只生成参数和三帧预览，**不会自行物化完整校正
数据集**。用下列受测应用阶段处理完整采集；若原始采集把 map pose 只写在 manifest，先按
第 2.3 节生成 staging 数据集，并把它作为 `--source`：

```bash
python scripts/materialize_g1_v1_v2_rectified_dataset.py \
  --source /path/to/g1_map_pose_stage \
  --calibration-report output/g1_v1_v2_rectification_audit/best_combination.json \
  --output output/g1_v1_v2_rectified_prepared \
  --maximum-stereo-delta-ms 10 \
  --recommended-maximum-depth-m 65.535
```

该阶段按报告固定顺序处理全部左右 PNG，保存输入/报告/输出哈希，并产出满足第 2 节时间与
pose 契约的 `prepared-stereo`。物化后必须对完整输出执行时间、哈希和双目几何审计，不能
只复用 653–953 帧的选型报告。第一段正式编排只选择该产物：

```bash
python scripts/run_stereo_mapping.py \
  --adapter prepared-stereo \
  --src /path/to/v1_v2_rectified_prepared_stereo \
  --run-dir output/g1_map \
  --recommended-max-depth-m 65.535 \
  --stop-after select
```

不得在原始 `2d_rect` 上继续使用 `--input-projection-model pinhole_rectified`；也不得把
优化报告的预览目录伪装成完整 `prepared-stereo`。

第二段在 `02_selected/` 中生成默认 Fast 深度。`--max-depth-m 65.535` 是不可省略的核心
配置，用格式上限取消旧的 5 m 截断；脚本还会校验 Fast 仓库 commit、checkpoint、
`cfg.yaml` 和哈希：

```bash
python scripts/run_fast_foundation_stereo_depth.py \
  --dataset output/g1_map/02_selected \
  --output output/g1_map/02_selected \
  --repo /path/to/Fast-FoundationStereo \
  --checkpoint "$FAST_FOUNDATION_STEREO_CHECKPOINT" \
  --iters 8 \
  --max-disp 416 \
  --volume-builder triton \
  --confidence-mode left-right \
  --max-depth-m 65.535 \
  --save-raw-products
```

第三段从已验证的 Fast 深度继续到直接融合预览：

```bash
python scripts/run_stereo_mapping.py \
  --adapter prepared-stereo \
  --src /path/to/v1_v2_rectified_prepared_stereo \
  --run-dir output/g1_map \
  --recommended-max-depth-m 65.535 \
  --floor-calibration-report /path/to/validated_floor_geometry_calibration.json \
  --floor-rotation-policy identity \
  --max-depth-m 65.535 \
  --geometry-max-depth-m 65.535 \
  --depth-ub 65.535 \
  --hydra-config-path config/hydra_g1_high_quality.yaml \
  --pipeline-config config/pipeline_config.yaml \
  --labelspace-path config/labels_pseudo.yaml \
  --labelspace-colors config/labels_pseudo.csv \
  --stop-after fuse \
  --resume
```

`--resume` 会因为完整的 `depth/*.png` 跳过原版推理，并优先把
`fast_foundation_stereo_run.json` 记录为深度 provenance。若该文件缺失，即使 PNG 存在也
不应把运行标记为默认 Fast。

如需选择原版 FoundationStereo，可改用单段式旧入口；这是兼容/复现方案，不是默认值：

```bash
python scripts/run_stereo_mapping.py \
  --adapter prepared-stereo \
  --src /path/to/v1_v2_rectified_prepared_stereo \
  --run-dir output/g1_map_foundation \
  --checkpoint "$FOUNDATION_STEREO_CHECKPOINT" \
  --recommended-max-depth-m 65.535 \
  --floor-calibration-report /path/to/validated_floor_geometry_calibration.json \
  --floor-rotation-policy identity \
  --accept-foundation-stereo-noncommercial-license \
  --max-depth-m 65.535 \
  --geometry-max-depth-m 65.535 \
  --depth-ub 65.535 \
  --stop-after fuse \
  --resume
```

检查 `output/g1_map/10_direct_rgbd_fusion/direct_rgbd_fusion_preview.png` 后：

- 若目标是带 durable commit 和质量门的语义图，把
  `output/g1_map/08_temporal_depth_filtered` 作为第 7.2 节实时 adapter 的 `--dataset`；
- 若只需运行旧式离线 DAAAM/Hydra `map`，当前不要用 wrapper 的默认 `map` 完成判定，
  可显式关闭日志路由并直接运行：

```bash
python scripts/run_pipeline.py output/g1_map/08_temporal_depth_filtered \
  --dataset-type ImageSequenceDataset \
  --config config/pipeline_config.yaml \
  --depth-scale 1000 \
  --depth-lb 0.25 \
  --depth-ub 65.535 \
  --fps 10 \
  --query-interval-frames 90 \
  --hydra-config-path config/hydra_g1_high_quality.yaml \
  --labelspace-path config/labels_pseudo.yaml \
  --labelspace-colors config/labels_pseudo.csv \
  --output-dir output/g1_map/11_daaam \
  --zmq-url none \
  --no-logging \
  --no-progress \
  --no-throttle
```

该直接命令规避输出路由缺陷，但仍是第 1.1 节所述旧式 map，不会自动升级成严格实时
语义完成门。

### 7.2 准实时：已准备数据集到最终语义 Hydra DSG

以下命令使用默认 Fast-FoundationStereo 预计算深度。`--dataset` 应指向第 7.1 节产生的
`08_temporal_depth_filtered/`，其中必须保留 `fast_foundation_stereo_run.json`，且报告中的
`settings.maximum_depth_m` 必须为 `65.535`。将 `--depth-backend` 改为
`foundation-worker` 会显式选择可选的原版 FoundationStereo。启动 DAM 前还必须保证
`checkpoints/dam/DAM-3B` 指向完整本地快照，且 OpenCLIP `ViT-L-14/openai` 权重已进入
Hugging Face cache；用 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 做启动预检，禁止把首次
模型下载放进受 semantic startup timeout 约束的建图进程。

下面命令中的 `--rate-hz 1` 只是一个单次构建的调度示例，不要求先做 1 Hz 试跑，也不得
再追加 3 Hz 等对比运行。实际任务应使用已经确认的一个稳定值直接完成正式全量构建。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_realtime_mapping.py \
  --dataset /path/to/prepared-pinhole-dataset \
  --run-dir output/realtime_semantic_map \
  --overwrite \
  --rate-hz 1 \
  --queue-capacity 8 \
  --depth-backend precomputed \
  --maximum-depth-m 65.535 \
  --static-map-backend hydra \
  --hydra-maximum-range-m 8 \
  --hydra-config-path config/hydra_g1_8m_12cm.yaml \
  --hydra-labelspace-path config/labels_pseudo.yaml \
  --hydra-labelspace-colors config/labels_pseudo.csv \
  --semantic-mode dam \
  --semantic-config config/pipeline_config_realtime.yaml \
  --segmentation-rate-hz 5 \
  --semantic-queue-capacity 256 \
  --semantic-minimum-observations 5 \
  --semantic-startup-timeout-s 180 \
  --semantic-drain-timeout-s 600 \
  --stop-after global
```

CLI 默认语义 startup/drain timeout 仍是 `120/60 s`；上面采用 `180/600 s` 是完整
1098 帧成功运行使用的较稳妥设置。正式 exact-label 构建不能沿用语义队列默认容量 `2`：
在一次 3.46 s 的前端服务尖峰中，容量 `2` 的 content-aware 队列驱逐了已经接收的单帧，
导致后处理以 `837/838` 覆盖率硬失败。上面的容量 `256` 用于吸收这种短时尖峰；它不是
吞吐率声明，最终仍必须检查语义队列零 drop、逐帧标签 `100%` 覆盖和 postpass 完整重放。

可选原版 FoundationStereo 在线 worker 的附加参数：

```text
--depth-backend foundation-worker
--foundation-stereo-root third_party/FoundationStereo
--checkpoint "$FOUNDATION_STEREO_CHECKPOINT"
--depth-profile online
--depth-confidence-mode left-right
--depth-lr-interval 3
```

该 worker 不会加载 Fast-FoundationStereo；需要默认 Fast 后端时必须保持
`--depth-backend precomputed`。

### 7.3 桌面小物体语义地图配置

面向纸杯、橙子等桌面目标时，使用以下三个独立 profile；它们是偏召回、适合开始做
A/B 的参数组合，不是已经在目标设备和目标物体集上证明“最优”的最终值：

- [config/fastsam/fastsam_tabletop_config.yaml](config/fastsam/fastsam_tabletop_config.yaml)
- [config/pipeline_config_tabletop.yaml](config/pipeline_config_tabletop.yaml)
- [config/hydra_g1_tabletop.yaml](config/hydra_g1_tabletop.yaml)

桌面任务仍应分成三个层次：

| 层 | 当前用途 | 是否可直接用于抓取 |
| --- | --- | --- |
| 5 cm 全局静态 TSDF/DSG | 房间、桌子、固定障碍物和语义检索上下文 | 否；全局分辨率和时效性不足以直接生成抓取位姿 |
| Khronos 对象独立重建 | 为静止小物体生成独立 object mesh；桌面重建体素为 `max(物体最大边 x 2%, 3 mm)` | 可作目标定位先验，不能代替当前帧碰撞检查 |
| 抓取局部实时层 | 从最新 RGB-D、目标 mask 和机器人状态计算局部点云、抓取位姿和碰撞空间 | 当前仓库尚未实现，需要下游抓取模块补充 |

相对通用 5 cm 配置，桌面 profile 的主要有效变化如下：

| 参数 | 通用值 | 桌面值 | 设置原因 |
| --- | ---: | ---: | --- |
| 全局 TSDF voxel / truncation | `0.05 / 0.15 m` | 不变 | 保留已验证房间级基线；3 cm A/B 未增加 mesh-bound 对象数 |
| FastSAM confidence / NMS IoU | `0.30 / 0.50` | `0.25 / 0.60` | 增加小物体和相邻 mask 召回，代价是更多误检/重复 mask |
| FastSAM 最小 mask | `300 px` | `150 px` | 放宽小目标像素面积门；仍不是厘米尺寸门 |
| realtime DAM 资格门 | `5` | `5` | 同一 MapMemory entity 的真实分割、深度有效观测数 |
| Hydra object max range | `4.0 m` | `2.0 m` | 优先近距离桌面标签形成对象 track，不限制房间级 TSDF |
| Hydra tracker 参数 / 有效首次通过 | `8 / 第9次` | `4 / 第5次` | 保持 allocation `0.5`，按真实置信度公式得到五次有效门 |
| 最小对象 AABB 体积 | `0.005 m³` | `0.0001 m³` | 等体积立方体边长约从 `17.1 cm` 降至 `4.6 cm` |
| 最小对象重建体素 | 未显式设置 | `0.003 m` | 对象独立重建保留小物体细节，同时约束噪声和内存 |
| MapMemory entity merge | `0.50 m` | `0.075 m` | 防止密集桌面上 DAM 命名前均为 `unknown` 的多个 track 被合并 |
| DSG binding center / AABB gap | `0.75 / 0.15 m` | `0.10 / 0.025 m` | 降低语义实体绑定到邻近错误 object mesh 的风险 |

“第 5 次首次通过”只表示对象满足 allocation confidence。Khronos 通常要等 track 超过
`temporal_window` 变为 inactive，或建图结束执行清理后，才把它送入 MeshObjectExtractor；
因此不能把该参数理解为第五帧立刻发布最终 object mesh。

使用预计算深度时，推荐命令为：

```bash
python scripts/run_realtime_mapping.py \
  --dataset /path/to/prepared-pinhole-dataset \
  --run-dir output/realtime_tabletop_map \
  --overwrite \
  --rate-hz 5 \
  --queue-capacity 8 \
  --depth-backend precomputed \
  --motion-analysis-width 320 \
  --minimum-dynamic-pixels 20 \
  --static-map-backend hydra \
  --hydra-config-path config/hydra_g1_tabletop.yaml \
  --hydra-labelspace-path config/labels_pseudo.yaml \
  --hydra-labelspace-colors config/labels_pseudo.csv \
  --semantic-mode dam \
  --semantic-config config/pipeline_config_tabletop.yaml \
  --segmentation-rate-hz 5 \
  --semantic-frontend-rate-hz 10 \
  --semantic-queue-capacity 2 \
  --semantic-minimum-observations 5 \
  --entity-merge-distance-m 0.075 \
  --object-binding-maximum-center-distance-m 0.10 \
  --object-binding-maximum-aabb-gap-m 0.025 \
  --semantic-startup-timeout-s 180 \
  --semantic-drain-timeout-s 600 \
  --stop-after global
```

`--motion-analysis-width 320 --minimum-dynamic-pixels 20` 比通用 `160/40` 更敏感，目的是
降低手、杯子等小运动区域漏入静态 TSDF 的概率，也更容易把光流噪声判为动态，必须单独
检查 false positive。`--rate-hz 5` 只是最大派发目标，不证明整条链已经在目标 GPU 上
达到 5 Hz；它同样不是频率测试指令。正式任务只执行一个已确认调度值，不再追加
1 Hz/3 Hz/5 Hz 对比；应以实际 drop ratio、service P95 和质量门验收该次构建。

如果桌面任务明确选择可选的原版 FoundationStereo 在线 worker，可把深度部分替换为：

```text
--depth-backend foundation-worker
--foundation-stereo-root third_party/FoundationStereo
--checkpoint "$FOUNDATION_STEREO_CHECKPOINT"
--depth-profile custom
--depth-valid-iters 16
--depth-scale 0.5
--depth-precision fp16
--depth-confidence-mode left-right
--depth-lr-interval 1
```

该组合比原版 worker 的 `online = 8 iterations / 0.15 scale` 保留更多杯口和小目标边界，
也明显更重；需要在目标设备对延迟、显存、有效深度率和边界稳定性做 A/B。默认方案仍是
预计算 Fast-FoundationStereo：8 iterations、全分辨率、FP16、左右一致性、不使用 5.0 m
业务截断（PNG 存储上限为 65.535 m）。
上述原版 worker 参数对 `precomputed` 输入没有作用，已有深度必须依据其 provenance 判断。

`pipeline_config_tabletop.yaml` 的 `depth_lb=0.20`、`depth_ub=2.0` 当前只由旧式离线
DAAAM orchestrator 使用。准实时 adapter 仍只检查 `depth>0` 和 mask 内至少 `25%`
深度有效；其当前真正生效的桌面对象距离门是 Hydra detector 的 `max_range=2.0 m`。

如果后续还要运行 `scripts/rebind_dsg_semantics.py`，也应显式传
`--maximum-center-distance-m 0.10 --maximum-aabb-gap-m 0.025`；否则 rebind 会回到通用
`0.75/0.15 m` 默认值，失去桌面 profile 的严格绑定约束。

最终静态 DSG 不能充当抓取执行时的最新世界状态。抓取前必须重新读取最新 RGB-D，核验
目标 mask 深度、短时 3D 中心、支撑面和碰撞空间。可把“图像年龄 `<=200 ms`、目标 mask
深度有效率 `>=80%`、中心重复定位 P95 `<=20 mm`”作为初始业务验收门，但这些指标目前
不是仓库已有的 CLI/YAML 参数，不能写成已实现能力。

## 8. 输出产物和“构建完成”的判定

### 8.1 高质量离线输出

离线 wrapper 虽以 `11_daaam/` 为目标，默认调用实际一定写到
`output/<dataset-name>/out_<t1>/`，并在其中再创建 orchestrator 的 `out_<t2>/`：

| 产物 | 含义 |
| --- | --- |
| outer `hydra_output/backend/mesh.ply` | Hydra 三维网格 |
| outer `hydra_output/backend/dsg.json`、`dsg_with_mesh.json` | Hydra backend 图；文件存在不证明 DAM 描述已全部提交 |
| inner `out_<t2>/dsg.json` | SceneGraphService 在结束时另存的 correction 后副本；当前不会同步另存 correction 后的 `dsg_with_mesh.json` |
| inner `out_<t2>/corrections.yaml` | DAM correction 和时序历史 |
| inner `out_<t2>/background_objects.yaml` | 可选；记录有 correction/位置的对象，并用 `in_hydra`、`filter_reason` 区分，不只包含被过滤对象 |
| outer `processing_stats.json` / `final_stats.json` | runner 性能和数量统计 |
| outer `cv_pipeline_config.yaml`、inner `pipeline_config.yaml` | 两层实际配置快照 |

离线入口只以目标目录下同时存在 `dsg_with_mesh.json` 和 `mesh.ply` 作为 map stage
完成条件；既不核验语义，也因目录错位在默认调用下必然找不到。不要依据该条件宣称
“最终语义地图完成”。

### 8.2 准实时输出

| 产物 | 含义 |
| --- | --- |
| `run_manifest.json` | Git、模型、配置、数据和时间契约 SHA-256 |
| `realtime_run_report.json` | 最终状态、运行配置、DAM runtime gate 和各模块计数 |
| `realtime_metrics.json` / `quality_context.json` | queue/service/E2E、drop 和质量门输入证据 |
| `realtime_checkpoint.json` | 完成帧、动态实体/轨迹、子地图和路径状态 |
| `map_memory.sqlite3` | 稳定实体、语义 correction、人工命名和 ACK |
| `static_depth/`、`dynamic_masks/`、`unknown_masks/` | 静态融合的逐帧审计证据 |
| `semantic_sidecar/label_frames/` | uint16 标签 PNG 及绑定 frame/time/run SHA 的逐帧 JSON |
| `semantic_sidecar/hydra_semantic_postpass_plan.json` | 可审计的全序列语义重放计划 |
| `hydra_semantic_postpass.json` | 逐帧标签重放、帧数和覆盖率报告 |
| `hydra_realtime/backend/mesh.ply` | 静态几何 mesh |
| `hydra_realtime/backend/dsg.json` | 最终语义 DSG |
| `hydra_realtime/backend/dsg_with_mesh.json` | 最终带 mesh 语义 DSG |
| `hydra_realtime/backend/semantic_dsg_commit.json` | 语义图持久化和哈希校验记录 |
| `quality_report.json` | 最终硬质量门结果 |

`label_frames`、postpass 计划/报告和 `semantic_dsg_commit.json` 只在相应的 Hydra +
DAM/exact-label 路径启用时产生。建议把以下条件作为“语义地图构建完成”：

1. 运行报告状态为 `complete`；
2. `semantic_postpass.status == complete`，`frames_replayed == frames_expected` 且
   `label_coverage == 1.0`；
3. `dam_runtime_gate.passed == true`，correction 已 drain，DSG
   pending/unmapped/error 均为 0；`rejected_no_mesh` 另行统计，不能被 PASS 隐藏；
4. `mesh.ply`、`dsg.json`、`dsg_with_mesh.json` 均存在且可以 reload，
   `semantic_dsg_commit.json` 的哈希验证有效；
5. `realtime_run_report.json.quality_passed == true`，`quality_report.json` 的必需
   hard gate 全部通过；
6. 最终配置、模型、数据和语义标签具有可追溯 SHA-256。

这些条件判定的是工程构建和持久化完整性，不等于人工 GT 语义验收。当前
[无人工 GT 阶段](docs/g1_semantic_map_diagnostic_no_gt_stage.md) 即使全部满足，也只能写
`diagnostic_no_gt_complete`；Mask AP、HOTA、entity/binding F1、query recall/FAR 和
held-out 结论仍不可计算或宣称。

普通 runtime 质量门允许 drop ratio `<=10%`；权威验收应额外要求所有请求帧完成、
零 drop、零 runtime error、零 resume，并要求 clean worktree 和验证器
`authoritative=true`。

## 9. 从语义地图到可查询地图（可选后处理）

最终 DSG 中对象需要有自然语言 `description`。如果需要把语义迁移到另一份候选几何，
或保留 `rejected_no_mesh` 实体供 spatial-only 查询，先执行条件性的 rebind：

```bash
python scripts/rebind_dsg_semantics.py \
  --dsg output/<geometry-run>/hydra_realtime/backend/dsg_with_mesh.json \
  --semantic-source-dsg output/<semantic-run>/hydra_realtime/backend/dsg_with_mesh.json \
  --semantic-source-report output/<semantic-run>/realtime_run_report.json \
  --memory output/<semantic-run>/map_memory.sqlite3 \
  --output output/<query-run>/dsg_rebound.json \
  --audit-output output/<query-run>/dsg_rebound.binding.json
```

如果源 DSG 已经是目标几何且不需要 spatial-only sidecar，可以跳过 rebind，并直接把它
作为下一步 `--dsg-file`。否则对 rebind 结果生成中英文统一向量和校验 manifest：

```bash
python scripts/prepare_query_dsg_embeddings.py \
  --dsg-file output/<query-run>/dsg_rebound.json \
  --output-file output/<query-run>/dsg_updated.json \
  --binding-report output/<query-run>/dsg_rebound.binding.json \
  --device cpu
```

默认模型为 `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`，输出 768 维
归一化向量。该步骤写出：

- `dsg_updated.json`：带 mesh 对象 embedding 的 DSG；
- `dsg_updated.manifest.json`：DSG SHA-256、模型名、维数和对象数量；
- `dsg_updated.semantic.json`：存在未绑定实体时生成的 checksum-bound
  `spatial_only/image_only` 语义索引。

图和查询必须使用完全相同的模型；维度相同不代表模型兼容。查询服务默认验证上述
manifest，不应使用 `--allow-unverified-embeddings` 绕过正式验收。

可选地，从满足当前 exact-label/postpass 合约的完整 DAM 运行生成 FastSAM 证据：

```bash
python scripts/prepare_query_evidence.py \
  --dsg-file output/<query-run>/dsg_updated.json \
  --semantic-run output/<semantic-run>
```

证据生成器只选择真实 FastSAM 调用帧，不把 bbox 传播 mask 当作原始分割证据。每个匹配
实体生成 overlay 和 cutout；只有 mask 内有有效深度时才生成 3D 点云。默认任何查询实体
缺少真实证据都会失败，只有显式 `--allow-missing` 才允许部分输出。任意旧版
“completed DAM run”并不一定满足当前 label journal/postpass 合约。

### 9.4 语义位置异常必须修复上游几何契约

当查询描述和机器人轨迹正确、但物体位置系统性偏离时，禁止在查询结果中平移坐标、把物体
吸附到最近 LiDAR 点、只修改某个 entity，或根据期望位置反推一个旋转角。这些做法会隐藏
pose、像素射线或深度源的根因。必须按以下顺序定位并重建：

1. 用采集保存的 `poses.txt` 旋转矩阵核验 `orientation_xyzw`，先排除四元数顺序错误；
2. 明确区分 TF 相机坐标系和存储图像的像素射线坐标系。即使
   `map_T_base_link @ base_link_T_head_camera` 正确，驱动输出的 `2d_rect` 图像仍可能
   缺失一个固定的 `tf_camera_T_image_camera`；
3. 若元数据未给出该固定变换，只能用与目标物体无关的多帧几何证据标定，并设置留出帧。
   当前实现以 map 坐标 LiDAR 的数据驱动地面高度、宽幅底部 FastSAM 地面 mask 和物理双目
   X 轴为约束，在全角度范围搜索固定光学 X 轴旋转；不得读取待验证物体坐标：

```bash
python scripts/calibrate_g1_image_rotation_from_lidar.py \
  --source-dataset /path/to/raw_capture \
  --prepared-dataset output/<geometry-run>/08_temporal_depth_filtered_strict \
  --label-dir output/<semantic-run>/semantic_sidecar/label_frames \
  --lidar-map /path/to/global_cloud_cleaned.pcd \
  --output output/<audit>/image_lidar_rotation_calibration.json
```

标定报告必须同时证明：原始四元数解释与矩阵一致、pose 为 `map` 坐标、目标物体没有参与
标定、训练帧和留出帧的 LiDAR/地面 mask 指标均改善。地面不能观测任意 yaw；当前脚本用
采集声明的双目 X 基线固定图像横轴，只估计绕光学 X 的缺失变换，不把错误的 `wxyz`
解释伪装成“相机调平”。

如果双目深度没有通过相机/LiDAR 独立验证，不得继续用该深度生成物体三维坐标。可保留
RGB/FastSAM 作为语义来源，改由 map 坐标 LiDAR 提供几何，并从所有对象的多帧 mask
统一重建，而不是修补单个查询结果：

```bash
python scripts/build_lidar_semantic_query_geometry.py \
  --source-query-map output/<query-run-v002> \
  --source-dataset /path/to/raw_capture \
  --prepared-dataset output/<geometry-run>/08_temporal_depth_filtered_strict \
  --label-dir output/<semantic-run>/semantic_sidecar/label_frames \
  --lidar-map /path/to/global_cloud_cleaned.pcd \
  --calibration-report output/<audit>/image_lidar_rotation_calibration.json \
  --output-query-map /path/to/query-map-v003
```

该重建使用 `map_T_base_link @ base_link_T_head_camera_xyzw @
tf_camera_T_image_camera` 投影全局 LiDAR，默认不设最大语义深度；每个物体由多帧 mask
共同支持、逐像素 Z-buffer 和最近的显著连贯表面确定。成功几何必须声明
`fastsam_masked_lidar_map_projection` 并绑定 LiDAR 与标定报告 SHA-256。没有足够 LiDAR
证据的对象必须降为 `image_only`，不能沿用失败 RGB-D 产生的旧坐标。原版 query map
保持不可变，重建结果写入新版本目录，最后用相同模型、相同查询和相同 LiDAR 底图生成
前后截图验收。

查询命令：

```bash
python scripts/query_sentence_dsg.py \
  --dsg output/<query-run>/dsg_updated.json \
  --query "白色天花板灯" \
  --top-k 5 \
  --min-similarity 0.55 \
  --min-margin 0
```

默认同时检索 `mesh_bound` 和 `spatial_only`，加 `--require-mesh` 才只保留真实 Hydra
object mesh。当前接口检索的是 object，不应把结果当成可靠的 room、邻接关系或导航拓扑
查询。

rebind、embedding/manifest 和 evidence 都不是 `run_stereo_mapping.py` 或
`run_realtime_mapping.py` 的内置 stage，因此“从原始双目到可查询并带证据的 DSG”目前
仍不是单一脚本。

## 10. 调参时的检查顺序

当某类物体没有出现在最终语义地图中，建议按以下顺序检查，而不是直接修改“15 cm”：

1. **输入**：左右目是否同步、单目去畸变与联合立体校正是否被正确区分、留出极线门是否
   通过、K/P/Q/baseline/pose/time 是否绑定；不得仅凭 `2d_rect`、`D=0` 或水平线预览判定；
2. **深度**：mask 内是否至少 25% 有效，是否误用了旧的 5 m 截断，左右和时序是否一致；
3. **FastSAM**：通用配置是否达到 confidence `0.3`、mask `300 px`；桌面配置是否达到
   `0.25`、`150 px`，并检查提高到 `IoU=0.60` 后是否出现重复 mask；
4. **BotSort**：ID 是否稳定，30 帧 buffer 和 ReID 是否成功跨过遮挡；
5. **DAM 资格**：推荐离线配置是否达到 8 次 track 观测；准实时是否达到 5 次真实
   FastSAM、深度有效并成功进入 MapMemory 的 entity 观测；
6. **Hydra 对象 track**：`max_range` 后是否至少有 20 个同实例二维标签像素；通用 tracker
   参数 `8` 是否达到实际第 9 次观测，或桌面参数 `4` 是否达到实际第 5 次观测；
7. **对象几何**：是否依次通过重建前 AABB 体积、对象独立 TSDF/voxel confidence、非空
   mesh 和重建后 AABB 体积；通用范围为 `0.005–10 m³`，桌面下限为 `0.0001 m³`；
8. **实体合并与语义绑定**：桌面运行是否显式使用 entity merge `0.075 m`、中心距离
   `0.10 m`、AABB gap `0.025 m`；是否确有 object mesh，是否被记为
   `rejected_no_mesh/spatial_only`；
9. **最终提交**：exact-label postpass、DAM runtime gate、MapMemory ACK、DSG durable
   commit 和质量门是否通过。

任何物理尺寸阈值都应结合目标距离、相机内参、mask 像素、有效深度、TSDF 体素和
对象完整重建程度共同分析，不能只用一个厘米数代表整条过滤链。

## 11. D0 故障注入的证据与判定规则

D0 不是“把某个数改坏后看图”，而是先验证质量采集器是否有资格参与后续选择。每类故障
必须在查看结果前冻结：注入剂量、eligible 样本、首先应报警的模块、主效应方向、control
误报警门和中/重剂量检出率。注入后不得换用更好看的次级指标覆盖主采集器失败。

G1 473–573 的完整实例在
`experiments/g1_20260724_473_573_v1_1/comparisons/diagnostic_gt_free_d0_all_families_20260728/`：
13 类均已执行，主采集器严格通过 11 类；JPEG Q60 和 camera–LiDAR 时间偏移未通过。
这说明“覆盖完整”和“D0 PASS”必须分开记录。

### 11.1 故障必须路由到协议指定的首个采集器

- 深度尺度优先看 E4 camera–LiDAR signed error，不能只看 E5 时序 agreement。实际负尺度
  会改善 E5 的内部 agreement，若把 E5 当主指标会得到错误方向；E4 在冻结 LiDAR 样本上
  能检出全部 ±5/10/20%。
- 位姿平移/yaw 才优先看 E5 重投影误差和 agreement drop。
- camera–LiDAR 时间偏移的严格主响应是投影有效率下降和边界误差上升。同源 LiDAR 点的
  投影位移可以作为次级诊断，但不能用它的 100% 检出覆盖主联合告警最低 14.5% 的失败。
- 模糊/JPEG 的 SIFT 内点保留率只代表 E3 特征可见性，不能冒充 Mask AP、ReID 或 HOTA。

### 11.2 lineage 与样本分母必须冻结

- 所有窗口选择使用 raw `source_idx`/`cam0_source_idx`；中间
  `source_frame_idx` 只表示上游数组位置，禁止当 raw tick。任何混用都要使运行失效并保留
  `POSTHOC_INVALIDATION.json`。
- 比较深度尺度时，像素 validity 必须从 control 冻结；若在注入后重新应用深度阈值，
  会改变样本集合并污染效应量。
- eligible 必须描述“该注入在此样本上能发生”。例如 control dynamic mask 为 0 像素的
  帧无法腐蚀或膨胀，不应作为漏检；G1 追加审计把每个 mask 剂量从错误的 67/76 修正为
  67/67，同时保留原摘要和修正依据。
- 对连续丢帧，应枚举所有可放置窗口并保存每个被删 raw tick，而不是只挑一个容易报警的
  位置。

### 11.3 注入产物与失败现场都属于正式证据

每个 D0 cell 至少保存：

- 注入前预注册与展开后的 CLI；
- 修改后的图像、JPEG bitstream、mask、depth/pose、timestamp、track/entity/query
  observation；
- feature match、LiDAR correspondence 或重投影 pair 的原始 NPZ/JSONL；
- 逐帧/逐对象表、control-paired delta、剂量汇总和可视化；
- stdout/stderr、`/usr/bin/time -v`、终止异常和事后失效说明；
- 对 cell/run 全部文件的 SHA-256 inventory 和根摘要。

输出目录必须 append-only；聚合失败、路径假设错误、lineage 错误或样本集漂移时，不清理
已经完成的 cell。修复后从新目录重跑，并在最终 failure ledger 同时链接失败运行和修复
运行。报告中的 montage 只是入口，不能替代修改后的 PNG/JPEG/NPZ/JSONL。

### 11.4 无人工 GT 时的结论上限

没有 reviewed human GT 时，D0 可以验证同步、极线、时序、LiDAR、mask 面积、ID
一致性、binding gate 和 top-1 margin 是否对已知注入响应；不能给出 Mask AP、
HOTA/IDF1、ReID、错误 mesh 率或 query Recall/MRR/FAR。即使某个 proxy collector
达到 100%，结论仍应写成 `diagnostic/proxy observability passed`，不能写成语义准确率
通过或正式 winner。
