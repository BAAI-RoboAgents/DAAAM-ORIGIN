# 实时动态语义地图开发与验收清单

## 1. 文档用途

本文档是 DAAAM 实时动态语义地图改造的需求基线、任务分解和验收索引。每个开发任务必须同时满足：

1. 代码已接入可执行流程，而不是只有接口或示例。
2. 自动化测试覆盖正常、边界和故障路径。
3. 运行报告能给出量化结果，失败时阻止错误数据进入下一阶段。
4. 不破坏现有 G1 离线一键建图和绝对时间契约。

状态定义：

- `[ ]`：未实现。
- `[~]`：代码已实现，但真实传感器、GPU 或长时数据验收尚未通过。
- `[x]`：代码、自动化测试和规定的本机验收均通过。

## 2. 当前基线与问题

G1 `20260713_170500` 已完成鱼眼转针孔、绝对时间 pose 对齐、内容安全关键帧筛选、FoundationStereo、RGB-D 约束、几何闭环、位姿图优化、时序深度过滤和 Hydra 建图。最终处理 789 帧。

| 模块 | 实测耗时 | 等效频率 |
|---|---:|---:|
| FoundationStereo，32 iterations | 1.240 s/帧 | 0.81 Hz |
| DAAAM CV | 0.228 s/帧 | 4.38 Hz |
| Hydra | 0.104 s/帧 | 9.62 Hz |
| DAAAM CV + Hydra 串行 | 0.332 s/帧 | 3.01 Hz |

历史离线基线的 Mesh 有 81,607 个顶点、106,047 个三角面；按原始 PLY
索引统计为 4,013 个连通分量，最大分量仅占 1.84%。后续审计确认 Hydra
会在体素块边界输出坐标相同但索引不同的顶点，因此质量门禁同时报告原始
索引和按 0.1 mm 焊接后的几何连通性，并以后者作为地图门禁。历史运行还
存在 `corrections_applied=0` 和退出时 1,149 条语义任务待处理的问题。

截至 2026-07-16，新的 G1 全量在线验收处理 789/789 帧，各阶段零丢帧、
零错误，所有硬质量门禁通过。运行目录为
`output/realtime_g1_full_foundation_hydra_5hz_s015_w160`。

| 验收项 | G1 全量结果 | 门限 |
|---|---:|---:|
| pose service P95 | 7.12 ms | `<30 ms` |
| tracking service P95 | 37.84 ms | `<50 ms` |
| FoundationStereo depth service P95 | 246.40 ms | `<250 ms` |
| Hydra service P95 | 68.91 ms | `<250 ms` |
| 深度有效率 / 时序一致性 | 79.55% / 94.32% | `>15% / >70%` |
| 左右一致性 / 覆盖率 | 83.24% / 33.33% | `>60% / >25%` |
| 动态污染率 | 0.00% | `<1%` |
| Mesh 焊接后分量 / 最大分量占比 | 461 / 58.82% | `<1000 / >10%` |
| FoundationStereo 峰值 CUDA / worker RSS | 0.79 GB / 4.05 GB | `<20 GB / <16 GB` |
| FoundationStereo 重启 / Hydra 拒绝 | 0 / 0 | `<=2 / 0` |

## 3. 目标和边界

### 3.1 功能目标

- 所有观测和异步结果按绝对 `sensor_time_ns` 对齐，并携带 `map_revision`。
- 前端、深度、语义和全局优化按不同频率运行，任何慢模块不得形成无界积压。
- 动态像素不进入永久静态地图；动态对象具有可查询的 3D 状态、轨迹和生命周期。
- 闭环可更新子地图与对象坐标，旧版本异步结果不能覆盖新地图。
- 用户命名、别名和锁定属性持久保存，自动语义更新不能覆盖人工编辑。
- 不同时间、不同起点的会话只有在配准通过质量门禁后才允许合并。
- 每一处理阶段均有机器可读质检结果；硬失败时停止下游处理。

### 3.2 实时目标

| 路径 | 目标频率/延迟 | 过载行为 |
|---|---:|---|
| pose/TF 输入 | 20-50 Hz，P95 `<30 ms` | 不丢最新状态，合并旧状态 |
| 地图关键帧调度 | 持续 1 Hz | 保留视觉事件，跳过严格重复帧 |
| 2D/3D 跟踪 | 每个地图关键帧，P95 `<50 ms` | 高频预测是非阻断增强项 |
| 分割 | 地图关键帧或事件触发，P95 `<250 ms` | 中间帧传播 mask |
| 深度与局部融合 | 持续 1 Hz，P95 `<250 ms` | 不确定时请求关键帧，不伪造深度 |
| 永久地图发布 | 持续 1 Hz，几何链端到端 P95 `<=1 s` | 动态/unknown 不进入永久层 |
| VLM/DAM 修正 | 异步，P95 `<5 s` | 同一实体请求合并、可重试 |

1 Hz 是地图链路目标，不替代外部 VIO 的 20 Hz 以上输入门限。文件回放保留
原始 `sensor_time_ns`，因此输入本身的空窗不算地图漏帧；权威报告必须证明有
关键帧时 pose→global 的端到端 P95 不超过 1 秒、零 drop/error。5/10/15 Hz
只作为非阻断压力测试。单元测试只验收调度逻辑，不等价于模型性能达标。

### 3.3 非目标与外部依赖

- 本仓库提供 pose backend 接口、增量位姿图、闭环门禁和 submap 修正；真实机器人部署仍需接入成熟的 stereo-inertial VIO、轮速或激光惯性前端，不在本仓库从零训练 VIO。
- FoundationStereo 只作为研究/非商用深度后端，其许可证和权重不得被本项目重新授权或提交。
- TensorRT engine、FoundationStereo 权重和 30 分钟真实设备压力测试结果属于部署产物，不进入源码仓库。

## 4. 系统契约

所有跨阶段消息使用下列主键：

```text
(sensor_time_ns, entity_id/track_id, map_revision, calibration_revision)
```

- `sensor_time_ns` 是采集时钟的绝对纳秒值，禁止用筛后序号或固定 FPS 重建时间。
- `map_revision` 在闭环、跨会话配准或人工地图变换后单调递增。
- 旧 revision 的结果只允许被显式重投影；否则记为 `stale_revision` 并拒绝。
- 同一实体、同一 revision、同一操作具有幂等键，重试不能产生重复状态。
- pose 包含 6x6 协方差；缺失或非正定时必须标记为不确定，不得伪造高置信度。

队列价值优先级从高到低为：

```text
loop_candidate > image_event_at_static_pose > pose_motion > watchdog > routine > strict_duplicate
```

第一帧、最后一帧和视觉事件不得因普通过载策略被删除。

## 5. 里程碑和可验收任务

### M0：基线与开发门禁

- [x] **M0-01 轻量导入与基线测试**
  - 目标：数据契约和实时核心模块不依赖 ROS、Hydra、CUDA 或 dotenv 才能导入。
  - 验收：`pytest -q` 全部通过；缺少可选包时只能跳过对应集成测试，不能使数据模型测试失败。
- [x] **M0-02 可复现配置与运行清单**
  - 目标：记录 Git SHA、submodule SHA、模型配置、校准版本、队列配置、硬件和 Python 环境。
  - 验收：实时 dry-run 生成 `run_manifest.json`，必填字段完整且可 JSON Schema 式校验。

### M1：绝对时间、多频率与有界调度

- [x] **M1-01 版本化消息契约**
  - 目标：实现 Frame、Pose、Depth、Mask、SemanticCorrection 和 MapUpdate 的统一 envelope。
  - 验收：拒绝非正绝对时间、负 revision、错误 covariance 和不匹配的 payload 时间。
- [x] **M1-02 内容价值有界队列**
  - 目标：队列满时按价值、deadline 和时间新鲜度淘汰；严格重复帧最先淘汰。
  - 验收：容量始终不超限；视觉事件不会被 routine/duplicate 挤掉；每次淘汰有明确 reason。
- [x] **M1-03 多频率调度与回压**
  - 目标：pose、tracking、segmentation、depth、fusion、semantic、global backend 独立节拍运行。
  - 验收：慢 semantic handler 不阻塞 pose/fusion；旧 revision 回写被拒；stop 后所有线程在 5 秒内退出。
- [x] **M1-04 实时指标**
  - 目标：记录各阶段 queue wait、service、端到端 P50/P95/P99、队列年龄、吞吐和丢帧原因。
  - 验收：确定性回放报告数值可复算；空样本不产生 NaN JSON。

### M2：FoundationStereo 在线深度

- [x] **M2-01 在线/精修双配置**
  - 目标：在线支持缩放、8/16 iterations、FP16/BF16、可选 `torch.compile`；精修保留全尺寸 32 iterations。
  - 验收：dry-run 清单准确显示实际 profile；非法精度、缩放或迭代数在加载模型前失败。
- [x] **M2-02 深度置信度与左右一致性**
  - 目标：输出 metric depth、有效掩码、置信度、遮挡/左右不一致掩码和逐帧统计。
  - 验收：合成 disparity 的一致区域置信度高，遮挡和矛盾区域被拒；所有产物尺寸与 RGB 一致。
- [x] **M2-03 非关键帧传播与降级**
  - 目标：按 pose+深度重投影传播，超过时间/视角/有效率阈值则请求新关键帧。
  - 验收：恒深度平移合成测试误差在配置阈值内；不确定时返回 `needs_keyframe` 而非伪造深度。
- [x] **M2-04 独立进程资源边界**
  - 目标：FoundationStereo 可在独立 Conda 环境运行，超时、崩溃和 OOM 可重试或降级，不拖死前端。
  - 验收：故障注入后前端继续处理 pose；重启次数和降级原因写入报告。
- [x] **M2-05 硬件性能验收**
  - 目标：目标设备在线深度/传播持续达到至少 1 Hz，service P95 `<250 ms`，显存不随帧数增长。
  - 验收：G1 回放报告达到门限；否则状态保持 `[~]` 并记录实际瓶颈。

### M3：静态地图与动态对象分层

- [x] **M3-01 类别无关运动判定**
  - 目标：结合 pose 预测背景流、实际光流和深度残差生成 dynamic/unknown/static mask，不只依赖类别。
  - 验收：相机运动的静态背景不过度误判；独立移动区域被检出；匹配失败区域为 unknown 并保守隔离。
- [x] **M3-02 动态对象 3D 状态**
  - 目标：维护 `track_id、entity_id、语义概率、位置、速度、尺寸、6x6协方差、轨迹、last_seen_ns`。
  - 验收：非均匀时间更新速度正确；短时遮挡可预测并重关联；协方差随缺失观测增大。
- [x] **M3-03 静态融合隔离**
  - 目标：dynamic 和 unknown 像素的深度在送入静态 TSDF 前清零，并输出可审计 mask。
  - 验收：移动物体区域静态深度有效像素数为 0；原输入数组不被原地破坏。
- [x] **M3-04 生命周期、移除与静态晋升**
  - 目标：动态对象超时从实时层移除但保留轨迹；连续稳定达到时间与观测阈值后才可晋升临时静态层。
  - 验收：移动、停下、再次移动、离场四阶段状态转换完全符合配置。
- [x] **M3-05 动态污染质量指标**
  - 目标：统计动态 mask 泄漏率、unknown 占比、被隔离深度比例和静态晋升数量。
  - 验收：质量门禁可在污染率超阈值时阻止永久融合。

### M4：在线 pose、闭环、子地图和路径

- [x] **M4-01 Pose backend 契约**
  - 目标：支持外部 VIO/odom/IMU backend，输出绝对时间 pose、covariance 和状态；odom 只作为 prior。
  - 验收：乱序、时钟跳变、过大 covariance 和 TF 版本不一致均被拒绝或降级。
- [x] **M4-02 增量位姿图**
  - 目标：固定窗口接收 odometry、RGB-D 和几何验证 loop constraint；优化不重复求解全部历史。
  - 验收：合成漂移轨迹加入正确闭环后终点误差下降；错误/重力不兼容闭环不能改变地图。
- [x] **M4-03 子地图 revision 修正**
  - 目标：局部融合写入 submap；闭环只更新 submap 全局变换和 revision，前端不中断。
  - 验收：revision 单调递增；对象和路径可由旧 revision 确定性变换到新 revision。
- [x] **M4-04 重复与往返路径合并**
  - 目标：按空间重叠、方向无关几何和时间范围合并重复 traversal，同时保留每次观测证据。
  - 验收：同向和反向重复路径合并为一个 canonical path；相邻但不同房间路径不误合并。
- [~] **M4-05 外部 VIO 实机验收**
  - 目标：接入目标机器人选定的成熟 VIO/LIO，pose 20-50 Hz、P95 `<30 ms`。
  - 验收：带真值回放报告 ATE/RPE 达到部署阈值；无真值时至少通过闭环残差和重复路径一致性门禁。
  - 当前：外部 backend 契约、协方差/时钟门禁和增量图优化已完成；目标机器人尚未选定并接入 VIO/LIO，不能以数据集 pose 回放替代此项实机验收。

### M5：异步语义、可编辑和可更新记忆

- [x] **M5-01 版本化幂等语义回写**
  - 目标：SemanticCorrection 带 entity、time、revision、operation_id；支持 ack、重试、合并和 stale reject。
  - 验收：同 operation 重放只应用一次；旧 revision 不覆盖新状态；pending/applied/rejected 可追踪。
- [x] **M5-02 用户命名、别名与锁定**
  - 目标：区域/对象可人工命名并添加别名；人工锁定字段不被 DAM/VLM 自动更新覆盖。
  - 验收：重启进程、自动修正和地图 revision 更新后名称保持；别名可反向查询同一实体。
- [x] **M5-03 跨会话地图注册**
  - 目标：每个 session 保留自身坐标系；只有带 covariance、inlier 和残差证明的 SE(3) 注册才能激活。
  - 验收：不同起点的同一实体经已验证变换合并；未验证/低质量变换不能写入 canonical map。
- [x] **M5-04 冲突和更新策略**
  - 目标：区分新增、移动、消失、重命名和观测冲突；保留审计历史，可回滚 revision。
  - 验收：人工名称优先、较新几何观测按置信度更新、删除为 tombstone 且历史可查询。
- [x] **M5-05 在线 DSG 增量应用**
  - 目标：默认不把全部 DSG 修正推迟到退出；修正应用后产生 ack 和统计。
  - 验收：模拟 DSG 在运行中接收修正，`applied > 0` 且退出时 pending 为 0；前端不等待 DAM。

### M6：逐阶段质量门禁

- [x] **M6-01 统一 GateResult 和策略**
  - 目标：每个 gate 输出 PASS/WARN/FAIL、指标、阈值、证据路径和是否阻断。
  - 验收：报告为稳定 JSON；hard FAIL 返回非零退出码并阻止下一阶段。
- [x] **M6-02 输入/时间/校准门禁**
  - 目标：检查绝对时间单调、双目同步、pose 对齐、校准模型和 TF revision。
  - 验收：1 ns pose 错位、乱序、过大双目时间差和鱼眼误送 pinhole backend 均失败。
- [x] **M6-03 深度/pose/动态门禁**
  - 目标：检查深度有效率与时序一致性、pose jump/covariance、动态污染和晋升稳定性。
  - 验收：每种合成异常都命中唯一可解释的 failure code。
- [x] **M6-04 地图/语义/资源门禁**
  - 目标：检查 Mesh 连通性、闭环残差、语义回写覆盖率、队列延迟、drop rate 和峰值资源。
  - 验收：G1 报告能明确指出 Mesh 碎片和语义 pending 问题，不能错误标为通过。

### M7：实时回放、故障恢复与部署验收

- [x] **M7-01 一键实时回放入口**
  - 目标：从带绝对时间的 image-sequence 以 1 Hz 权威回放，支持 dry-run、resume、stop-after 和报告目录；5/10/15 Hz 仅为可选压力测试。
  - 验收：非均匀时间戳按真实间隔回放；manifest、metrics、quality report 可复现。
- [x] **M7-02 故障注入**
  - 目标：支持 pose 延迟/丢失、GPU stage 变慢/OOM、乱序、动态目标移入移出和 worker 崩溃。
  - 验收：每种故障有预期降级/阻断结果，无死锁、无无界队列、无静默数据错位。
- [x] **M7-03 checkpoint 与恢复**
  - 目标：持久化 scheduler offset、submap revision、动态对象和 semantic pending 队列。
  - 验收：中断恢复后不重复融合已提交帧，不丢失人工编辑，operation_id 保持幂等。
- [x] **M7-04 G1 全量软件验收**
  - 目标：对 G1 执行实时 dry-run/预计算深度回放，验证时间、队列、动态隔离、revision 和质量报告。
  - 验收：789 帧输入无时间错位；所有 hard gate 结果与实际产物一致。
- [~] **M7-05 目标设备 30 分钟验收**
  - 目标：真实传感器连续运行 30 分钟，无无界积压、内存/显存持续增长或进程失联。
  - 验收：输出 P50/P95/P99、ATE/RPE、Mesh 连通性、IDF1/HOTA、污染率和恢复次数；所有部署门限通过。
  - 当前：RTX 4090 上 789 帧全链路回放通过，但时长约 4.4 分钟且输入来自文件；仍需真实双目、IMU/VIO 和目标机器人连续 30 分钟采集。

## 6. 验收证据索引

| 任务 | 自动化证据 | 本机/数据证据 |
|---|---|---|
| M0 | `test_manifest_and_checkpoint.py`、`test_realtime_replay.py` | `output/realtime_g1_manifest_dryrun/run_manifest.json` |
| M1 | `test_realtime_core.py` | 全量报告各队列高水位 1、零 drop/handler error |
| M2 | `test_foundation_stereo_profiles.py`、`test_depth_realtime.py`、`test_depth_worker.py` | 全量深度 P95 246.40 ms，789 请求全部完成，0 restart |
| M3 | `test_dynamic_mapping.py`、`test_static_map_backend.py` | 全量动态污染 0%，167 个过期对象保留历史 |
| M4 | `test_online_slam_and_paths.py` | 27 个 submap、24 条 canonical path；M4-05 保持外部验收 |
| M5 | `test_map_memory.py`、`test_semantic_delivery.py`、`test_scene_graph_live_corrections.py` | 真实 Spark DSG 修正 `applied=1, pending=0` |
| M6 | `test_quality_gates.py`、`test_mesh_quality.py` | 全量 7 个 hard gate 全 PASS，Mesh 指标与产物一致 |
| M7 | `test_realtime_replay.py`、`test_static_map_backend.py` | G1 789/789、全阶段 789、零时间错位；真实 Hydra 2 帧中断后恢复到 4 帧，`rebuilt=2`；M7-05 保持外部验收 |

环境验收结果：

- 轻量 Conda 环境：`81 passed, 2 skipped`；可选 Spark/Hydra 集成按依赖缺失跳过。
- 项目 `.repro/venv`：`83 passed`，包含真实 Spark DSG 和 Hydra 时间接口测试。
- FoundationStereo 固定提交：`6e8806816b533e4d13ddbb95ffa907b797060a62`。
- 权重 SHA256：`8d7850b9dc68d1366722a02a39745704b1db41471211be6abd93ef463727e6be`，权重不进入仓库。
- 数据 `tick_index.json` SHA256：`6afb9064d2e3487333104708a61377f1b50c8dc6dcabccc7a9823fce4024560e`。

全量报告中的 `semantic.applied=0` 表示该 G1 序列未配置在线 DAM/VLM 请求，
不是投递失败；非空修正的实时 ACK、幂等、重试和真实 DSG 应用由 M5 自动化
集成测试验收。部署时若产生语义请求，`pending_ratio` 仍是硬门禁。

## 7. 测试矩阵

| 场景 | 必须验证 |
|---|---|
| pose 不变 + 新物体出现 | 帧保留；进入跟踪；动态/unknown 不直接融合静态 TSDF |
| pose 抖动 + 图像相同 | 严格重复帧优先丢弃，绝对时间映射不变 |
| 动态物体移动、停下、再次移动、离开 | 动态状态、静态晋升撤销、过期移除和轨迹保留正确 |
| 非均匀时间戳 | 速度、deadline、watchdog、传播时限均按真实时间计算 |
| 慢 DAM / 慢深度 | pose 与局部前端继续，队列有界，结果按 revision 回写 |
| 正确/错误闭环 | 正确闭环降低漂移，错误闭环被几何和重力门禁拒绝 |
| 同向/反向重复路径 | 合并 canonical path，观测次数和时间范围保留 |
| 用户命名后自动更新 | canonical name 不变，自动结果进入 alias/候选或被拒绝 |
| 跨会话不同起点 | 已验证 SE(3) 后合并；未验证时保持隔离 |
| 中断并恢复 | 不重复融合，不丢人工编辑，不重复应用语义操作 |

## 8. 验收命令

```bash
# 全部轻量与集成测试
pytest -q

# 使用项目环境执行 Spark/Hydra 可选集成测试
source .repro/venv/bin/activate
python -m pytest -q

# 实时回放 dry-run 与配置检查
python scripts/run_realtime_mapping.py --dataset DATASET --dry-run

# 使用已有深度的 G1 回放
python scripts/run_realtime_mapping.py --dataset DATASET --depth-backend precomputed --rate-hz 1

# 单独执行质量门禁
python scripts/evaluate_mapping_quality.py --run-dir RUN_DIR --config config/realtime_quality_gates.yaml

# FoundationStereo 在线 profile（在 foundation_stereo Conda 环境）
conda run -n foundation_stereo python scripts/run_foundation_stereo_depth.py \
  --dataset DATASET --profile online --checkpoint CHECKPOINT
```

## 9. 完成定义

项目进入“实时动态语义地图 MVP”必须同时满足：

1. M0-M7 的软件任务全部 `[x]`，硬件任务至少 `[~]` 且有真实失败报告。
2. G1 离线一键链路回归通过，绝对时间与 pose 行映射零错位。
3. 干净 HEAD 的 1 Hz 全数据回放使用真实分割/跟踪/DAM 与 Hydra ACK，零 drop/error、
   几何链端到端 P95 `<=1 s` 且无 hard gate 失败；实际深度性能由目标 GPU 报告独立判定。
4. 动态对象不会进入永久静态深度输入，人工命名跨重启和地图 revision 保持。
5. 所有未达硬件门限均保留为可量化阻塞项，不得以“代码已实现”标记项目性能完成。
