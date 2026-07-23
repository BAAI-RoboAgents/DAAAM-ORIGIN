# DAAAM 实时动态语义地图阶段进展报告

> 报告日期：2026-07-17  
> 当前阶段：软件 MVP 权威验收完成，进入真机 VIO 与目标设备部署验收阶段  
> 权威版本：`ab047db58c8657ba53cf15c7d9174eef0f25cc6a`  
> 地图链路目标：1 Hz；外部 VIO 输入目标：20–50 Hz

## 1. 阶段结论

项目的软件主链已经达到阶段性完成条件：真实分割、逐帧跟踪、动态隔离、
Hydra 静态建图、DAM 异步修正、DSG 持久化和质量门禁已在干净 Git HEAD 上
完成 789 帧全量权威回放。

最新结果为：

- 789/789 帧 dispatched/completed，0 drop、0 runtime error、0 resume。
- 7/7 个硬质量门全部 PASS，0 warning。
- 权威验证器 32/32 checks PASS，`passed=true`、`authoritative=true`。
- Tracking P95 为 38.02 ms，满足 `<50 ms`。
- 几何链 global E2E P95 为 301.57 ms，满足 `<=1 s`。
- DAM 修正最终 51 applied、5 superseded、0 pending/failed/rejected。
- Hydra Mesh 微小组件面积占比为 0.899%，满足 `<5%`。

按照 [TODOs.md](TODOs.md) 当前任务状态，35 个里程碑条目中 33 个为 `[x]`，
剩余两项为 `[~]`：

1. M4-05 外部 VIO 实机验收。
2. M7-05 目标设备连续 30 分钟验收。

因此当前总体判断是：**软件 MVP 已通过权威回放验收，但尚不能标记为真机部署完成。**

本文以 `ab047db` 的 1 Hz 真实 DAM 权威运行作为最新数值口径。
[TODOs.md](TODOs.md) 中仍保留的旧 5 Hz/no-DAM 全量数据属于历史基线，
不应再用于判断当前性能或语义投递状态。

## 2. 本阶段完成内容

### 2.1 权威实时目标统一为 1 Hz

- 地图派发频率统一为 1 Hz 最大频率。
- 保留原始绝对 `sensor_time_ns`，输入自身空窗不会被压缩或伪造帧填补。
- 5/10/15 Hz 只保留为可选、非阻断压力测试，不再作为发布门槛。
- 外部 VIO 的 20 Hz 以上输入要求保持不变，不与地图 1 Hz 目标混淆。

### 2.2 真实分割、跟踪和 DAM 异步旁路接通

- FastSAM TensorRT 分割真实运行，共执行 425 次分割。
- BotSort 对 789 个地图关键帧执行完整跟踪。
- ReID crop 改为批处理 GPU 上传，并限制 ECC 尾延迟。
- DAM worker 只在实时 GPU 空闲窗口运行，不阻塞几何主链。
- worker 退出前必须完成 drain；超时或强制终止会阻止权威验收。

### 2.3 DAM 修正写入最终 DSG

- 修正不再只停留在内存或临时状态，而是持久写入：
  - `dsg.json`
  - `dsg_with_mesh.json`
- 两个产物写入后执行 reload、实体/operation 校验和 SHA256 校验。
- Hydra shutdown 不再覆盖已经完成语义提交的最终 DSG。
- MapMemory correction 状态与 DSG durable ACK 相互交叉验证。

### 2.4 三个主要硬失败已关闭

| 历史问题 | 修复前证据 | 最新结果 | 状态 |
|---|---:|---:|---|
| Tracking 尾延迟 | P95 约 51.57 ms，超过 50 ms | 38.02 ms | PASS |
| DAM/DSG 最终投递不完整 | 历史 `applied=0` 或退出仍有大量 pending | 51 applied，0 pending/failed | PASS |
| Hydra Mesh 微岛过多 | 100 帧预检 tiny area 5.31%，超过 5% | 全量 0.899% | PASS |

Mesh 根因是 `extra_integration_distance` 生成的 padding voxel 被钳到
`1e-4` 最小权重，而网格阈值也默认是 `1e-4`，导致无有效观测的 padding
参与 Marching Cubes。当前在
[config/hydra_g1_high_quality.yaml](config/hydra_g1_high_quality.yaml) 中设置：

```yaml
active_window:
  mesh_integrator:
    min_weight: 0.5
```

该阈值过滤低权重 padding，同时保留正常 RGB-D 表面观测；回归测试会防止配置
再次落回错误默认值。

## 3. 最新权威运行

### 3.1 运行身份与可复现性

| 项目 | 最新值 |
|---|---|
| Git SHA | `ab047db58c8657ba53cf15c7d9174eef0f25cc6a` |
| Git 状态 | `git_dirty=false`，干净 detached worktree |
| 数据集 | G1 `20260713_170500`，时间对齐/针孔/双目/时序深度版本 |
| `tick_index.json` SHA256 | `d04ec6b16932293ab48156dde167503984c02b25c2c125176b548db382f9eaf8` |
| 地图频率 | 1 Hz，throttle 开启，不允许 source burst |
| 深度后端 | `precomputed` |
| 静态地图后端 | Hydra |
| 语义模式 | 真实 DAM |
| 运行平台 | RTX 4090 48 GB、32 CPU、约 128 GB RAM |
| Python | 3.12.3，项目 `.repro/venv` |

关键证据：

- [run_manifest.json](output/g1_1hz_dam_authoritative_ab047db_789f/run_manifest.json)
- [realtime_run_report.json](output/g1_1hz_dam_authoritative_ab047db_789f/realtime_run_report.json)
- [realtime_metrics.json](output/g1_1hz_dam_authoritative_ab047db_789f/realtime_metrics.json)
- [quality_report.json](output/g1_1hz_dam_authoritative_ab047db_789f/quality_report.json)
- [benchmark_validation.json](output/g1_1hz_dam_authoritative_ab047db_789f/benchmark_validation.json)

### 3.2 完整性与节拍

| 指标 | 结果 |
|---|---:|
| 请求 / 派发 / 完成 | 789 / 789 / 789 |
| Resume 帧数 | 0 |
| Drop / handler error | 0 / 0 |
| 各几何主阶段处理帧数 | 789 |
| Tracking 调用 | 789 |
| Segmentation 调用 | 425 |
| Scheduler elapsed | 1310.38 s（21 分 50.38 秒） |
| 绝对时间节拍 sleep | 1309.48 s |
| 报告吞吐 | 0.602 Hz |

报告吞吐低于 1 Hz 不是丢帧或性能失败。1 Hz 是最大派发频率，回放还保留了
原始采集时间中的非均匀间隙；权威报告同时证明所有 789 帧都已完成。

### 3.3 关键延迟

| 阶段 | Service P50 | Service P95 | 门限 | 结果 |
|---|---:|---:|---:|---|
| Pose | 0.41 ms | 0.51 ms | 30 ms | PASS |
| Tracking | 22.46 ms | 38.02 ms | 50 ms | PASS |
| Segmentation | 72.42 ms | 103.19 ms | 250 ms | PASS |
| Depth stage | 38.17 ms | 78.96 ms | 250 ms | PASS |
| Dynamic | 18.25 ms | 46.20 ms | 100 ms | PASS |
| Fusion | 21.86 ms | 41.77 ms | 250 ms | PASS |
| Global/Hydra | 118.95 ms | 175.21 ms | 250 ms | PASS |
| Semantic frontend | 210.39 ms | 388.11 ms | 5000 ms | PASS |

端到端 P95：

| 链路 | P95 | 门限 | 结果 |
|---|---:|---:|---|
| Global geometry | 301.57 ms | 1000 ms | PASS |
| Semantic frontend | 476.94 ms | 5000 ms | PASS |

注意：本次权威全量使用预计算深度。这里的 `depth stage` 延迟代表读取、校验和
调度预计算深度的链路延迟，**不等价于 FoundationStereo 在线模型推理耗时**。
目标设备上的在线 FoundationStereo 1 Hz、显存和 worker 稳定性仍需单独验收。

## 4. 质量门结果

| 硬门 | 关键实测 | 门限 | 状态 |
|---|---:|---:|---|
| 时间/校准 | stereo delta 0 ms，pinhole | `<=10 ms` | PASS |
| Pose | 最大平移 0.0817 m，最大旋转 5.36° | `<=0.5 m / <=20°` | PASS |
| Runtime | 0 drop、0 error，无 P95 超限 | 全部不超限 | PASS |
| Depth | 有效率 69.23%，时序一致性 97.85% | `>=15% / >=70%` | PASS |
| 左右一致性 | 一致性 83.29%，覆盖率 100% | `>=60% / >=25%` | PASS |
| Dynamic | 污染率 0%，unknown 34.17% | `<=1% / <=60%` | PASS |
| Semantic | 51 applied，0 pending/unmapped | pending ratio `<=10%` | PASS |
| Map | 最大组件面积 41.10%，tiny area 0.899% | `>=10% / <=5%` | PASS |

最终 [quality_report.json](output/g1_1hz_dam_authoritative_ab047db_789f/quality_report.json)
为 7/7 hard gate PASS、0 hard failure、0 warning。

## 5. DAM 与最终 DSG 证据

| 指标 | 结果 |
|---|---:|
| DAM prompts | 20 |
| Correction messages | 57 |
| 唯一 correction operations | 56 |
| Applied / superseded | 51 / 5 |
| Pending / failed / rejected | 0 / 0 / 0 |
| DSG mapped entities | 114 |
| DSG verified entities / operations | 36 / 51 |
| DSG pending / unmapped | 0 / 0 |
| Worker drain | complete |
| Worker exit code | 0 |
| Timeout / forced termination | false / false |

57 条 correction message 中有一条重复 `operation_id` 被幂等折叠，没有生成重复状态。

最终语义提交：

- `commit_valid=true`
- Commit manifest SHA256：
  `9e523fd60bfc9bbe6705910479972c37bfee1882207e187bda0f55ed72af0192`
- [semantic_dsg_commit.json](output/g1_1hz_dam_authoritative_ab047db_789f/hydra_realtime/backend/semantic_dsg_commit.json)
- [dsg.json](output/g1_1hz_dam_authoritative_ab047db_789f/hydra_realtime/backend/dsg.json)
- [dsg_with_mesh.json](output/g1_1hz_dam_authoritative_ab047db_789f/hydra_realtime/backend/dsg_with_mesh.json)

## 6. Hydra 地图结果

| 指标 | 结果 |
|---|---:|
| Vertices / faces | 29,634 / 41,612 |
| Surface area | 35.1629 m² |
| 0.1 mm 焊接后 components | 349 |
| Significant components | 117 |
| 最大组件面积占比 | 41.10% |
| Tiny component area | 0.3162 m² |
| Tiny component area ratio | 0.899% |
| Invalid faces | 0 |

已额外执行 `mesh.ply` 与 `dsg_with_mesh.json` 逐面交叉检查：

- 两者均为 29,634 vertices / 41,612 faces。
- Face topology 完全一致。
- Colors、labels、timestamps、first-seen timestamps 完全一致。
- 坐标最大差约 `5e-9 m`。
- Face index 全部合法，Spark DSG reload 成功。

## 7. 自动化测试与代码状态

- 项目测试：`163 passed`。
- 仅有 5 条第三方弃用警告，无测试失败。
- Ruff scoped check 与 `git diff --check` 通过。
- 权威运行使用独立干净 worktree，避免主工作区本地修改污染 provenance。

本阶段主要提交包括：

| Commit | 内容 |
|---|---|
| `5f2573d` | 权威实时目标统一为 1 Hz |
| `d2cd750` / `39c3aa1` | DAM 进程清理和单 GPU 空闲窗口调度 |
| `e19dc00` / `b053fb9` | 防止 Hydra 覆盖语义 DSG，并验证最终 artifacts |
| `15ed624` / `df16ee3` / `9c0ccb0` | MapMemory/DSG 交叉检查、完整 drain、最终 DSG 持久化 |
| `feeebf1` | Tracking 尾延迟优化 |
| `d252828` | 按实际 runtime 名称执行 semantic frontend 门禁 |
| `ab047db` | 过滤 Hydra 低权重 padding mesh |

## 8. 尚未完成：真机 VIO 与 30 分钟验收

### 8.1 当前已具备

- ROS 2 Jazzy 与 DAAAM overlay 可加载。
- OpenVINS 已在 `.repro/openvins_ws` 构建，固定提交：
  `69488123ed9362dd44b6f28e7f4680abbff1442b`。
- OpenVINS executable、launch 和 `/ov_msckf/odomimu` 输出链存在。
- 已有严格绝对时间、frame、covariance 和 no-latest-TF 契约。
- 已有真机录制和评估工具：
  - [record_openvins_acceptance.py](scripts/record_openvins_acceptance.py)
  - [evaluate_vio_acceptance.py](scripts/evaluate_vio_acceptance.py)

### 8.2 明确阻塞项

当前主机没有相机、IMU 或 OpenVINS ROS publisher，且缺少：

1. 同一 ROS domain 的左右目 `sensor_msgs/msg/Image` 和
   `sensor_msgs/msg/Imu` 实时流。
2. G1 cam0/cam1 到 IMU 的 `T_imu_cam`。
3. Camera–IMU `time_offset`。
4. IMU noise density、random walk、update rate 和轴向约定。
5. G1 专用 `estimator_config.yaml`、`kalibr_imucam_chain.yaml`、
   `kalibr_imu_chain.yaml`。
6. 已确认的 topic 名、QoS、frame_id、双目最大时间差和机器人往返路线。

默认 EuRoC 配置不能替代 G1 标定，789 帧文件 pose 回放也不能替代真机 VIO。

### 8.3 三阶段真机验收

| 阶段 | 建议录制 | 最小有效时长 | 晋级条件 |
|---|---:|---:|---|
| 快速联调 | 70 s | 60 s | 六项 VIO gate 全 PASS |
| 稳定性预检 | 310 s | 300 s | Gate 全 PASS，topic/时间/frame 稳定 |
| 正式验收 | 1830 s | 1800 s | VIO 与完整系统 30 分钟指标全 PASS |

VIO 硬门限：

- Odometry 平均频率 `>=20 Hz`。
- Pose 输入/处理延迟 P95 `<30 ms`。
- 首尾闭环平移 `<=0.50 m`。
- 首尾闭环旋转 `<=10°`。
- 往返重复路径刚体对齐 P95 `<=0.30 m`。
- IMU、左右目证据覆盖完整，时间严格单调，frame/covariance 合法。

正式 30 分钟还需同时证明：

- 无无界队列、内存或显存持续增长。
- 无关键进程失联、静默传感器中断或时间跳变。
- 输出 P50/P95/P99、Mesh 连通性、动态污染率和恢复次数。
- 输出 IDF1/HOTA；有真值时输出 ATE/RPE，无真值时明确采用 no-GT 闭环门禁。

## 9. 下一阶段建议

### P0：完成真机准入

1. 获取 G1 相机–IMU 联合标定和 IMU 噪声参数。
2. 固化 G1 OpenVINS 三份配置及 SHA256。
3. 接通 raw 左右目、IMU 和 `/ov_msckf/odomimu`。
4. 依次执行 70 s、310 s、1830 s 三阶段验收，前一阶段失败时禁止晋级。
5. 将 VIO、资源、Mesh、动态跟踪和进程健康合并为一个 30 分钟最终报告。

### P1：补强自动验收

1. 将 PLY/DSG 逐面一致性检查纳入权威 validator，替代当前独立复核步骤。
2. 增加真机 preflight：检查 topic 类型、QoS、频率、最大 gap、双目 skew、
   时间域、frame_id、标定完整性和文件哈希。
3. Recorder 增加 odometry 端到端 latency、消息缺口和丢消息统计。
4. 单独完成 FoundationStereo 在线 1 Hz 与资源稳定性验收。
5. 为 30 分钟测试增加 RSS/显存趋势和进程 restart 自动判定。

### P2：性能余量优化

- Tracking P95 已通过，但 P99 为 59.49 ms，可继续降低尾延迟。
- Semantic frontend P99 为 850.83 ms、最大 1.356 s，虽远低于 5 s 门限，
  仍可优化 crop 合并、缓存和 DAM prompt 去重。
- 最终版本可补跑 5/10 Hz 可选压力测试，评估过载策略，但不阻断 1 Hz 发布。

## 10. 阶段完成定义

当前已经满足“实时动态语义地图软件 MVP”的阶段定义：

- 干净 HEAD 的 1 Hz 全数据回放。
- 真实分割、跟踪和 DAM。
- Hydra durable DSG ACK。
- 789 帧零 drop/error。
- 几何链 E2E P95 小于 1 秒。
- 所有硬质量门和权威 validator checks 通过。

项目进入“真机部署完成”仍必须补齐：

- M4-05 外部 VIO 实机验收。
- M7-05 目标设备连续 30 分钟全系统验收。

在这些证据产生前，项目状态应保持为：**软件 MVP 完成，真机部署验收进行中。**
