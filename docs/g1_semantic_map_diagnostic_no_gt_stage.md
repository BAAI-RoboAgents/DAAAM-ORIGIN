# G1 语义地图当前阶段：无人工 GT 的全链路诊断

状态：`diagnostic_complete_existing_artifacts / diagnostic_gt_free`

数据：`/home/user/datasets/g1_20260724`

基准窗口：源帧 473–573；已有完整链路证据覆盖 844 个选中帧。原始
`docs/g1_semantic_map_experiments_v1_1.md` 仍是正式评测规范，本文件只改变当前执行阶段，
不降低或伪造正式验收条件。

执行配置：
[config/g1_semantic_map_diagnostic_no_gt.yaml](../config/g1_semantic_map_diagnostic_no_gt.yaml)

本轮证据入口：
[EVIDENCE_LEDGER.md](../experiments/g1_20260724_473_573_v1_1/diagnostic_no_gt/EVIDENCE_LEDGER.md)。
“完成”仅表示已把现有 844 帧构建产物做完无 GT 诊断、导出和封存。D0 已完成 13/13
类诊断覆盖，但预注册主采集器只严格通过 11/13，因此状态为 `FAIL / diagnostic only`；
独立 nominal 重跑、正式 Q1 与依赖人工 GT 的精度指标仍未完成。D0 统一入口为
`experiments/g1_20260724_473_573_v1_1/comparisons/diagnostic_gt_free_d0_all_families_20260728/REPORT.md`。

严格轨道的
`experiments/g1_20260724_473_573_v1_1/final/report.md` 和
`experiments/g1_20260724_473_573_v1_1/final/decision.json` 保持不变：formal matrix
run 数仍为 `0`，P1 仍因缺少 reviewed GT 而停止，正式排名不允许，held-out GT/query
未解封且 P8 未执行。本阶段不得覆盖这两个文件或把诊断结果回写成其正式结论。

## 1. 当前阶段目标

暂时跳过 L0–L6 人工 GT、双人独立标注、第三人裁决和独立 held-out 封存。直接使用现有
数据和现有标定，把输入、双目校正、关键帧、深度、尺度、位姿、时序过滤、动态隔离、
分割、跟踪、实体合并、DAM、Hydra、绑定、exact-label postpass 和查询资产逐环节跑通，
重点回答：

1. 每个环节是否收到满足契约的输入并产生完整、可追溯的输出；
2. 中间数据的分布、时序稳定性、空间一致性、资源开销和失败长尾是什么；
3. 已知故障注入后，指标和可视化是否能按预期响应；
4. 错误从哪个环节开始、如何传播到最终 DSG/mesh/query；
5. 哪些结论是精确工程事实，哪些只是代理指标，哪些必须等待人工 GT。

当前阶段可以选择 `engineering_candidate`，不能产生正式算法 winner、语义准确率排名或
V1.1 held-out 终验结论。

本阶段明确跳过独立 held-out 封存。原 held-out 分层中的 563–573 会随连续链路进入诊断，
且其输入、输出和 proxy 指标会被查看，因此它**仅用于诊断且已非盲测**。manifest 必须标记
`original_held_out_stratum: true`、`diagnostic_only: true`、`blind_test: false`；不得使用
`held-out PASS`、`P8` 或“未见测试集”等表述。

## 2. 结论等级

每个指标和图必须记录 `evaluation_basis`：

| 等级 | 含义 | 当前可用示例 |
| --- | --- | --- |
| `exact` | 由系统契约或持久化产物直接验证 | 帧数、时间、hash、LR 覆盖、drop/error、pending、commit、mesh 拓扑、延迟 |
| `proxy` | 与目标质量相关，但没有独立人工真值 | 自动特征 dy、LiDAR 边界外深度误差、mask 时序稳定度、内部 track/entity 一致性 |
| `unavailable` | 缺少所需人工或独立真值 | Mask AP、HOTA/IDF1、实体 P/R/F1、名称准确率、查询 Recall/MRR |
| `not_applicable` | 当前窗口不存在可判定条件 | 已确认没有闭环时的全局闭环增益 |

所有 `proxy` 图必须显示 `NO HUMAN GT / PROXY`，不得把自动 FastSAM、DAM、MapMemory 或
Hydra 输出回灌为自己的真值。

## 3. 逐环节评估和证据

| 环节 | 精确工程指标 | 无 GT 代理/不可得项 | 必须保存 |
| --- | --- | --- | --- |
| E0 输入与 provenance | 文件存在、帧/时间/pose 单调、hash、坐标/单位契约 | 人工挑战标签不可得 | manifest、tick index、环境、命令、配置、git patch、文件清单 |
| E1 同步与校正 | stereo delta、覆盖、共同有效面积、相机顺序负控 | 自动对应 dy/正视差为 proxy | 原/校正图、匹配点、ROI、逐帧统计、最差帧 |
| E2 关键帧 | 保留率、原因、最大时间/位姿空洞、成本 | 真实语义事件召回不可得 | 每帧决策、选/未选索引、事件代理图 |
| E3 深度 | 有效率、LR 一致性、置信、遮挡、延迟、显存 | 绝对稠密精度为 proxy | disparity、raw/valid depth、confidence、consistency、occlusion、metadata |
| E4 尺度 | calibration-only 来源、scale、CI、hash | 遮挡边界外 LiDAR 误差为 proxy | 原/缩放深度、LiDAR 投影、分距离统计 |
| E5 时序诊断 | 重投影 agreement、误差、故障注入剂量响应 | 自然失败原因仍为 diagnostic | 全部 pair 记录、P50/P95/worst 面板 |
| E6 位姿 | 约束数、inlier、残差、收敛/fallback、轨迹变化 | 绝对轨迹精度不可得 | source/refined pose、约束、优化日志、残差图 |
| E7/E8 闭环/全局 | proposal/verification/拒绝理由、source 保持 | 无真闭环时全局增益 N/A | 候选、验证图、pose graph、fallback 证据 |
| E9 深度过滤 | 删除量、前后有效率、时序 agreement 变化 | 薄结构/真实边缘保留为 proxy | 前后 depth、reject/support/evidence 图和逐帧统计 |
| E10 动态隔离 | dynamic/unknown 应用与静态深度泄漏 | dynamic F1、static FPR、ghost 真值不可得 | 三类 mask、static depth、面积时序、注入响应 |
| E11 分割 | mask 数/面积/调用失败/延迟 | Mask AP、boundary F、small recall 不可得 | 原 mask、overlay、空/碎片/大 mask 失败集 |
| E12 跟踪 | 生命周期、内部碎片、carry-forward、延迟 | HOTA/IDF1/真实 ID switch 不可得 | track timeline、逐帧实例、断裂/合并证据 |
| E13 实体合并 | merge/reassign/conflict、距离与观察数 | entity P/R/F1 不可得 | MapMemory DB、merge graph、实体版本和审计日志 |
| E14 DAM | 调用、覆盖、延迟、pending/drain、跨观察一致性；可做有限总体双遍 Codex 视觉复核、分歧与显著物体漏识诊断 | 正式名称准确率/召回率仍不可得，Codex 双遍不替代双人人工 GT | prompt/response、crop/观察、operation、拒绝原因、逐实体 RGB-mask-描述面板、两遍原始判定、保守共识、分歧和显著物体清单 |
| E15 增量链 | 各层产物数、覆盖和成本 | 自动 smoke query 增益仅 proxy | geometry/frontend/DAM 三层漏斗 |
| E16 Hydra | 顶点/面/面积/组件、对象数、内存/延迟 | 表面精度/对象召回为 proxy/unavailable | TSDF 相关日志、mesh、DSG、top/side preview、timing |
| E17 绑定 | applied/rejected/no-mesh/conflict、距离/AABB gap | binding P/R/F1 不可得 | 每次 binding event、entity↔mesh 图、拒绝样本 |
| E18 postpass | 帧覆盖、hash、pending/unmapped/error、durable commit | 无 | label frames、plan、日志、commit 和重载检查 |
| Q1 查询 | embedding/manifest/evidence 可追溯性 | Recall/MRR/FAR 不可得 | smoke query top-k、证据图、拒绝原因；不写入冻结 L6 |

## 4. 证据保留策略

本阶段默认 `retain_all_intermediates: true`，禁止清理或覆盖上游产物。每次运行使用新的
run 目录，并保留：

- 所有帧的 raw numeric products 和可用的全分辨率 mask/label/depth；
- 全部候选、被拒绝项、失败项、空输出和恢复记录；
- 逐命令 stdout/stderr、`/usr/bin/time -v`、GPU/RAM/queue/latency 时间序列；
- 完整 CLI、展开后的 YAML、模型/checkpoint/repository hash、Python/CUDA/driver；
- 原始、prepared、selected、geometry、semantic 和 map 之间的双向帧索引；
- `artifact_inventory.jsonl` 与 `artifact_inventory.csv`，至少包含路径、类型、字节数、
  mtime、SHA-256、来源阶段和上游 artifact ID；
- 逐帧 CSV/JSON、阶段摘要 JSON、失败案例索引、可视化、HTML/Markdown 报告；
- 对大产物优先原地冻结并用 hash 引用，避免无意义复制；若要修改上游，先建立只读快照。

每个 stage 至少落盘：

```text
stage_manifest.json
inputs.json
outputs.json
per_frame.jsonl
metrics.json
failures.jsonl
commands.jsonl
stdout.log
stderr.log
telemetry.jsonl
visualizations/
artifact_inventory.json
```

`stage_manifest.json` 必须记录 stage mode、起止时间、expected/processed/missing/dropped、
代码/配置/模型 hash 和上游 artifact ID。`outputs.json` 中每个产物必须记录 producer、
source-frame 范围、shape/dtype、坐标系、单位和 SHA-256。保存所有 RGB/校正图、disparity、
raw/filtered depth、confidence/LR/occlusion、pose/约束、support/judged、三类运动 mask、
FastSAM mask/overlay、track/entity operation、label frame、SQLite、mesh、DSG、embedding、
evidence，以及后端实际产生的候选、被拒绝项、logit/feature。截图只是索引，不能替代
PNG/NPY/PLY/DB 等原始数据。

运行结束后生成全文件 inventory，包含相对路径、类型、字节数、mtime、SHA-256、producer
stage 和 parent artifact hash，并保存 inventory 自身 hash。数据库须在 producer 关闭后
做一致性快照；SQLite WAL 必须一并保留或在 checkpoint 后记录操作。

证据分为四层，均不得由下层替代上层：

1. `source`：原始数据和既有上游产物原地冻结，以逐文件 SHA-256 和树根 hash 引用；
2. `stage-native`：模型/算法原生的 NPY、PNG、PLY、JSONL、SQLite、DSG、mesh、日志及
   rejected/empty/failure 输出，保留全分辨率；
3. `derived`：逐帧联表、统计摘要、failure index、关联/传播分析，必须可追溯到
   stage-native artifact；
4. `presentation`：图表、montage、Markdown/HTML，只作入口，不能代替原生证据。

每次分析完成后还要对 `derived` 和 `presentation` 再做一次 SHA-256 inventory，并将
该 inventory 的产品 hash 写入 provenance。这样形成“原始/中间产物 → 分析表 →
图表/报告 → evidence inventory”的闭合证据链。当前代码、配置、协议和测试目录也纳入
内容哈希；工作树未提交时，不得只记录 Git commit 而忽略实际文件内容。

当前磁盘使用率较高时，仍不得静默删除证据。需要降级时只能先停止新运行并报告预计空间。

## 5. 执行顺序

1. 冻结现有输入、V1/V2 组合校正、LiDAR scale proposal、代码和模型 provenance；
2. 复验原始数据、坐标、时间、校正和 E0/E1 硬契约；
3. 对已有 844 帧全链路产物做逐帧重分析，建立基线和失败索引；
4. 在 473–573 窗口执行 nominal、可用的 isolated 和 D0 fault-injection 诊断；
5. 只对不依赖人工 GT 的因素做参数/交互分析，报告效应量和资源代价；
6. 直接 RGB-D 融合预览仍需单独人工几何确认；“跳过人工标注”不自动授权
   `--accept-direct-fusion-preview`，未签字时后续地图只能标为 diagnostic；
7. 对原 held-out 分层 563–573 只作为连续链路的诊断尾段运行自动指标，并明确标记
   `original_held_out_stratum=true`、`diagnostic_only=true`、`blind_test=false`；若结果
   参与调参，未来正式 P8 必须更换未暴露窗口；
8. 生成完整证据清单、可视化和阶段结论；人工 GT 恢复后再回到 V1.1 P1。

## 6. 明确限制

- 473–573 是同一连续近距离室内机动，不支持闭环、跨房间拓扑、长期实体一致性或泛化结论。
- P0 的 101 帧 prepared 用于完整输入/极线审计，其中 7 帧左右时间差超过 10 ms。双目
  深度链只能使用冻结的 94 帧 eligible 子集；7 帧必须保留 lineage 和失败占位，不能送入
  深度，也不能静默删除。101 帧 P0 审计和 94 帧 mapping 上游是两条不同证据链。
- 现有 844 帧运行可证明链路闭合，但发生过从 600 帧恢复；它不是 zero-resume 权威运行。
- 当前全量运行的 `global` service P95 超过既定 250 ms 门，因此功能性地图可用，但整体
  quality 不能标为 PASS。
- 自动校正、LiDAR scale、语义 label、track、entity 和 query 都不能替代独立人工 GT。
- 该阶段的最终标签最多为
  `engineering-diagnostic-on-g1-existing-data`。

## 7. 恢复人工标注和正式轨道的条件

正式轨道必须重新从 V1.1 P1 开始，不能把当前 proxy 或自动提案改名为 GT：

1. 完成 L0–L6 reviewed GT；
2. 人工冻结 challenge tags 和 25 个 anchor，且选择不依赖当前候选成败；
3. 完成双人独立标注、第三人裁决，并保留原始版和裁决版；
4. 仅用 calibration 473–487 生成并签字冻结 `scale_473_487.json`，补齐遮挡边界
   `not-judgeable` 和距离分箱；
5. 完成 GT pilot，在查看正式候选前冻结 GT-dependent thresholds、最小实际差异和统计规则；
6. L5 reviewed GT 冻结后生成、审核并封存 48-query set；
7. 由独立审核者封存正式 GT/query，只向调参人员暴露版本号和 SHA-256；
8. 因 563–573 已进入本阶段诊断且不再盲，正式 P8 改用未参与本阶段分析的新窗口；
9. 冻结代码、配置和模型，从空目录重新运行 formal matrix 和三 seeds；本阶段结果不能
   直接晋级为 formal；
10. 独立审核通过以上证据后，才可更新严格轨道 decision，并在 P8 一次性解封终验。
