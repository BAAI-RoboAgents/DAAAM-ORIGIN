# DAAAM 物体感知、单一 ID 与语义地图性能提升开发计划

本文把 [`Intro.md`](Intro.md) 中的技术分析转化为可执行开发计划，目标是在不破坏现有
RGB-D/Hydra 几何主链的前提下，提高物体发现率、跨帧身份一致性、开放词汇命名质量和
`entity_id → Hydra object mesh` 的最终绑定覆盖。

操作命令、完整构建阶段和当前参数定义仍以 [`Pipeline.md`](Pipeline.md) 为准；本文只管理
研发顺序、交付物、测试和验收门槛。

## 1. 目标与非目标

### 1.1 总体目标

建立以下可验证的一对一关系：

```text
一个真实物体
    ↔ 一个 MapMemory entity_id
    ↔ 一个 canonical semantic_id
    ↔ 一个有真实 mesh 的 Hydra O(n)
    ↔ 一个稳定、可解释的开放词汇名称
```

最终质量不再只用“生成了多少名称”衡量，而使用四层联合指标：

```text
真实物体召回
× 单一 ID 正确率
× 正确命名率
× 真实 mesh 绑定率
```

### 1.2 具体目标

1. 修正或验证 RGB/BGR、tracker tick、semantic ID 和恢复运行等基础契约。
2. 为每一层建立可观测漏斗，能明确对象在哪一级丢失或被错误合并。
3. 降低同一物体多 ID、多个物体共用一个 ID、重复 DAM correction 等问题。
4. 提升小物体、遮挡目标和不同视角下的 mask 与名称稳定性。
5. 提升已命名 entity 到真实 Hydra object mesh 的绑定覆盖，同时控制误绑定。
6. 建立 FastSAM、SAM/SAM2 video 和未来三维分割模型的公平 A/B 评测框架。
7. 为跨天、不同起点、同一世界坐标系下的增量地图奠定身份恢复基础。

### 1.3 本计划暂不直接解决

- 不把 `quality_passed=true` 当作语义准确率结论。
- 不在缺少人工真值时宣称某个模型“识别率更高”。
- 不把自由文本 caption 直接等同于规范物体类别。
- 不仅凭同一坐标系 pose 就承诺跨任务全局 ID 稳定。
- 不在缺少漏斗证据时盲目降低 Hydra 体积、观察次数或绑定距离门槛。
- 不把 SAM2 单帧 AutomaticMaskGenerator 误称为视频跟踪方案。

## 2. 当前基线

### 2.1 工程运行基线

| 指标 | `fast_semantic_10cm` | 旧 `tabletop_offline_q16` |
| --- | ---: | ---: |
| 完成帧 | 1,575 | 1,098 |
| FastSAM 调用 | 679 | 585 |
| FastSAM detections | 33,344 | 32,280 |
| depth-valid tracked observations | 16,177 | 14,994 |
| 进入 MapMemory 的唯一局部 tracks | 503 | 760 |
| MapMemory entities | 314 | 730 |
| 送 DAM 的唯一 entities | 240 | 480 |
| DAM corrections | 293 | 491 |
| 历史唯一非零 semantic IDs | 360 | 777 |
| Hydra commit `object_count` | 92 | 51 |
| 已验证 mesh-bound entities | 74 | 22 |
| entity→mesh 交付覆盖 | 30.83% | 4.58% |
| 按 commit object 数计算的绑定占比 | 80.43% | 43.14% |
| `rejected_no_mesh` operations | 183 | 464 |

证据来源：

- [`fast_semantic_10cm/realtime_run_report.json`](output/g1_260720_1424_indoor_fast_semantic_10cm/realtime_run_report.json)
- [`fast_semantic_10cm/semantic_dsg_commit.json`](output/g1_260720_1424_indoor_fast_semantic_10cm/hydra_realtime/backend/semantic_dsg_commit.json)
- [`tabletop q16/realtime_run_report.json`](output/g1_20260717_tabletop_offline_q16_20260720_142700/realtime_run_report.json)
- [`tabletop q16/semantic_dsg_commit.json`](output/g1_20260717_tabletop_offline_q16_20260720_142700/hydra_realtime/backend/semantic_dsg_commit.json)

这些是管线转化率，不是带真值的识别准确率。两次运行的 merge/binding 阈值和 Hydra 配置
不同，不能作为只改变单一变量的严格 A/B。

### 2.2 已知高优先级风险

| 风险 | 当前证据 | 影响 |
| --- | --- | --- |
| RGB/BGR 接口约定可能不一致 | 数据层输出 RGB，固定 Ultralytics/BoxMot 路径按 BGR 处理 | 分割、ReID、ECC 均可能受影响；需 A/B 证明 |
| 非分割帧向 BotSort 传空 detections | 两次实跑 bbox warp 都为 0，传播全部 carry-forward | 新 track 确认、轨迹连续和 mask 传播存在风险 |
| FastSAM 与 BotSort 阈值未联动 | FastSAM conf 低于 BotSort new-track 门槛 | proposal 增多但新轨迹召回未必提升 |
| mask 重叠 first-wins | 按 track ID 抢占空标签像素 | 低质量或背景 mask 可能覆盖真实物体 |
| MapMemory 主要按最近三维中心合并 | 初始标签多为 `unknown`，名称门区分力弱 | 近邻物体误合并或同物体碎片化 |
| DAM 同实体可能重复提交 | 240 entities 产生 293 corrections | 名称覆盖、重复计算和版本冲突 |
| 历史 label PNG 未完全 canonical remap | 唯一 semantic IDs 多于最终 entities | Hydra postpass 可能看到同实体的旧 ID 碎片 |
| 恢复运行未完整恢复 identity 状态 | checkpoint 不含完整 tracker/ID/mask 状态 | tracker ID 重用和旧 entity 粘连风险 |
| Hydra object 形成/绑定损失大 | 240 个 DAM entities 最终绑定 74 个 | 大量已命名对象无法进入最终语义 mesh |

## 3. 实施原则

1. **先正确，再换模型。** P0 未通过前，不用 SAM/SAM2 结果判断模型优劣。
2. **一次只改变一个主要变量。** 每个 A/B 固定输入帧、深度、pose、随机种子和其他参数。
3. **每层独立验收。** 分割、跟踪、实体关联、命名和 mesh 绑定分别有指标。
4. **任何自动合并都必须可审计。** 保存候选、代价、阈值、最终决策和回滚信息。
5. **不牺牲 precision 换表面 coverage。** 绑定数增加必须同时抽样验证误绑定率。
6. **保留可回退基线。** 新逻辑使用配置开关，能够恢复当前行为进行对照。
7. **产物可复算。** 统计来自持久化报告/数据库，而不是仅存在于 Dashboard 内存。

## 4. 阶段总览与依赖关系

| 阶段 | 目标 | 前置依赖 | 核心交付物 | 进入下一阶段的门槛 |
| --- | --- | --- | --- | --- |
| P0 | 正确性与可观测性 | 当前基线 | 数据契约测试、完整漏斗、恢复状态保护 | 所有契约测试通过；报告可定位每一级损失 |
| P1 | 单一 ID 关联 | P0 | 多特征全局关联、稳健实体状态、可回滚合并 | 标注集 IDF1/entity purity 明显优于 P0 |
| P2 | mask 质量 | P0；建议接 P1 | 去重/层级/重叠仲裁、原始 mask 保真 | 小物体 recall 提升且重复/嵌套率下降 |
| P3 | DAM 命名 | P0、P1、P2 | 多视角选择、结构化输出、规范词汇 | 名称准确率和多视角一致率提升 |
| P4 | Hydra object/mesh | P0–P3 | Hydra 全漏斗、场景化配置、全局 mesh binding | entity→mesh 覆盖提升且误绑定受控 |
| P5 | 模型 A/B | P0 至少完成 | FastSAM/SAM2/3D 模型公平评测 | 以真值和系统成本作出模型选择 |

推荐顺序：

```text
P0.1 数据契约
    ↓
P0.2 Tracker 生命周期 ──> P0.3 漏斗遥测
    ↓                         ↓
P0.4 DAM/ID 去重与恢复 ──────┘
    ↓
P1 单一 ID
    ├─> P2 mask 质量
    └─> P3 DAM 多视角命名
              ↓
         P4 Hydra mesh
              ↓
         P5 模型 A/B
```

## 5. P0：正确性与可观测性

P0 是阻断阶段。完成前不进行大规模模型替换或激进阈值调优。

### P0.1 统一颜色通道契约

任务：

1. 明确项目内部统一使用 RGB。
2. 在 FastSAM/Ultralytics 和 BoxMot 适配边界按第三方真实契约转换。
3. SAM/SAM2、DAM 保持其各自明确的输入约定。
4. 增加红/蓝色块 sentinel 测试，验证各模型实际接收到的通道。
5. 用相同 100–300 帧执行改前/改后 A/B，记录 proposal、ReID 距离和跟踪结果。

主要修改位置：

- [`scripts/run_realtime_mapping.py`](scripts/run_realtime_mapping.py)
- [`src/daaam/utils/segmentation.py`](src/daaam/utils/segmentation.py)
- [`src/daaam/tracking/services.py`](src/daaam/tracking/services.py)
- [`src/daaam/realtime/semantic.py`](src/daaam/realtime/semantic.py)

验收：

- sentinel 测试能在每个适配边界准确检测 R/B 交换；
- 同一图片经适配后，FastSAM 与 DAM 看到的可视颜色一致；
- 报告记录通道契约版本，历史结果可追溯；
- 不在缺少 A/B 数据时把该风险写成已确认精度根因。

### P0.2 修复非分割帧的 tracker 生命周期

任务：

1. 最小实现：只在真实 segmentation frame 调用完整 `BotSort.update()`。
2. 非分割帧先继续走独立 mask propagation。
3. 后续实现 predict-only：只执行 Kalman/ECC 预测，不改变 confirmed/lost/removed 状态。
4. 把传播步数从隐式处理帧语义补充为时间戳约束，防止掉帧后传播过久。
5. 记录 confirmed/unconfirmed/lost/removed 的状态转移计数。

验收：

- 非分割帧不再被解释为“传感器确认没有目标”；
- 新出现目标可跨分割间隔完成确认；
- 若启用 predict-only，`propagation_bbox_warps` 在有相机/目标运动的测试段大于 0；
- ID switch 和短轨迹比例不高于当前基线；
- 保留 `legacy_empty_tick` 配置开关用于回归对照。

### P0.3 暴露 BotSort 参数并建立前端漏斗

需要暴露到 YAML、CLI 和 Dashboard：

- `track_high_thresh`
- `track_low_thresh`
- `new_track_thresh`
- `match_thresh`
- `proximity_thresh`
- `appearance_thresh`
- `track_buffer`
- `frame_rate`
- `with_reid`
- `cmc_method`

新增逐帧/累计漏斗：

```text
FastSAM proposals
→ score > track_high_thresh
→ score >= new_track_thresh
→ first/second association matches
→ new tracks
→ confirmed tracks
→ output tracks
→ mask depth-valid
→ MapMemory entities
```

验收：

- 运行报告和 Dashboard 对同一任务的计数一致；
- 每个阈值记录“配置值、有效值、来源”，避免第三方隐式默认值；
- FastSAM threshold 调整时能直接看到新 track 转化率变化；
- 新字段有配置解析、边界值和报告序列化测试。

### P0.4 DAM 提交去重和真实观测计数

任务：

1. `semantic_minimum_observations` 按唯一 `entity_id + sensor_time` 计数。
2. 同一 entity/revision 同一时间只允许一个 pending prompt。
3. 一个 batch 内按 `entity_id` 去重，保留质量最高候选。
4. `unknown`、空结果或低置信结果不永久封死重试机会。
5. 报告 unique prompted entities、duplicate suppressed、retry 和 correction versions。

验收：

- 同帧多个 tracks 合并同一 entity 时 observation count 只增加 1；
- 单一 revision 不产生多条无意义 correction；
- 当前 fast 基线中 293 corrections / 240 entities 的差额能被明确分类；
- 队列排空、幂等 operation 和 supersede 行为回归测试通过。

### P0.5 canonical semantic-ID remap

任务：

1. 持久化 provisional semantic ID → canonical entity semantic ID 的 remap journal。
2. exact-label postpass 前验证每个非零 label 都有唯一 entity owner。
3. 选择流式重映射或安全重写历史 PNG，避免同一 entity 保留多个历史 ID。
4. semantic ID 达到 9999 前发出硬错误，不能静默越过 labelspace。
5. label manifest 记录 remap hash，并参与 postpass 一致性校验。

验收：

- postpass 输入的唯一非零 semantic ID 数不再因历史临时 ID 高于 canonical entity 数；
- 0 始终作为背景；有效实例 ID 限于 `1..9999`；
- 中断并恢复 remap 不会产生半写入 PNG；
- manifest/hash 不一致时 postpass 拒绝提升临时产物。

### P0.6 恢复运行的 identity 状态

至少保存或显式隔离：

- BotSort 生命周期和下一个 track ID；
- `(session_id, local_entity_id) → entity_id`；
- `entity_id ↔ semantic_id`；
- 下一个 semantic ID；
- mask propagation 状态；
- prompt/correction revision；
- label remap journal 和配置 hash。

若短期无法完整恢复 tracker，则每次 resume 必须生成新的 session epoch，禁止新 tracker ID 与
旧的固定 `replay` session 无条件碰撞。

验收：

- 在固定帧中断并恢复，恢复后的 entity/semantic ID 不与旧对象错误粘连；
- 同一任务连续运行和中断恢复的最终映射差异可解释；
- checkpoint schema 有版本号、向后兼容策略和损坏检测；
- 增加 resume 前后身份一致性集成测试。

### P0.7 全链路拒绝原因

报告需要区分：

```text
segmentation_filtered
tracker_not_confirmed
mask_depth_invalid
entity_create / entity_merge / entity_conflict
prompt_not_ready / duplicate_suppressed / queue_full
hydra_track_not_allocated
object_volume_rejected
object_not_reconstructed
binding_distance_rejected
binding_entity_conflict
binding_semantic_owner_conflict
rejected_no_mesh
```

验收：任一最终未绑定 entity 都能从报告回溯到至少一个明确拒绝阶段，而不是只看到统一的
`rejected_no_mesh`。

## 6. P1：提高单一 ID 质量

### P1.1 设计实体状态

为每个 MapMemory entity 持久化：

- 稳健三维中心和 covariance；
- 累计 AABB/尺寸统计及不确定度；
- 最近观测时间、观测次数和可见状态；
- ReID/CLIP appearance prototype 与样本数；
- canonical label、aliases、名称置信度和锁定状态；
- 所属 session/local tracks；
- merge/split/reassign 审计历史。

dimensions 不再只保留创建时估计，应使用对异常值不敏感的累计更新。

### P1.2 同帧全局一对一关联

把逐条最近邻改为同帧联合匹配。候选代价建议为：

```text
cost = w_center × Mahalanobis(center)
     + w_gap × normalized_AABB_gap
     + w_iou × (1 - AABB_IoU)
     + w_app × cosine_distance(appearance)
     + w_size × size_inconsistency
     + w_time × temporal_penalty
```

约束：

- 同帧一个可见 entity 最多匹配一个 track；
- 一个 track 最多匹配一个 entity；
- 几何、外观和时间均有独立 hard gate；
- `unknown` 不能使名称门完全失去区分力；
- 匹配矩阵、每项代价和最终 Hungarian 结果写入审计。

### P1.3 支持纠错

实现：

- track reassign；
- entity split；
- 受控 entity merge；
- 错误 merge 回滚；
- correction 与 semantic ID 的随动迁移；
- 冲突时人工锁定优先。

所有操作必须有幂等 operation ID、前后版本、影响的 label frames 和 DSG 节点清单。

### P1 验收

- 标注集 IDF1/HOTA 优于 P0 基线；
- 每个 GT 物体的 entity 数下降；
- entity purity 上升，多个 GT 共用一个 entity 的比例下降；
- 相邻同类物体测试不因放宽 merge distance 被合并；
- 关联延迟 P95 满足准实时预算；
- 旧 MapMemory 数据库有迁移和只读回退路径。

## 7. P2：提高 mask 质量

### P2.1 候选过滤和层级

任务：

- 让 `max_mask_region_area` 真正生效；
- 区分 thing/stuff，不让地面/墙面与普通实例使用同一逻辑；
- 用 mask IoU、containment 和深度层识别 duplicate/nested/whole-part；
- 保存“完整物体”“部件”“背景表面”的显式层级；
- 对零 mask、尺寸异常和 bbox/mask 不一致进行硬校验。

### P2.2 重叠像素仲裁

取消 track-ID first-wins。候选像素评分至少考虑：

- segmentation score；
- mask 深度有效率；
- 中心深度连续性和边界深度跳变；
- track 稳定度；
- whole/part 优先级；
- 遮挡前后关系。

输出每帧 overlap audit，包括竞争 IDs、得分、胜者和重叠像素数。

### P2.3 保留原始 mask

- DAM 主输入使用原始 RLE/PNG bool mask；
- polygon 仅用于索引、轻量传输或可视化；
- 若必须 polygon 化，保留孔洞和多个连通分量；
- 对杯把、椅背孔洞、栏杆、线缆建立专项回归样本。

### P2.4 高分辨率与小物体

- 固定 P0/P1 后再重建更高分辨率 FastSAM TensorRT engine；
- 分辨率、conf、NMS IoU、最小/最大面积联合网格搜索；
- 报告小物体 recall、proposal 数、重复率、P50/P95 和显存；
- 不接受只提高 proposal 数但 confirmed/entity/mesh 无改善的配置。

### P2 验收

- mask AP50/AP75、boundary F1 和小物体 recall 提升；
- duplicate/nested mask 比例下降；
- overlap 像素有可解释的评分决策；
- depth-valid mask 转化率不下降；
- DAM focal mask 的孔洞和细结构回归测试通过。

## 8. P3：提高 DAM 命名质量

### P3.1 多视角候选窗口

每个 entity 保留 2–3 个互补视图，评分考虑：

- mask 面积与清晰度；
- 可见比例和遮挡；
- 深度有效率；
- 与已有视图的外观/视角差异；
- 背景占比；
- track/entity 稳定度。

达到观察门槛只表示“允许提交”，不再强制立即使用当前帧。

### P3.2 mask-aware 输入和结构化输出

建议 DAM 输出：

```json
{
  "category": "office chair",
  "caption": "a red office chair beside the desk",
  "attributes": ["red", "wheeled"],
  "is_whole_object": true,
  "confidence": 0.78
}
```

要求：

- `category` 与 `caption` 分离；
- category 经过规范词汇表/embedding 归一化；
- 部件描述不能覆盖已确认的 whole-object canonical label；
- confidence 要有明确来源和校准，不能继续把回退 0.5 当作概率；
- 保留原始模型文本供审计。

### P3.3 多视角融合和重试

- 对 2–3 个视图分别预测或联合推理；
- 对规范 category 投票，对属性做集合合并；
- 视图冲突、`unknown`、低置信和部件名允许重试；
- 同一 entity 的旧名称作为 alias，canonical 更新遵循版本/锁定规则；
- 持久化 CLIP/ReID 特征并接入 P1，而不是只计算不使用。

### P3 验收

- category accuracy、caption 可用率、多视角一致率提升；
- 部件误命名率下降；
- duplicate prompt/correction 明显下降；
- `unknown` 有受控重试且不会形成无限队列；
- 人工锁定名称不会被自动 correction 覆盖。

## 9. P4：Hydra object mesh 与最终绑定

### P4.1 建立 Hydra 对象漏斗

至少记录：

```text
nonzero label pixels
→ InstanceForwarding candidates
→ ExternalTracker tracks
→ allocation confidence passed
→ volume gate passed
→ reconstruction confidence passed
→ real object node/mesh
→ entity binding candidate
→ accepted binding
```

每一级包含 ID、观察次数、体积、range、置信度、拒绝原因和关联的 semantic/entity ID。

### P4.2 场景化配置

分别维护：

- room-scale：更大范围、较粗背景体素、较高小噪声抑制；
- tabletop：较近范围、更细对象重建、更低最小体积；
- future lidar-aligned：激光雷达几何用于全局结构或位姿约束，但语义对象仍需明确 RGB-D
  可见性和时间同步。

优先评估参数：

- `object_detector.max_range`
- `min_cluster_size`
- `tracker.min_num_observations`
- `min_object_allocation_confidence`
- `min_object_volume` / `max_object_volume`
- `min_object_reconstruction_confidence`
- `object_reconstruction_resolution`
- `min_reconstruction_resolution`
- binding center distance / AABB gap

### P4.3 全局 mesh binding

把按处理顺序的候选抢占改为全局一对一分配，代价组合：

- semantic ID match；
- AABB gap/IoU；
- center distance；
- entity/object 尺寸一致性；
- 时间和观察支持；
- mesh 是否已被预留。

继续保持 `allow_unmeshed_fallback=false`，不能为了覆盖率创建无真实 mesh 的伪对象。

### P4 验收

- 每个 `rejected_no_mesh` 能定位到具体 Hydra 漏斗级别；
- entity→mesh 交付覆盖高于 30.83% fast 基线；
- 按 commit object 数计算的绑定占比不低于 80.43%，或下降有明确 precision 收益；
- 人工抽样绑定 precision 不下降；
- tabletop 小物体 mesh recall 提升；
- 全局匹配结果不依赖 entity 处理顺序。

## 10. P5：FastSAM、SAM2 与三维模型 A/B

### P5.1 受控实验组

对同一段 150–300 帧以及完整代表场景依次运行：

1. 当前 FastSAM 基线；
2. P0 修复后的 FastSAM；
3. P1–P4 改进后的 FastSAM；
4. SAM ViT 单帧 AutomaticMaskGenerator；
5. SAM2 单帧 AutomaticMaskGenerator；
6. SAM2 video predictor/memory 传播；
7. 若引入真正的三维模型，单独定义 2D→3D、实时性和 identity 接口后再加入。

### P5.2 固定条件

- 相同 RGB、深度、pose 和帧范围；
- 相同 thing/stuff 规则和 mask 后处理；
- 相同 P1 entity association；
- 相同 DAM 视图与 prompt；
- 相同 Hydra 配置和绑定阈值；
- 记录模型版本、checkpoint hash、engine hash、随机种子和硬件。

### P5.3 决策标准

不以单一 mask 指标选模型。综合比较：

- GT 物体召回；
- mask AP50/AP75、boundary F1、小物体 recall；
- duplicate/nested mask；
- IDF1/HOTA/ID switch；
- entity purity/fragmentation；
- DAM category accuracy；
- entity→mesh binding precision/recall；
- P50/P95 延迟、显存、吞吐和磁盘产物大小。

只有当端到端可用对象数和 identity 质量提升，且运行成本可接受时，才替换默认 FastSAM。

## 11. 标注集与评测工具

### 11.1 最小标注集

首批标注 30–50 个代表帧及其跨帧物体轨迹，覆盖：

- 小物体：杯子、瓶子、键盘、遥控器等；
- 遮挡和重新出现；
- 相邻同类物体；
- 快速相机转动和明显视角变化；
- 反光、无纹理和深度空洞区域；
- 0–2 m tabletop 与 3–5 m room-scale；
- whole-object/part、thing/stuff 和嵌套 mask；
- 至少一段中断恢复测试；
- 至少一段与激光雷达全局地图对齐的场景。

### 11.2 指标定义

| 层级 | 必须报告的指标 |
| --- | --- |
| 分割 | AP50/AP75、boundary F1、小物体 recall、proposal、duplicate/nested ratio |
| 跟踪 | confirmed ratio、ID switch、IDF1/HOTA、轨迹寿命、tracks per GT |
| 深度几何 | depth-valid ratio、中心误差、AABB IoU、背景泄漏、协方差校准 |
| MapMemory | entity purity、fragmentation、同帧冲突、entities per GT |
| DAM | category accuracy、caption usability、part-name error、view consistency |
| Hydra | object recall/precision、mesh 体积误差、binding precision/recall |
| 系统 | P50/P95、GPU/CPU 内存、队列高水位、掉帧、磁盘、恢复一致性 |

### 11.3 评测产物

每次实验输出：

- 机器可读 `evaluation.json`；
- 版本/参数/输入 hash；
- 按帧 overlay 和失败样本索引；
- entity/track/semantic/Hydra ID 对照表；
- 相对于上一基线的 delta；
- 不少于 20 个失败案例的人工分类；
- 自动生成的 Dashboard 对比页。

## 12. 测试策略

### 12.1 单元测试

- RGB/BGR sentinel；
- BotSort 参数传递和严格边界值；
- predict-only 状态机；
- observation 去重；
- semantic ID 容量与 remap；
- MapMemory 全局匹配代价和一对一约束；
- entity split/merge/reassign 幂等性；
- polygon 孔洞/原始 mask 保真；
- DAM 结构化输出解析和重试；
- Hydra 漏斗统计与全局 binding。

### 12.2 集成测试

- 固定 10–30 帧的轻量全链回放；
- 分割间隔内新对象进入和确认；
- 同帧两个相邻同类对象不误合并；
- 同一对象遮挡后恢复原 entity；
- 同一 entity 多 track 不重复 prompt；
- provisional semantic IDs 在 postpass 前全部 canonical；
- 中断/恢复后 identity 不碰撞；
- correction 能且只能绑定到真实 mesh；
- run report、SQLite、commit manifest 和 Dashboard 计数一致。

### 12.3 完整回归

至少使用：

- `g1_260720_1424_indoor_fast_semantic_10cm` 对应场景；
- `g1_20260717_tabletop_offline_q16_20260720_142700` 对应场景；
- 一段专门的小物体 tabletop；
- 一段 lidar-aligned room-scale；
- 一段 resume replay。

每完成一个 P 阶段运行一次完整回归；P0 内部小任务使用轻量片段快速反馈。

## 13. Dashboard 改造要求

Dashboard 每个模块至少展示：

- 当前模块是否初始化/运行/排空/完成；
- 当前有效参数和来源；
- 输入、输出和拒绝计数；
- P50/P95 延迟和队列高水位；
- 最近失败原因与可点击证据；
- track/entity/semantic/Hydra ID 联动查询；
- 当前帧 RGB、深度彩虹图、原始 mask、仲裁 label 和最终对象 overlay；
- 完成后可用左右箭头浏览历史结果；
- 长时间阶段的已处理/总数、ETA 和定期提醒；
- 旧基线与当前运行的 delta。

参数面板按风险分组：

1. 输入/深度范围；
2. FastSAM；
3. BotSort/ReID；
4. MapMemory identity；
5. DAM；
6. Hydra object extractor；
7. entity→mesh binding；
8. 质量门和恢复策略。

任何会改变身份语义或持久化格式的参数都要显示警告，并写入 run manifest。

## 14. 里程碑和建议交付批次

### M0：基线冻结

- 固定两次基线报告与配置 hash；
- 建立最小标注集；
- 输出统一 evaluation schema；
- 冻结当前行为的回归测试。

完成定义：可以在同一输入上重复得到可比较的基线报告。

### M1：P0 正确性版本

- 通道契约；
- tracker 非分割帧策略；
- BotSort 参数与漏斗；
- DAM 去重；
- semantic-ID remap；
- resume epoch/state；
- 全链拒绝原因。

完成定义：P0 所有验收项通过，能准确解释每个 entity 的产生、命名和丢失位置。

### M2：P1 单一 ID 版本

- entity 状态升级；
- 多特征全局匹配；
- split/reassign/rollback；
- identity Dashboard 审计。

完成定义：标注集 identity 指标优于 M1，且无数据库迁移阻塞。

### M3：P2/P3 感知与命名版本

- mask 层级/去重/重叠仲裁；
- 原始 mask 保真；
- 多视角 DAM；
- 结构化类别与词汇规范。

完成定义：分割和命名指标优于 M2，系统延迟在预算内。

### M4：P4 语义 mesh 版本

- Hydra 全漏斗；
- room/tabletop/lidar-aligned 配置；
- 全局 mesh binding；
- rejected 逐级诊断。

完成定义：绑定覆盖提升、人工绑定 precision 不下降、所有 commit 校验通过。

### M5：P5 模型选择版本

- 完整 FastSAM/SAM2/3D A/B；
- 质量、延迟、显存和维护成本决策；
- 更新默认配置与文档。

完成定义：模型选择有真值、端到端和系统成本三方面证据。

## 15. 风险、回退与发布门槛

### 15.1 主要风险

| 风险 | 缓解措施 |
| --- | --- |
| 放宽合并/绑定提高 coverage 但产生误绑定 | 同时报告 precision；全局一对一；人工抽样 |
| 更高分辨率导致实时性下降 | TensorRT engine 分档；P95/显存硬门 |
| SAM2 产生更多嵌套/部件 mask | P2 层级和去重先完成 |
| 多视角 DAM 增加 GPU 时间 | 候选窗口上限、批处理、缓存和早停 |
| MapMemory schema 变化破坏旧任务 | 版本化 migration、备份、只读兼容和回滚 |
| resume 恢复部分状态造成更隐蔽冲突 | schema/hash 完整校验；不完整时强制新 epoch |
| Hydra 阈值过低产生噪声 mesh | 漏斗定位后单参数 A/B；体积与重建置信双门 |
| lidar 与 RGB-D 时间/坐标不一致 | 标定 hash、时间同步报告、独立对齐验收 |

### 15.2 回退策略

- 每项新算法有独立配置开关；
- 保留 legacy tracker tick、legacy nearest-entity 和 legacy overlap 路径供对照；
- MapMemory schema 迁移前自动备份；
- semantic label remap 使用临时目录和原子提升；
- Hydra postpass 失败时不覆盖上一份已验证产物；
- correction 与 DSG 提交继续使用幂等 operation/hash；
- 默认配置只在完整回归通过后更新。

### 15.3 发布硬门

一个里程碑可标记“稳定”前必须满足：

1. 对应单元、集成和完整回归通过；
2. 没有 pending/unmapped/error 或未解释的队列残留；
3. 所有报告、数据库、manifest 和 Dashboard 计数一致；
4. 人工标注集没有 identity/命名/绑定指标回退；
5. P95 延迟和显存不超过预先批准的预算；
6. 中断恢复与旧数据库兼容测试通过；
7. `rejected_no_mesh` 等拒绝均有可审计原因；
8. 文档、配置说明和迁移说明同步更新。

## 16. 首轮建议任务清单

按以下顺序开始，避免并行修改过多变量：

- [ ] 冻结 M0 输入、报告和配置 hash。
- [ ] 建立红/蓝 sentinel，确认所有图像边界的通道契约。
- [ ] 为非分割帧增加“不更新 tracker 生命周期”的最小修复。
- [ ] 把 BotSort 隐式阈值写入配置、run manifest 和 Dashboard。
- [ ] 增加 FastSAM→BotSort→depth→MapMemory 前端漏斗。
- [ ] DAM observation 按唯一帧计数，并按 entity/revision 去重。
- [ ] 建立 semantic-ID remap journal 和 9999 容量保护。
- [ ] resume 使用新 session epoch，随后设计完整状态恢复。
- [ ] 补齐 Hydra candidate→mesh→binding 漏斗。
- [ ] 标注首批 30–50 帧和跨帧 GT identity。
- [ ] 运行 P0 FastSAM A/B，形成 M1 评审报告。

完成首轮任务后，再决定 P1 全局关联和 P2 mask 层级的具体实现拆分。

## 17. 文档维护

- 架构、原理和当前问题：[`Intro.md`](Intro.md)
- 开发优先级、验收和里程碑：本文 `P.md`
- 操作步骤、CLI、完整参数：[`Pipeline.md`](Pipeline.md)
- Dashboard 使用说明：[`DASHBOARD.md`](DASHBOARD.md)
- 每次运行的事实证据：对应 `output/<run>/realtime_run_report.json`、
  `quality_report.json`、`semantic_dsg_commit.json` 和 `run_manifest.json`

当实现或默认参数发生变化时，先更新代码和测试，再更新 `Pipeline.md` 的操作事实；只有架构
结论或优化优先级变化时才更新 `Intro.md`/`P.md`，避免多份文档长期漂移。
