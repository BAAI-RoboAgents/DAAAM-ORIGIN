# DAAAM 语义地图中的物体感知、单一 ID 与语义落图

本文说明 DAAAM 如何把逐帧 RGB、深度和相机位姿转换成带开放词汇名称的三维对象，重点
解释物体候选分割、跨帧跟踪、三维身份合并、DAM-3B 命名、Hydra 对象重建，以及这些
模块为什么会产生漏识别、误识别、重复 ID 或 `rejected_no_mesh`。

文中的“当前实现”主要指 Dashboard 调用的准实时主链
[`scripts/run_realtime_mapping.py`](scripts/run_realtime_mapping.py) 与
[`RealtimeSemanticAdapter`](src/daaam/realtime/semantic.py)。旧式离线 orchestrator、CODa
复现链和不同 Hydra 配置会有参数差异。操作命令、完整阶段定义及全部配置项以
[`Pipeline.md`](Pipeline.md) 为准；本文负责架构、原理、身份语义、诊断和优化决策。

## 1. 先说结论：物体“识别”是一条多级漏斗

项目不是用一个模型直接完成“看见物体并写进地图”，而是依次回答六个不同问题：

1. **哪里可能是一个物体？** FastSAM/SAM 产生二维、类别无关的 mask。
2. **相邻帧里是否还是同一个目标？** BoT-SORT 用运动、重叠与 ReID 维持临时 `track_id`。
3. **这个二维目标在三维哪里？** 深度、内参和位姿把 mask 像素反投影到世界坐标系。
4. **它是否与历史对象相同？** MapMemory 把局部轨迹合并为权威 `entity_id`。
5. **它是什么？** DAM-3B 根据 RGB 与 focal mask 生成开放词汇描述。
6. **它是否真的成为地图对象？** Hydra 必须重建出可用 object mesh，随后才能把
   `entity_id`、`semantic_id`、自然语言名称和 DSG 对象节点可靠绑定。

因此：

- “FastSAM 有 mask”不等于“识别正确”；
- “DAM 已命名”不等于“最终地图中存在该对象”；
- “质量门 PASS”不等于有标注真值上的语义准确率高；
- 直接把 FastSAM 换成更大的 SAM，只会改变漏斗的第一层，不能自动修复 ID 关联、深度
  几何、DAM 命名或 Hydra mesh 形成问题。

从当前两次实跑报告看，分割和 DAM 任务都能运行并排空，**最大可量化损失发生在 Hydra
object mesh 形成以及 MapMemory entity 到真实 mesh 的绑定**。但这不代表二维前端已经
准确：当前没有人工标注真值，RGB/BGR 接口、空检测 tracker tick、重叠 mask、实体合并和
重复 prompt 等问题仍会污染下游。

## 2. 总体架构与真实执行时序

### 2.1 逻辑数据流

```text
RGB ──> FastSAM ──> detections + masks
RGB + detections ──> BoT-SORT + ReID ──> track_id + mask_idx ──> tracked mask
tracked mask + depth + intrinsics + world pose ──> 三维反投影
                                                    │ position / AABB
                                                    v
                                              MapMemory.observe
                                                    │ entity_id
                               ┌────────────────────┴───────────────────┐
                               v                                        v
                      semantic_id 标签图                         DAM-3B 描述
                               │                                        │ correction
                               v                                        v
               RGB + depth + pose + exact labels                MapMemory 版本化更新
                               │                                        │
                               v                                        │
                      Hydra/Khronos postpass                            │
                               │ object mesh / DSG node                  │
                               └──────────────────┬──────────────────────┘
                                                  v
                                     entity ↔ semantic_id ↔ O(n)
                                                  │
                                                  v
                                    持久化 DSG、mesh 与提交清单
```

`sentence-t5-xl` 是可选的文本语义后处理；Hydra RoomFinder 属于空间场景图的房间层构建。
两者都独立于本文讨论的物体命名与单一 ID 主链，也不是每个准实时任务都必须经过的环节。

### 2.2 `semantic_mode=dam` 准实时任务的真实顺序

1. 初始化深度后端、FastSAM、BoT-SORT/ReID、DAM worker、MapMemory 和标签空间。
2. 在线遍历帧：几何主链和语义旁路同时工作；语义 label image 按帧持久化。
3. 在线阶段结束后，用完整的 RGB、静态深度、位姿和精确标签图执行 Hydra
   `exact-label postpass`。该结果会替代临时 live geometry 输出。
4. 排空 DAM 队列，把 correction 版本化写入 MapMemory。
5. 将 correction 绑定到真实 Hydra object mesh，持久化并重载校验最终 DSG 和 hash。

这解释了 Dashboard 中“运行已到 90% 以上但仍未结束”的常见现象：最后的 postpass、DAM
排空、mesh 绑定和提交校验都可能耗时，而且属于硬质量门的一部分。

## 3. 模块职责与数据契约

| 环节 | 当前主要实现 | 输入 | 输出与职责 |
| --- | --- | --- | --- |
| 深度估计 | FoundationStereo 或预计算深度 | 左右图/已有深度 | 米制深度、可选置信度；决定 mask 是否能形成三维观测 |
| 二维实例候选 | FastSAM-X TensorRT | 单帧图像 | `bbox + score + bool mask`；类别统一为 0，不负责命名 |
| 临时跨帧跟踪 | BoT-SORT + CLIP ReID | detections、图像 | 进程内 `track_id`，不是全局对象 ID |
| mask 传播 | `RealtimeSemanticAdapter` | 历史 mask、tracker bbox | 非分割帧的临时标签覆盖 |
| 三维观测 | `masked_geometry` | mask、深度、内参、位姿 | 世界坐标中心、轴对齐尺寸和几何置信度 |
| 长期实体记忆 | MapMemory | 局部轨迹、三维几何、临时标签 | 权威 `entity_id`、别名、版本和 correction 状态 |
| 开放词汇命名 | NVIDIA DAM-3B | RGB、focal mask、prompt | 自然语言对象描述；不负责跨帧 ID |
| 整数标签桥接 | 10000 个伪标签槽位 | entity/track 映射 | `semantic_id` 标签图、占位名和稳定可视化颜色 |
| 三维地图 | Hydra + Khronos | RGB-D、位姿、label image | TSDF/mesh、object mesh、Dynamic Scene Graph |
| 最终绑定 | SceneGraphService / DSG sink | entity 几何、semantic ID、Hydra 对象 | 一对一 `entity_id ↔ O(n)` 绑定和持久化校验 |
| 可选语义后处理 | sentence-t5-xl、RoomFinder | 描述和图结构 | 文本嵌入、语义相似度和房间层 |

### 3.1 最重要的输入输出契约

二维前端约定：

```text
image: H × W × 3 uint8
detections: N × 6 [x1, y1, x2, y2, score, class_id]
masks: N 个 H × W bool 数组
tracks: M × 8 [x1, y1, x2, y2, track_id, score, class_id, mask_idx]
```

`mask_idx` 把 tracker 输出重新连回原始 mask。任何通道顺序、坐标缩放、mask 尺寸、深度
单位或位姿方向的契约错误，都会同时影响分割、ReID、三维中心和最终 mesh 绑定。

## 4. ID 层级：什么才是“同一个物体”

项目中同时存在多种 ID。它们用途不同，不能互相替代：

| ID | 产生位置 | 生命周期 | 含义 |
| --- | --- | --- | --- |
| `mask_idx` | 单次分割调用 | 单帧 | detection 与 bool mask 的数组下标 |
| BoxMot 原生 ID | BoT-SORT | tracker 实例内 | 跟踪器内部轨迹编号 |
| `track_id` | TrackingService | 当前进程/跟踪器 | 供语义 adapter 使用的临时轨迹编号 |
| `local_entity_id` | 语义 adapter | 当前 session | 形式为 `botsort:<track_id>` 的局部观测源 |
| `entity_id` | MapMemory | 单个 MapMemory 数据库 | `entity-<uuid>`；当前设计中的权威逻辑对象 ID |
| `semantic_id` | 语义 adapter | 主要为单次运行 | 标签图中的正整数代理；0 是背景，不是物体类别号 |
| Hydra `O(n)` | Hydra DSG | 当前 Hydra 图 | Hydra 独立创建的对象节点 ID |
| `operation_id` | correction 流程 | 可持久化 | correction 的 SHA-256 幂等键，不是对象 ID |

TrackingService 为预留 0 会把 BoxMot 返回的编号再加 1；固定 BoxMot 版本自身已经从 1 起号，
所以当前只是产生一个无语义影响的编号偏移。它不会直接降低精度，但调试和跨产物核对时必须
使用适配器输出的 `track_id`，不能把内部编号混用。

### 4.1 单一 ID 绑定的目标

理想的不变量是：

```text
一个真实物体
    ↔ 一个 MapMemory entity_id
    ↔ 一个 semantic_id
    ↔ 一个有真实 mesh 的 Hydra O(n)
```

实际系统不能从第一帧就保证这一点。FastSAM mask 可能拆成多个部件，BoT-SORT 可能发生
ID switch，同一实体也可能被多个局部 track 反复发现。因此 `track_id` 只是短期线索，
`entity_id` 才承担合并后的身份；`semantic_id` 负责把身份写进像素标签图；Hydra `O(n)`
则是另一套几何对象编号，必须在结束阶段显式绑定。

### 4.2 身份稳定性的边界

- 同一个正在运行的 MapMemory 数据库中，`entity_id` 可持久化并接受版本化 correction。
- Dashboard 的不同任务默认各自创建 MapMemory 数据库，因此不会天然共享全局物体 ID。
- 当前 checkpoint 主要保存帧进度、动态/子图状态和路径，不完整保存 BotSort 状态、
  track→entity 映射、entity→semantic 映射、mask 历史和下一个 semantic ID。
- 因此“恢复已有运行”不是严格的 tracker 状态续接。若要支持跨天、不同起点的增量地图，
  除了所有 pose 位于同一世界坐标系，还需要加载既有 MapMemory/DSG/TSDF，恢复或隔离
  session/semantic ID，执行全局重定位、数据关联、冲突消解与可回滚更新。

仅仅提供同一坐标系 pose 能解决几何对齐的一部分，但不能单独保证同一物体仍使用同一 ID。

## 5. FastSAM：类别无关的候选实例分割

### 5.1 作用、原理与输出

FastSAM 在每个分割帧中产生若干候选 bbox 和像素级 mask。它的职责是回答“哪些像素
可能属于同一个区域”，不是回答“该区域是什么”。当前适配器把所有候选的 `class_id`
固定为 0，真正的名称来自后续 DAM-3B。

当前准实时默认参数来自
[`pipeline_config_realtime.yaml`](config/pipeline_config_realtime.yaml) 和
[`fastsam_config.yaml`](config/fastsam/fastsam_config.yaml)：

| 参数 | 准实时默认 | 含义与增减影响 |
| --- | ---: | --- |
| TensorRT 输入 | `480 × 640` | 越大通常越利于小物体和边界，但延迟、显存增加；更改后可能要重建 engine |
| `fastsam_conf` | `0.30` | 降低可提高候选召回，也会增加背景、局部和重复 mask |
| `fastsam_iou` | `0.50` | NMS 重叠阈值；提高它会允许更多重叠候选保留，并非过滤更严格 |
| `retina_masks` | `true` | 尽量把 mask 恢复到原图分辨率 |
| `min_mask_region_area` | `300 px` | 小于此像素面积的候选在推理后被过滤 |

桌面配置使用更低的 `conf=0.25`、更高的 `iou=0.60` 和更小的最小面积 `150 px`，目的是
提高小物体召回，但也更容易产生物体整体、部件和相邻区域的重叠候选。

### 5.2 当前限制

1. **没有 thing/stuff 区分。** 地面、墙、桌面和可移动物体可能进入同一候选池。
2. **缺少 whole/part 层级与 mask 去重。** 一个杯子、杯把和杯口可能被当成多个实例。
3. **`max_mask_region_area` 当前虽在配置模型中声明，但没有进入实际过滤。** 超大背景 mask
   不会因此被排除。
4. **重叠区不是按置信度或遮挡关系决策。** 当前标签合成按 `track_id` 排序后 first-wins，
   先写入的 mask 抢占像素。
5. **颜色通道接口存在待验证风险。** 数据加载层输出 RGB，而固定 Ultralytics/BoxMot 的
   ndarray 路径按 BGR 约定转换。除非 TensorRT engine 特意按对应通道导出，否则存在
   R/B 再次换序的可能。应通过红蓝色块 sentinel 和同帧 A/B 验证，不能仅凭肉眼下结论。

代码入口为 [`segmentation.py`](src/daaam/utils/segmentation.py)；服务封装见
[`segmentation/services.py`](src/daaam/segmentation/services.py)。

### 5.3 是否应该更换 SAM

替换模型可能提升二维边界或小物体召回，但应区分三种方案：

- 当前仓库的 SAM ViT 是**逐帧自动 mask 生成器**；
- 当前仓库的 SAM2 也是逐帧 `AutomaticMaskGenerator.generate(frame)`，没有使用 video
  predictor、跨帧 memory 或 prompt propagation；
- “SAM3D”不是本项目现有的即插即用 backend。不同名称可能指 2D mask 的 3D 提升、
  三维点云分割或独立的 3D 对象模型，必须额外定义输入输出、实时性和 ID 接口。

所以，直接把 checkpoint 名称改为 SAM/SAM2 不会自动获得视频级单一 ID。更稳妥的长期
结构是：**关键帧发现新对象 + SAM2 video 传播像素 mask + BoT-SORT/ReID 做身份仲裁 +
RGB-D/MapMemory 做三维确认**。在模型 A/B 前，应先修复通道契约、tracker tick 和 mask
去重，否则更密的 SAM mask 可能只会产生更多碎片与重复实体。

## 6. BoT-SORT + CLIP ReID：临时跨帧身份

### 6.1 核心原理

BoT-SORT 的当前流程包括：

1. Kalman filter 预测 bbox 运动；
2. ECC 估计相机全局运动并修正预测；
3. 高置信候选使用 IoU 与 ReID 外观距离进行第一轮 Hungarian 匹配；
4. 低置信候选只用第二轮 IoU 尝试挽回已有轨迹；
5. 未匹配候选超过新轨迹阈值才创建 track；
6. 轨迹在若干 tracker tick 内未重新关联后进入 lost/removed 生命周期。

固定 BoxMot 版本的有效默认值为：

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `track_high_thresh` | `0.50` | 第一轮关联候选下界 |
| `track_low_thresh` | `0.10` | 低于该值直接忽略 |
| `new_track_thresh` | `0.60` | 新建轨迹所需最低置信度 |
| `match_thresh` | `0.80` | 第一轮匹配距离门 |
| `proximity_thresh` | `0.50` | 几何距离过大时禁止仅靠 ReID 挽救 |
| `appearance_thresh` | `0.25` | ReID 外观距离门 |
| `track_buffer` | `30` | lost 轨迹保留的 tracker tick 数 |
| `frame_rate` | `30` | 用于折算 buffer；项目当前未覆盖 |
| `with_reid` | `true` | 启用 CLIP ReID |
| `cmc_method` | `ecc` | 相机运动补偿方法 |

项目目前只显式传入其中一部分；high/low/new/match/proximity/appearance 尚未完整暴露到
Dashboard。实现见 [`tracking/services.py`](src/daaam/tracking/services.py)。

### 6.2 FastSAM 和跟踪阈值必须联合调节

FastSAM `conf=0.25/0.30` 不表示所有候选都能成为新物体。按照当前 BotSort 默认值：

- `0.10 < score < 0.50`：只能参与低分第二轮，帮助已有轨迹恢复；
- `0.50 < score < 0.60`：能参与第一轮，但不能新建轨迹；当前严格比较下恰好 0.50
  不属于两组；
- `score ≥ 0.60`：才可能创建新轨迹。

因此只降低 FastSAM 阈值，常见结果是计算量和重复候选增加，而新物体召回没有同比提升。
Dashboard 应同时显示 `FastSAM proposal → >0.5 → ≥0.6 → confirmed track` 漏斗。

ReID 也不是类别识别器：它对 bbox crop 计算外观 embedding，用来比较“看起来是否相似”。
背景占比大、同类物体外观接近、遮挡或视角变化都可能导致误关联/漏关联。又因为所有 FastSAM
候选的 class 都是 0 且 `per_class=false`，当前所有 mask 位于同一个关联池，类别不能帮助排除
不合理匹配。

### 6.3 非分割帧与 mask 传播

准实时配置按较低频率执行分割，但当前每个语义帧仍调用 tracker；非分割帧传入空
detections。固定 BoxMot 版本在这种情况下不会返回 lost track 的预测 bbox，所以当前实跑
出现了两个现象：

- `propagation_bbox_warps = 0`；
- 所有传播实例都走原位置 `carry_forward`。

这也是轨迹确认的风险：新建但未确认的 track 可能在下一次空检测 tick 中丢失。推荐的最小
修复是只在真实分割帧调用完整 `BotSort.update()`；更完整的方案是增加只做 Kalman/ECC、
不改变轨迹生命周期的 predict-only 路径。

当前传播最多保留 2 帧、256 个 track，历史/audit 缓存为 32 帧。它不使用光流、深度、
相机 pose 或遮挡关系；没有 tracker bbox 时会直接复制旧 mask。因此相机移动、遮挡、形变
或深度边界变化时，传播标签可能留在错误位置。

### 6.4 重叠标签的 first-wins 行为

当前 tracker rows 先按 `track_id` 排序，再执行：

```python
label_image[mask & (label_image == 0)] = semantic_id
```

所以较小 track ID 优先占据重叠像素，决策没有使用 mask score、深度有效率、时间稳定度或
前后遮挡。即使某个先写 mask 随后无法形成三维观测，它仍可能占住该帧标签像素。建议改为
显式评分竞争，并对 stuff、whole-object 和 part mask 分层处理。

## 7. 从二维 mask 到三维 RGB-D 观测

### 7.1 当前几何计算

对每个 mask，准实时 adapter 执行：

1. 有效深度定义为 `finite && depth > 0`；
2. 有效深度像素至少占 mask 的 25%，否则不创建三维观测；
3. 最多抽取 20,000 个像素；
4. 使用针孔模型反投影：

```text
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
Z = depth
```

5. 用 `world_T_camera` 把点变换到世界坐标系；
6. 世界点中位数作为对象位置；各轴 5%–95% 分位范围作为 AABB 尺寸；
7. 每个尺寸至少取 5 cm，避免退化盒。

实现见 [`masked_geometry.py`](src/daaam/realtime/masked_geometry.py)。

### 7.2 深度为什么影响识别与单一 ID

深度本身不决定名称，但决定一个二维 mask 能否成为三维实体、能否与历史实体合并，以及
Hydra 是否有足够表面形成 object mesh。深度图中的空洞、截断、背景泄漏或错误尺度会导致：

- 有效比例不足，mask 在 MapMemory 之前被丢弃；
- 背景深度把对象中心拉远，错误合并到其他实体；
- AABB 被放大，多个邻近物体误合并；
- 几何支持不足，最终出现 `rejected_no_mesh`。

深度 PNG 肉眼看起来大部分为黑色不一定表示数据坏了：定量深度通常以带 scale 的 uint16
数值保存，普通图片查看器并不会按实际有效距离自适应着色。应检查原始数值、单位/scale、
0 值比例和 `depth_metadata`，而不是按灰度亮度判断。Dashboard 的彩虹图中空白区域表示
无效或超出显示/截断范围；把深度上限改为 5 m 会增加 3–5 m 范围的几何支持，但不会补回
立体模型本身没有估计出的深度。

当前实时语义反投影只检查正深度，没有直接使用 `DepthFrame.confidence`，也没有复用 YAML
中的 `depth_lb/depth_ub` 做第二次范围门控。25% 门槛还允许大量无效像素，少量背景泄漏
可能主导中心与尺寸。推荐引入置信度、左右一致性、深度连通分量和 median/MAD 或 DBSCAN
离群点剔除，并记录中心协方差供后续数据关联使用。

## 8. MapMemory：从局部轨迹到权威 entity_id

### 8.1 当前关联规则

`MapMemory.observe()` 的核心逻辑是：

1. 若 `(session_id, local_entity_id)` 已有映射，直接复用原 `entity_id`，不再做空间复核；
2. 若是新局部轨迹，只在 `entity_type` 相同、输入标签能匹配 canonical label/alias、三维中心
   距离不超过 `entity_merge_distance_m` 的历史实体中选最近者；
3. 没有候选时创建新的 `entity-<uuid>`；
4. 记录局部轨迹到实体的永久映射，后续观测更新位置、置信度和计数。

在准实时链中，新实体初始名称通常都是 `unknown`。DAM 重命名时旧的 `unknown` 又会保留为
alias，因此标签条件的区分力很弱，匹配事实上接近“同类型实体中的最近中心”。当前算法不
使用 ReID/CLIP、AABB IoU、尺寸、时间互斥、运动状态或全局一对一分配。

### 8.2 合并距离的双向风险

- 距离过小：同一真实物体的不同 track 无法合并，出现 entity 碎片化和多个名字；
- 距离过大：桌面上相邻物体或同一大 mask 内的部件被误合并，名称相互覆盖。

例如近期 `fast_semantic_10cm` 运行采用 0.5 m 合并尺度，进入 MapMemory 的 503 个唯一
局部 track 合并成 314 个实体；旧 q16 运行使用 0.075 m，进入 MapMemory 的 760 个唯一
局部 track 只合并成 730 个实体。这里不包括从未通过深度门、因而没有进入 MapMemory 的
tracker IDs。前者更容易过合并，后者明显更容易碎片化；不能只看实体数量判断哪一个正确。

### 8.3 当前身份污染点

- 一个局部 track 一旦写入 session 映射，就不会因后续几何冲突自动重分配；若发生 ID
  switch，两个真实物体的观测会长期污染同一个 entity。
- dimensions 只在创建时写入，后续位置会更新但尺寸不会形成稳健累计估计。
- 多个同帧 track 可能依次合并到同一 entity，每个都增加 observation count；阈值 5 可能
  在单帧内被多个碎片提前满足。
- 历史 label PNG 在写出后不会因后续实体合并而统一重写，最终 sidecar 中可能同时保留
  同一实体的临时 semantic IDs。

长期改进应使用同帧全局 Hungarian 一对一匹配，代价至少组合三维中心马氏距离、AABB
IoU/gap、外观 cosine、尺寸一致性和时间连续性；同时加入同帧互斥、协方差、稳健几何更新，
并允许受审计的 split/reassign。

## 9. DAM-3B：开放词汇命名，不是身份跟踪器

### 9.1 触发、输入与输出

当前准实时 adapter 只把满足 `semantic_minimum_observations` 的实体送给 DAM，默认阈值为
5。有效计数主要来自真实分割且深度有效的观测；传播帧不作为新的 DAM 真实观测。任务输入
是完整 RGB 与 focal mask，默认提示词为：

```text
Describe what you see in this region.
```

这里的“观察次数”是观测条目数，并非去重后的 `sensor_time` 数；同一帧中多个 track 若合并到
同一 entity，可能一次增加多次。Dashboard 准实时链的实际门槛是 CLI
`--semantic-minimum-observations`。YAML 中的 `query_interval_frames`、离线 AssignmentService
的 `min_frames_max_size`、`N_masks` 和代表帧打分权重，并不直接控制这条实时提交路径。

送给 DAM 的 focal mask 也不是无损的原始 bool mask：当前会用 `RETR_EXTERNAL` 提取外轮廓、
Douglas–Peucker 简化，再用 polygon 填充重建。内部孔洞会消失，杯把、栏杆、线缆等细结构
可能变粗或被抹掉，这会改变模型看到的目标区域。

DAM 返回自由文本描述，例如 `red office chair`。它不是闭集分类器，也不生成或维护
`entity_id`。自然语言 description、规范 category 和物体 identity 是三个不同层次：

- description 可以很详细，但措辞可能随视角变化；
- category 应是经过规范化的稳定类别名；
- identity 表示物理世界中的同一个实例，必须由跟踪与三维关联维护。

当前 correction 主要把自由文本作为对象名称写回 MapMemory，因此同一对象不同视角可能出现
近义词、部件名或错误名称。DAM annotation 的原始 confidence 为 0，实时侧会回退到默认
`automatic_confidence=0.5`；它不是经过标定的分类概率，不能直接用作识别准确率。

### 9.2 多图 worker 不等于多视角识别

`dam_multi_image` 会把多个任务放入一个吞吐批次，但批次中的图可以来自不同 entity；这不
等同于对同一对象融合多视角证据。当前实体一旦达到观察阈值，通常立即提交当时帧；候选质量
排序主要用于尚未提交时的替换，并非从完整历史中选最优代表视角。

同一批次也没有按 `entity_id` 强制去重。若多个 track 在同帧合并到同一个 entity，可能产生
多个 prompt 和 correction。近期运行中 240 个被提交实体收到了 293 个 correction，说明
重复或版本替换真实存在。

PE-Core-L14-336/CLIP 特征是特定配置中的代表帧视觉特征。当前准实时 correction 转换没有把
该特征用于 MapMemory 或 Hydra 的身份关联，所以它暂时不能阻止 ID switch。合理方向是：

- 为每个 entity 保留 2–3 个清晰、尺度适中、角度互补的候选视图；
- 用 mask-aware crop，减少完整画面中的背景干扰；
- 让 DAM 输出结构化 `category/caption/attributes/is_whole/confidence`；
- 对 category 做词汇规范化，多视角投票后再更新 canonical label；
- 对 `unknown` 或低置信结果允许重试；
- 持久化 CLIP/ReID 特征并用于实体关联，而不只是计算后丢弃。

本次主流程使用本地 `nvidia/DAM-3B` 的 `dam_multi_image` worker。某些配置仍保留
`gpt-4.1`/`agent_model_name` 字段，但该 worker 不使用这些字段，不能据此认为流程调用了
OpenAI 在线模型。

## 10. “10000 类标签空间”是什么意思

日志中的“10000 类标签空间”实际指**容量为 10000 个伪语义 ID 的标签空间**，不是
ImageNet 那样预先训练好的 10000 个类别。若写成“1000 类”则是数字笔误；当前配置的
准确值是 `10000`。

Hydra 在处理第一帧前必须知道语义标签的整数范围，因为实例/语义标签图中的每个像素
传给 Hydra 的是一个整数 ID，而不是一段自然语言。当前
[`labels_pseudo.yaml`](config/labels_pseudo.yaml) 因此预先声明：

- `total_semantic_labels: 10000`；
- `object_labels` 为 `0..9999`；
- 初始名称为 `label_0`、`label_1`、……、`label_9999`；
- `dynamic_labels` 和 `invalid_labels` 初始为空。

配置文件为了声明完整范围把 0 也列在 `object_labels` 中，但准实时标签图实际把 0 当作
背景，运行时实例使用的是正整数 `1..9999`。

配套的 [`labels_pseudo.csv`](config/labels_pseudo.csv) 为每个 ID 提供稳定的 RGBA
可视化颜色。它们共同形成一个可供运行时占用的 ID 池，工作过程如下：

```text
FastSAM mask / BotSort track
            ↓
分配或复用稳定 semantic_id
            ↓
Hydra 标签图和 DSG 暂时使用 label_<id>
            ↓
DAM-3B 生成开放词汇描述
            ↓
把该 ID 的名称更新为真实描述，并同步对象层与 mesh labelspace
```

例如某个实体被分配 `semantic_id=42` 时，初始化阶段只知道占位名称 `label_42`；如果
DAM 后续给出 `red office chair`，SceneGraphService 会把 ID 42 对应的 labelspace 名称
更新为该描述。在同一实体完成规范化映射后，该整数 ID 用来连接逐帧 mask、Hydra 对象、
mesh、语义 correction 和最终 DSG；变化的是它对应的自然语言名称与元数据。

因此，“10000 个标签已加载”只表示以下初始化条件满足：标签 ID 范围、占位名称和颜色表
已经就绪，Hydra 可以安全接收语义标签图。它**不表示**：

- 系统内置了 10000 个真实物体类别；
- DAM 已经识别出 10000 种物体；
- 当前场景中存在或成功构建了 10000 个对象；
- 这些标签已经通过几何绑定或语义质量验收。

实际识别了多少对象，应查看最终 DSG 对象节点、DAM correction、mesh 绑定和
`rejected_no_mesh` 等运行报告，而不能从标签空间容量推断。10000 是当前 labelspace 的条目
数；由于 0 被实时链保留为背景，可直接使用的正实例 ID 是 9999 个。若运行需要超过该范围，
必须同时扩展 YAML、颜色表和语义 ID 管理逻辑。

还要注意两个边界：

- 实时 adapter 的 `semantic_id` 从 1 开始分配，0 留作背景；当前计数器没有在 9999 处主动
  报错，超长运行需要增加容量保护。
- track 在绑定 MapMemory entity 之前可能先获得临时 semantic ID；多个 track 后来合并到
  同一 entity 时会在内存中规范化为同一个 ID，但已经持久化的历史 PNG 不会自动全部改写。
  因此报告中的“历史唯一非零标签数”可能大于最终 MapMemory entity 数。最终 postpass 前
  最好生成 canonical remap journal 并重写或流式重映射历史 label frames。

所以更准确的初始化状态表述是：

> 在 `semantic_mode=dam` 下，Hydra 与语义前端已成功初始化：FastSAM、BotSort/ReID、
> DAM-3B，以及容量为 10000 个伪语义 ID 的开放词汇标签空间均已加载；首帧前就绪门槛
> 已满足。

## 11. Hydra/Khronos：从标签像素到 object mesh 和 DSG

### 11.1 exact-label postpass 的作用

准实时阶段把每帧 label image 写入 `semantic_sidecar/label_frames`。结束后，系统把 RGB、
静态深度、位姿和这些标签逐帧重放给 Hydra，以避免 live 几何链与异步语义链在时间上错位。
报告必须满足 `frames_replayed == frames_expected`、标签配置 hash 一致且没有缺帧，postpass
结果才会被提升为正式 `hydra_realtime` 产物。

Hydra 看到的仍是整数标签，不知道 `red office chair` 这段文字。它依据相同 label 的空间
支持、观察次数和体积建立 object track，再由 MeshObjectExtractor 独立重建对象 mesh。
DAM 名称通过后续 correction 绑定到该真实 mesh，而不是凭空创建一个带名字的 DSG 节点。

### 11.2 影响对象形成的关键参数

当前 room-scale 10 cm 配置
[`hydra_g1_fast_5m_10cm.yaml`](config/hydra_g1_fast_5m_10cm.yaml) 与 tabletop 配置
[`hydra_g1_tabletop.yaml`](config/hydra_g1_tabletop.yaml) 的重点差异如下：

| 参数 | room-scale 10 cm | tabletop | 作用 |
| --- | ---: | ---: | --- |
| 背景 `voxel_size` | `0.10 m` | `0.05 m` | 越小细节越多，但内存和计算量显著增加 |
| `truncation_distance` | `0.30 m` | `0.15 m` | TSDF 表面融合带宽 |
| object `max_range` | `5.0 m` | `2.0 m` | 超出范围的 label 不参与对象检测 |
| `min_cluster_size` | `20 px` | `20 px` | InstanceForwarding 的二维标签像素门槛 |
| `min_num_observations` | `8` | `4` | ExternalTracker 的支持观测尺度 |
| `min_object_volume` | `0.005 m³` | `0.0001 m³` | 小于阈值的对象不分配 mesh |
| `only_extract_reconstructed_objects` | `true` | `true` | 没有足够重建支持的对象不输出 mesh |
| `object_reconstruction_resolution` | `-0.02` | `-0.02` | 负值表示最大物体尺度的 2%，不是固定 2 cm |
| `min_reconstruction_resolution` | 未显式设置 | `0.003 m` | tabletop 对很小对象设置 3 mm 下限 |

ExternalTracker confidence 按观察次数逐步增加，而 extractor 要求 allocation confidence
严格大于 0.5。按当前实现，`min_num_observations=8` 通常要到第 9 次支持才越过分配门；
tabletop 的值 4 通常到第 5 次支持才越过门。只有 label 出现过但观察次数、体积或重建置信度
不足时，不会得到可绑定的 object mesh。

### 11.3 entity 到真实 mesh 的绑定

最终 SceneGraphService 只接受有真实 mesh 的 Hydra 对象候选，并综合：

- semantic ID 是否一致；
- 三维中心距离；
- AABB gap；
- AABB IoU；
- mesh 与 entity 是否已被其他对象预留。

候选在中心距离或 AABB gap 至少一项通过门槛后进入排序，排序优先考虑 semantic 匹配、
AABB gap/IoU、中心距离与稳定 node ID。sink 同时维护“一个 entity 只能占一个 semantic ID”
和“一个真实 mesh 只能绑定一个 entity”的约束。

`rejected_no_mesh` 的准确含义是：该 DAM/MapMemory 实体在最终候选中没有获得可接受且未冲突
的真实 object mesh。它可能由以下任一上游问题导致：

- label 观测次数或像素支持不足；
- 深度无效、截断或表面重建不足；
- mask 太碎、太大、包含多个深度层或发生传播漂移；
- Hydra 对象体积/置信度门槛过严；
- 多个 entity 竞争同一个 mesh；
- MapMemory 中心/AABB 与 Hydra 对象几何不一致。

因此不能把所有 `rejected_no_mesh` 都归因于 DAM 命名错误，也不能简单降低一个阈值。应增加
`label candidate → Hydra track → allocated → reconstructed → mesh → bound` 漏斗遥测后再调参。

## 12. 真实运行漏斗与当前瓶颈

以下数字来自两个已经完成的工程运行。它们衡量**管线转化率**，没有人工真值，不能当作
mask recall、分类准确率或 IDF1。

| 指标 | `fast_semantic_10cm` | 旧 `tabletop_offline_q16` | 解读 |
| --- | ---: | ---: | --- |
| 完成帧 | 1,575 | 1,098 | 均为完整运行 |
| FastSAM 调用 | 679 | 585 | 两次运行的分割失败/空结果均为 0 |
| FastSAM detections | 33,344 | 32,280 | 每次约 49/55 个候选，候选很多不代表正确 |
| depth-valid tracked instances | 16,177 | 14,994 | 进入三维观测的逐帧实例数 |
| 进入 MapMemory 的唯一局部 tracks | 503 | 760 | 不含未通过深度门的 tracker IDs |
| MapMemory entities | 314 | 730 | 新运行合并 189 个；旧运行只合并 30 个 |
| 送 DAM 的 entities | 240 | 480 | 达到真实观测门的实体 |
| corrections | 293 | 491 | 大于实体数，说明存在重复/版本替换 |
| 历史唯一非零标签 | 360 | 777 | 可高于最终 entity 数，因为历史 PNG 未完全规范化 |
| Hydra 最终 object 节点 | 92 | 51 | commit 汇总的 `object_count`，不等于逐节点 mesh 认证数 |
| 已验证 mesh-bound entities | 74 | 22 | 最终带语义且有真实 mesh 的实体 |
| entity→mesh 覆盖 | 30.83% | 4.58% | `74/240` 与 `22/480` |
| 按 commit object 数计算的绑定占比 | 80.43% | 43.14% | `74/92` 与 `22/51`，是汇总比率 |
| 未绑定的已提交唯一 entities | 166 | 约 458 | `240-74` 与 `480-22`；后者为算术推导 |
| `rejected_no_mesh` operations | 183 | 464 | correction operation 数，不是唯一实体数 |

新运行的其他证据：

- 896 个传播帧、11,316 个传播实例中，bbox warp 为 0，全部是 carry-forward；
- 累计记录 63,562,389 个 mask 重叠像素，说明 first-wins 不是罕见边缘情况；
- 分割延迟 P50/P95 为 58.48/81.60 ms，跟踪延迟 P50/P95 为 5.51/31.70 ms；
- 1,575 帧 exact-label postpass 全部完成，1,430 帧含非零标签；
- 314 个映射实体中，最终 74 个实体绑定真实 mesh，82 个 correction operation 应用到 DSG，
  183 个 operation 因无可用 mesh 被拒绝，pending 和 unmapped 均为 0。

相关证据：

- [`fast_semantic_10cm/realtime_run_report.json`](output/g1_260720_1424_indoor_fast_semantic_10cm/realtime_run_report.json)
- [`fast_semantic_10cm/foundation_vs_fast_semantic.json`](output/g1_260720_1424_indoor_fast_semantic_10cm/foundation_vs_fast_semantic.json)
- [`tabletop q16/realtime_run_report.json`](output/g1_20260717_tabletop_offline_q16_20260720_142700/realtime_run_report.json)

### 12.1 对这些数字的正确解释

新配置显著提高了 entity→mesh 覆盖和按 commit object 数计算的绑定占比，但 0.5 m 级实体
合并距离也可能把邻近物体过度合并；旧 q16 的 0.075 m 又可能严重碎片化。没有人工标注时，
覆盖率提高不能证明 identity 更准确。必须抽样检查“一个真实物体对应几个 entity、一个
entity 是否混入多个物体、名称是否与完整物体一致”。

两次运行的门槛也不同：`fast_semantic_10cm` 的 entity merge / binding center / binding
AABB-gap 分别是 `0.5 / 0.75 / 0.15 m`，q16 分别是 `0.075 / 0.10 / 0.025 m`。因此两列
数字不是只改变一个变量的严格模型 A/B，不能把全部差异归因于 FastSAM 或 Hydra 分辨率。

质量报告的 semantic gate 主要验证 worker 就绪、prompt catch-up、队列排空、graph 已连接、
correction 已处理且没有 pending/unmapped/error。`rejected_no_mesh` 会被报告和告警，但不会
自动使整个运行失败。因此 `quality_passed=true` 表示工程产物完整和协议闭环，不表示语义
precision/recall 已达标。

## 13. 常见问题的分层诊断

| 现象 | 优先检查模块 | 关键证据 | 不应先做的事 |
| --- | --- | --- | --- |
| 画面中物体完全没有 mask | FastSAM/输入图像 | proposal 数、按尺寸召回、RGB/BGR A/B | 先调 DAM prompt |
| 有 mask，但没有 track | BotSort 阈值/空 tick | `>0.5`、`≥0.6`、confirmed 漏斗 | 只继续降低 FastSAM conf |
| mask 在非分割帧漂移 | mask propagation | bbox warp 与 carry-forward 比例 | 归因于 DAM |
| 同一物体出现多个 ID | tracker + MapMemory | ID switch、tracks/entity、entity purity | 只换分割模型 |
| 相邻物体被同一 ID 合并 | mask 重叠 + MapMemory | 同帧互斥、中心/AABB、merge distance | 继续放宽合并距离 |
| 名称错但 ID/mesh 正确 | DAM/代表帧 | focal mask、候选视图、重复 correction | 重建整个几何地图 |
| 已命名但地图中没有对象 | Hydra mesh/最终绑定 | object funnel、`rejected_no_mesh` | 只改语言模型 |
| 深度预览大面积空白 | FoundationStereo/截断 | 有效深度比例、0 值、confidence、距离范围 | 按 PNG 明暗直接判坏 |
| 恢复任务后 ID 异常复用 | checkpoint/session | tracker 与 semantic ID 是否恢复/隔离 | 假设 pose 对齐就足够 |

## 14. 推荐优化路线

### P0：先修正确性与可观测性

1. 用红/蓝 sentinel 和同帧 A/B 明确 FastSAM、BoxMot ReID、ECC、SAM/SAM2、DAM 的
   RGB/BGR 边界，只在第三方适配器入口转换。
2. 非分割帧不再调用完整 `BotSort.update(empty_detections)`；实现 predict-only 或独立传播。
3. 把 BotSort high/low/new/match/proximity/appearance 全部暴露到 YAML 和 Dashboard。
4. DAM prompt 按 `entity_id + revision` 去重；真实观察计数按唯一帧计数。
5. postpass 前对历史 label frames 执行 canonical semantic-ID remap。
6. 恢复运行时保存完整 identity 状态，或为新 tracker session 使用不可冲突的 epoch。
7. 增加端到端漏斗计数和每个拒绝原因，不再只看最终 PASS/FAIL。

这些问题修复前，不建议把模型替换作为第一优先级，因为模型 A/B 会被接口和生命周期问题
干扰，难以判断真实性能差异。

### P1：提高单一 ID 质量

- 同帧全局一对一关联，禁止多个可见目标无条件合并到同一 entity；
- 联合三维中心协方差、AABB gap/IoU、尺寸、ReID/CLIP 和时间连续性；
- 用稳健累计统计更新中心和尺寸，而不是仅保留创建时 dimensions；
- 支持可审计的 track reassign、entity split 和错误合并回滚；
- 将最终 mesh binding 也改为全局分配，避免按处理顺序抢占候选。

### P2：提高分割质量

- 实现有效的最大 mask 面积门和 thing/stuff 分层；
- 增加 mask-IoU、containment、whole/part 与深度层去重；
- 重叠像素按 score、深度有效率、时序稳定度与遮挡关系竞争；
- 桌面小物体采用更高输入分辨率时重新构建 TensorRT engine，并测量延迟/显存；
- 保留原始 RLE/PNG mask 给 DAM，polygon 仅用于索引或可视化，避免孔洞和细结构丢失。

### P3：提高 DAM 命名质量

- 选择 2–3 个互补代表视图，而不是达到阈值时立刻使用当前帧；
- 采用 mask-aware crop 和结构化输出；
- 将 category 与 caption 分离，规范 category 后多视角投票；
- 低置信、`unknown`、部件名和视图冲突时允许重试；
- 使用人工词汇表或开放词汇 embedding 约束名称，不把任意自由文本直接当 canonical class。

### P4：针对 Hydra object mesh 调参

- 增加 candidate/track/allocation/volume/reconstruction/mesh/binding 全漏斗；
- 按场景尺度分别维护 room-scale 和 tabletop 配置，避免用同一体积/距离阈值覆盖所有物体；
- 对小物体优先检查 `max_range`、观察次数、最小体积和对象重建分辨率；
- 对 `rejected_no_mesh` 抽样回放 RGB、深度、label、MapMemory AABB 与 Hydra object AABB；
- 只有在明确哪一级拒绝后，才降低对应阈值。

### P5：再比较 FastSAM、SAM2 和三维分割方案

建议对同一段 150–300 帧视频依次执行：

1. 当前基线；
2. P0 修复后的 FastSAM 基线；
3. 单帧 SAM/SAM2 automatic mask；
4. SAM2 video propagation；
5. 若引入真正 3D 模型，再单独比较 2D→3D 接口、延迟和 ID 质量。

只在上一步数据契约固定后比较下一步，才能判断提升来自模型本身还是工程修复。

## 15. 推荐评测指标

应先标注 30–50 个代表帧和一组跨帧真实物体轨迹，至少覆盖小物体、遮挡、快速转动、相邻
同类物体、反光/无纹理区域和 3–5 m 距离。指标按模块分层：

| 层级 | 推荐指标 |
| --- | --- |
| 分割 | mask AP50/AP75、boundary F1、小物体 recall、每帧 proposal、重复/嵌套率 |
| 跟踪 | confirmed 比例、ID switch、IDF1/HOTA、轨迹寿命、每个 GT 物体的 track 数 |
| 深度几何 | mask 深度有效率、中心误差、AABB IoU、背景泄漏率、几何协方差 |
| MapMemory | entity purity、entity fragmentation、同帧冲突、每个 GT 物体的 entity 数 |
| DAM | category accuracy、caption 可用率、部件误命名率、多视角一致率、重试率 |
| Hydra | object mesh recall/precision、mesh 体积误差、entity→mesh binding precision/recall |
| 系统 | P50/P95 延迟、GPU/CPU 内存、队列高水位、掉帧率、恢复后一致性 |

最终核心指标不应只是“识别了多少名字”，而应同时报告：

```text
真实物体召回
× 单一 ID 正确率
× 正确命名率
× 真实 mesh 绑定率
```

任何一层接近 0，最终可用语义对象数都会显著下降。

## 16. 代码、配置与产物索引

### 16.1 主要运行产物怎么读

| 产物 | 主要用途 |
| --- | --- |
| [`realtime_run_report.json`](output/g1_260720_1424_indoor_fast_semantic_10cm/realtime_run_report.json) | 帧数、分割/跟踪/传播/DAM/postpass/DSG 全漏斗和延迟；首先判断模块是否真正执行 |
| [`quality_report.json`](output/g1_260720_1424_indoor_fast_semantic_10cm/quality_report.json) | 工程质量门结果；PASS 不是有真值的识别准确率 |
| [`map_memory.sqlite3`](output/g1_260720_1424_indoor_fast_semantic_10cm/map_memory.sqlite3) | entity、session/local ID 映射、名称版本、correction 与交付状态的权威数据库 |
| [`semantic_sidecar/label_frames`](output/g1_260720_1424_indoor_fast_semantic_10cm/semantic_sidecar/label_frames) | 每帧 uint16 semantic ID PNG 及审计 JSON；用于 exact-label postpass 和 ID 追踪 |
| [`semantic_dsg_commit.json`](output/g1_260720_1424_indoor_fast_semantic_10cm/hydra_realtime/backend/semantic_dsg_commit.json) | 最终 object 数、绑定/拒绝统计、hash 和提交一致性 |
| [`dsg_with_mesh.json`](output/g1_260720_1424_indoor_fast_semantic_10cm/hydra_realtime/backend/dsg_with_mesh.json) | 最终 Dynamic Scene Graph、对象节点属性和 mesh 关联 |
| [`mesh.ply`](output/g1_260720_1424_indoor_fast_semantic_10cm/hydra_realtime/backend/mesh.ply) | 全局几何表面；单看 PLY 不能判断 DAM 命名或 entity 绑定质量 |

### 16.2 实现与配置入口

| 内容 | 位置 |
| --- | --- |
| 准实时主程序 | [`scripts/run_realtime_mapping.py`](scripts/run_realtime_mapping.py) |
| 分割统一适配 | [`src/daaam/utils/segmentation.py`](src/daaam/utils/segmentation.py) |
| TrackingService | [`src/daaam/tracking/services.py`](src/daaam/tracking/services.py) |
| track/mask 数据模型 | [`src/daaam/tracking/models.py`](src/daaam/tracking/models.py) |
| 实时语义、传播、标签与 DAM 提交 | [`src/daaam/realtime/semantic.py`](src/daaam/realtime/semantic.py) |
| RGB-D mask 几何 | [`src/daaam/realtime/masked_geometry.py`](src/daaam/realtime/masked_geometry.py) |
| MapMemory 关联与 correction | [`src/daaam/memory/store.py`](src/daaam/memory/store.py) |
| DAM worker | [`src/daaam/grounding/workers/dam_grounding.py`](src/daaam/grounding/workers/dam_grounding.py) |
| SceneGraph/mesh 绑定 | [`src/daaam/scene_graph/services.py`](src/daaam/scene_graph/services.py) |
| 语义质量门 | [`src/daaam/quality/gates.py`](src/daaam/quality/gates.py) |
| 准实时默认配置 | [`config/pipeline_config_realtime.yaml`](config/pipeline_config_realtime.yaml) |
| FastSAM 配置 | [`config/fastsam/fastsam_config.yaml`](config/fastsam/fastsam_config.yaml) |
| 10000 标签空间 | [`config/labels_pseudo.yaml`](config/labels_pseudo.yaml) |
| Dashboard 模块和参数定义 | [`src/daaam/dashboard/workflows.py`](src/daaam/dashboard/workflows.py) |
| 操作级完整流程 | [`Pipeline.md`](Pipeline.md) |

## 17. 特定运行说明

### 17.1 G1 `fast_semantic_10cm`

该运行是本文第 12 节的主要准实时证据源，使用 FoundationStereo 5 m 深度、FastSAM、
BoT-SORT/ReID、MapMemory、DAM-3B 和 Hydra exact-label postpass。它已完成产物提交和 hash
校验，但仍有 183 个 `rejected_no_mesh` correction operations，所以应理解为“工程流程
完整、语义 mesh 覆盖仍需提升”。

### 17.2 CODa 复现

CODa 复现实际使用 `dam_multi_image` grounding worker 和本地 `nvidia/DAM-3B`。部分深度
已经预先写入 rosbag；这不是所有项目运行的通用要求。该完整运行生成 5,310 个场景图节点、
8,313 条边和 8 个房间。精确参数见
[`pipeline_config.yaml`](output/coda/out_20260713_103618/pipeline_config.yaml)，复现环境和过程见
[`REPRODUCTION_CODA.md`](REPRODUCTION_CODA.md)。
