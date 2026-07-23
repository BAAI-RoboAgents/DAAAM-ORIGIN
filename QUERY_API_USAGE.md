# DAAAM 多语种语义查询 REST API

需要 mesh 俯视定位、候选列表和 FastSAM 图片证据的浏览器交互界面时，使用独立查询服务；
它不与地图构建 dashboard 或构建进程耦合。启动方法见
[`SEMANTIC_QUERY_UI.md`](SEMANTIC_QUERY_UI.md)。

查询服务对 DSG 物体描述做本地向量检索，中文和英文使用同一个多语种编码器，不依赖
LLM 翻译。默认模型为：

```text
sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

该模型输出 768 维向量。图侧物体描述与查询文本必须由完全相同的模型编码；“维度都是
768”不能证明模型兼容。服务默认要求与 DSG 同名的 checksum-bound manifest，并在加载模型
前校验文件哈希、模型名、维度和可查询物体数。未能绑定到 Hydra object mesh、但在 MapMemory
中有可靠语义和空间记录的实体会写入 checksum-bound `*.semantic.json` 旁路索引，不会修改
权威 DSG。

## 1. 为 DSG 生成多语种查询嵌入

输入 DSG 的 object 节点需要已有 `attributes.metadata.description`。工具不会覆盖权威输入图：

```bash
source .repro/venv/bin/activate

python scripts/prepare_query_dsg_embeddings.py \
  --dsg-file output/<run>/dsg_rebound.json \
  --output-file output/<run>/dsg_updated.json \
  --binding-report output/<run>/dsg_rebound.binding.json \
  --device cpu
```

`--binding-report` 可省略，工具会尝试发现与输入 DSG 同目录、同 stem 的 `.binding.json`。
报告中最终状态为 `rejected_no_mesh` 的实体会从原始 MapMemory 恢复描述、位置、包围盒和真实
观测时间，并进入语义旁路。严格 mesh 绑定结果仍保留在 DSG object 节点中。

如果描述由旧版 `corrections.yaml` 按语义 ID 提供，也可以继续使用：

```bash
python scripts/prepare_zed_query_dsg.py \
  --run-dir output/<run> \
  --sentence-model-name sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

输出必须同时保留：

```text
dsg_updated.json
dsg_updated.manifest.json
dsg_updated.semantic.json
```

只有存在未绑定语义实体时才会生成 `dsg_updated.semantic.json`。三个文件通过 SHA-256 相互
绑定，任何一个被修改后都必须重新运行嵌入生成器。

如果需要 top-1 原图证据，再从生成该 DSG 的完整 DAM 运行构建 FastSAM 证据：

```bash
python scripts/prepare_query_evidence.py \
  --dsg-file output/<run>/dsg_updated.json \
  --semantic-run output/<completed-dam-run>
```

额外输出 `dsg_updated.evidence.json`、`query_evidence/*.png`、
`query_evidence/cutouts/*.png` 和 `query_evidence/point_clouds/*.npz`。证据覆盖 DSG mesh
object 和语义旁路中的实体；`cutouts` 保存带 alpha mask 的紧凑原图前景，`point_clouds`
保存由同一 FastSAM mask、逐像素深度、相机内参和世界位姿联合反投影得到的彩色三维点。
生成器优先选择存在有效深度的真实 FastSAM 调用帧，不使用中间帧的
BotSORT/carry-forward 传播 mask；同时校验 DSG、原图、深度、语义 mask、帧号、绝对时间、
相机位置和运行配置哈希。证据图在原图上绘制半透明 mask、轮廓和外接框；抠图、点云及
相机位姿均由 manifest 的 SHA-256 绑定。

manifest v1 的关键字段为：

```json
{
  "dsg_sha256": "...",
  "queryable_objects": 216,
  "dsg_queryable_objects": 49,
  "geometry_counts": {
    "mesh_bound": 49,
    "spatial_only": 167,
    "image_only": 0
  },
  "embedding": {
    "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "dimension": 768,
    "normalized": true
  }
}
```

修改或替换 JSON 后必须重新生成 manifest；否则服务会拒绝启动。旧 Sentence-T5 图必须重新
编码，不能只改 manifest 中的模型名。

## 2. 启动服务

推荐先只监听本机：

```bash
python scripts/serve_query_api.py \
  --dsg output/<run>/dsg_updated.json \
  --host 127.0.0.1 \
  --port 8765 \
  --min-similarity 0.55 \
  --min-margin 0
```

可用的检索配置：

| 参数 / 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `--sentence-model-name` / `DAAAM_QUERY_SENTENCE_EMBEDDING_MODEL_NAME` | 多语种 MPNet | 必须与 manifest 完全一致 |
| `--min-similarity` / `DAAAM_QUERY_MIN_SIMILARITY` | `0.55` | top-1 低于该值时返回 `found=false` |
| `--min-margin` / `DAAAM_QUERY_MIN_MARGIN` | `0` | top-1 与 top-2 差值下限；`0` 表示关闭 |
| `--allow-unverified-embeddings` | 关闭 | 仅用于临时读取无 manifest 的旧图，不建议生产使用 |

`0.55` 来自首轮 46 个 DAM 描述的高精度校准，不是跨场景通用概率阈值：当时 22 个
地图外中英文负查询最高分为 `0.5346`，而常用已知物体查询大多为 `0.59–0.84`。少数过短、
含糊的有效词也可能被拒绝，应优先补充颜色/材质/物体类型。场景内存在多个同类物体时，
top-1/top-2 分差通常很小，因此默认不启用 margin；只有需要唯一实例消歧并完成标注校准后
才建议设置，例如 `0.03`。

健康检查：

```bash
curl -sS http://127.0.0.1:8765/health | jq
```

示例：

```json
{
  "status": "ok",
  "queryable_objects": 216,
  "mesh_bound_objects": 49,
  "spatial_only_objects": 167,
  "image_only_objects": 0,
  "evidence_available_objects": 216,
  "embedding_dimension": 768,
  "sentence_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
  "embedding_model_verified": true,
  "encoder_device": "cpu",
  "default_min_similarity": 0.55,
  "default_min_margin": 0.0,
  "llm_enabled": false,
  "default_llm_model": "qwen3.7-plus"
}
```

## 3. 本地多语种检索

`POST /v1/query/retrieve` 不访问外部模型。中文和英文可直接查询：

```bash
curl -sS http://127.0.0.1:8765/v1/query/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"白色天花板灯","top_k":3}' | jq
```

请求字段：

| 字段 | 约束 | 说明 |
| --- | --- | --- |
| `query` | 非空字符串 | 中文或英文物体描述 |
| `top_k` | `1..50`，默认 `5` | 接受后最多返回的候选数 |
| `min_similarity` | `-1..1`，可选 | 覆盖本次请求的余弦下限 |
| `min_margin` | `0..2`，可选 | 覆盖本次请求的 top-1/top-2 分差下限 |
| `require_mesh` | 布尔值，默认 `false` | 只返回已绑定真实 object mesh 的结果 |

接受结果：

```json
{
  "query": "白色天花板灯",
  "found": true,
  "rejection_reason": null,
  "top_score": 0.82,
  "top1_margin": 0.08,
  "min_similarity": 0.55,
  "min_margin": 0.0,
  "matches": [
    {
      "rank": 1,
      "score": 0.82,
      "node_id": "O(52)",
      "entity_id": 474,
      "semantic_label": 474,
      "description": "A long white ceiling light fixture...",
      "position_m": [-0.127, 1.101, 4.332],
      "dimensions_m": [1.12, 0.31, 0.08],
      "geometry_status": "mesh_bound",
      "geometry_confidence": 0.93,
      "source": "dsg",
      "first_observed_s": 79.838,
      "last_observed_s": 85.826
    }
  ],
  "top1_evidence": {
    "evidence_id": "O_52",
    "image_url": "/v1/evidence/O_52.png",
    "frame_index": 417,
    "sensor_time_ns": 1784199791000000000,
    "observed_s": 79.838,
    "bbox_xyxy": [420, 105, 866, 314],
    "mask_pixels": 42108,
    "mask_source": "fastsam_segmentation",
    "source_image_sha256": "...",
    "annotated_image_sha256": "..."
  }
}
```

`node_id` 以 `O(...)` 开头表示 DSG object 节点；以 `M(...)` 开头表示 MapMemory 语义旁路实体。
后者的 `geometry_status` 通常为 `spatial_only`，可以查询、显示 RGB-D mask 点云和三维
包围盒并查看 FastSAM 原图证据，但没有可单独着色的 Hydra object mesh。其查询坐标优先
使用证据帧逐像素反投影的稳健中心和 5%–95% 世界坐标包围盒，不再使用旧版独立中值估计。
需要严格 Hydra 几何结果时在请求中加入 `"require_mesh":true`。

保存 top-1 证据图片：

```bash
curl -sS http://127.0.0.1:8765/v1/evidence/O_52.png \
  -o /tmp/O_52_fastsam_evidence.png
```

未命中时 `top1_evidence=null`。命中但旧 DSG 尚未生成证据 sidecar 时也返回 `null`，不会用
不相关图片兜底。

地图中不存在目标或结果过于模糊时仍返回 HTTP 200，但不会强行返回最相似的错误物体：

```json
{
  "query": "红色消防栓",
  "found": false,
  "rejection_reason": "below_min_similarity",
  "top_score": 0.21,
  "top1_margin": 0.01,
  "min_similarity": 0.55,
  "min_margin": 0.0,
  "matches": [],
  "top1_evidence": null
}
```

`rejection_reason` 可能为 `below_min_similarity` 或 `below_min_margin`。`top_score` 和
`top1_margin` 仅用于调试与校准，不是概率。

CLI 使用同一套模型契约和拒识规则：

```bash
python scripts/query_sentence_dsg.py \
  --dsg output/<run>/dsg_updated.json \
  --query '白色天花板灯' \
  --top-k 3 \
  --min-similarity 0.55
```

CLI 默认同时检索 `mesh_bound` 和 `spatial_only`；添加 `--require-mesh` 可限制为真实 object
mesh。

不传 `--query` 即进入中英文交互模式；每次命中后会打印 top-1 证据图的本地路径：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  python scripts/query_sentence_dsg.py \
  --dsg output/<run>/dsg_updated.json \
  --top-k 3 \
  --min-similarity 0.55
```

要在每次查询时同时导出图片证据和 mesh 俯视位置图，增加
`--visual-output-dir`：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  python scripts/query_sentence_dsg.py \
  --dsg output/<run>/dsg_updated.json \
  --top-k 3 \
  --min-similarity 0.55 \
  --visual-output-dir output/<run>/query_results
```

交互模式下每个问题会建立一个独立的 UTC 时间戳目录，其中包含：

- `rank_01_<node>_evidence.png` 等所有可用 top-k FastSAM 原图证据；
- `mesh_topdown_query.png`，直接从当前 DSG mesh 绘制的 XY 俯视图，星号为 top-1、圆点为
  其他候选、三角形为证据帧相机位置；
- `query_result.json`，记录查询、得分、三维位置、证据帧，以及 DSG 和输出图片的 SHA-256。

没有精确 FastSAM 证据的候选会在 JSON 中明确标记 `evidence_available=false`，但只要有可靠
三维坐标仍会在俯视图上标出；不会拿其他对象的图片替代。

## 4. 受证据约束的问答

`POST /v1/query/ask` 是可选的 LLM 路径。密钥只由服务端读取：

```bash
export DAAAM_KEY='你的密钥'
export DAAAM_LLM_MODEL='qwen3.7-plus'
```

调用：

```bash
curl -sS http://127.0.0.1:8765/v1/query/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"白色天花板灯在哪里？","top_k":5}' | jq
```

`/ask` 先把问题改写成简短视觉检索短语，再使用与 `/retrieve` 相同的拒识规则。响应同样包含
`found`、`rejection_reason`、`top_score`、`top1_margin` 和 `matches`。如果检索被拒绝，
`matches=[]`，回答模型会收到明确的“证据不足”状态，不能把低分候选当成地图事实。

## 5. 调用方约定

调用方必须先检查 `found`，不能直接访问 `matches[0]`：

```python
with urlopen(request, timeout=10) as response:
    result = json.load(response)

if not result["found"]:
    print("地图中未找到可信匹配：", result["rejection_reason"])
else:
    print(result["matches"][0]["node_id"])
```

错误码：

| HTTP 状态 | 含义 |
| --- | --- |
| `200` | 查询执行成功；是否找到目标由 `found` 表示 |
| `400` | 编码失败、图/模型不兼容或运行期阈值非法 |
| `422` | 请求字段、范围或空文本校验失败 |
| `503` | `/ask` 未配置服务端 LLM 密钥 |
| `502` | OpenAI 兼容端点、模型或凭据请求失败 |

没有 description/embedding 的 unknown 对象不会进入查询索引。当前接口只查询 object 节点，
不能把物体级结果推断为可靠的 room 归属、邻接关系或导航拓扑。
