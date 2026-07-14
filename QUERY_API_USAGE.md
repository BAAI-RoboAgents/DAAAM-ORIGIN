# DAAAM 语义查询 REST API 使用说明

本文说明如何将已复现的 G1 场景图作为查询服务提供给外部模块。

服务加载的场景图为：

```text
output/g1_20260713_170500_sync10ms_full/out_20260714_114803/dsg_updated.json
```

其中有 103 个带自然语言描述和 768 维 Sentence-T5 向量的物体节点。服务提供两种查询方式：

| 接口 | 是否访问外部模型 | 用途 |
| --- | --- | --- |
| `POST /v1/query/retrieve` | 否 | 本地 Sentence-T5 余弦相似度检索 |
| `POST /v1/query/ask` | 是，需服务端 `DAAAM_KEY` | 中文问题改写、检索、基于候选节点的受约束回答 |

`/v1/query/ask` 使用 OpenAI 兼容接口和默认模型 `qwen3.7-plus`。API 密钥永远不由调用方通过 HTTP 传递，只在服务进程的环境变量中读取。

## 1. 环境准备

从仓库根目录执行：

```bash
cd /home/user/Code/DAAAM_Origin
source /opt/ros/jazzy/setup.bash
source .repro/ros2_ws/install/setup.bash
source .repro/venv/bin/activate
```

纯本地检索不需要设置密钥。若要启用模型问答，在**启动服务之前**设置：

```bash
export DAAAM_KEY='你的密钥'
export DAAAM_LLM_MODEL='qwen3.7-plus'  # 可选；这是默认值
```

默认使用的 OpenAI 兼容地址为：

```text
https://llm-g3o8d3j71xbf6prc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

若需替换地址，设置 `DAAAM_LLM_BASE_URL` 或使用 `--base-url`。不要将密钥写入脚本、配置文件、请求 JSON 或终端日志。

## 2. 启动服务

推荐先仅监听本机，端口使用 `8765`：

```bash
python scripts/serve_query_api.py \
  --dsg output/g1_20260713_170500_sync10ms_full/out_20260714_114803/dsg_updated.json \
  --host 127.0.0.1 \
  --port 8765
```

模型和场景图完成加载后应看到类似输出：

```text
Serving 103 queryable objects at http://127.0.0.1:8765 (OpenAPI: /docs)
Uvicorn running on http://127.0.0.1:8765
```

交互式 API 文档：<http://127.0.0.1:8765/docs>

健康检查：

```bash
curl -sS http://127.0.0.1:8765/health
```

示例响应：

```json
{
  "status": "ok",
  "queryable_objects": 103,
  "sentence_model": "sentence-transformers/sentence-t5-large",
  "llm_enabled": true,
  "default_llm_model": "qwen3.7-plus"
}
```

`llm_enabled: false` 不影响 `/v1/query/retrieve`；它表示服务进程未读取到 `DAAAM_KEY`，因此不能调用 `/v1/query/ask`。设置密钥后必须重启该服务进程。

### 端口已被占用

如果出现 `Errno 98 ... address already in use`，先找出监听进程：

```bash
ss -ltnp '( sport = :8765 )'
```

确认该 PID 的命令确实是你要替换的 `scripts/serve_query_api.py` 后，再停止它：

```bash
kill -TERM <PID>
```

然后重新执行启动命令。也可以改用其它端口，例如 `--port 8766`。

## 3. 接口定义

### `GET /health`

返回服务状态、已加载的可查询物体数量、Sentence-T5 模型名和 LLM 是否可用。

### `POST /v1/query/retrieve`

纯本地检索。服务将 `query` 编码为 Sentence-T5 向量，并与图中物体描述的向量计算余弦相似度。此接口不访问 LLM、DashScope 或其他网络服务。

请求：

```json
{
  "query": "white ceiling light",
  "top_k": 5
}
```

`query` 为非空字符串；`top_k` 取值为 1–50，默认 5。由于当前物体描述主要是英文，建议此接口直接使用英文的物体外观短语。

响应：

```json
{
  "query": "white ceiling light",
  "matches": [
    {
      "rank": 1,
      "score": 0.899,
      "node_id": "O(52)",
      "semantic_label": 474,
      "description": "A long, rectangular, white light fixture...",
      "position_m": [-0.127, 1.101, 4.332],
      "first_observed_s": 79.838,
      "last_observed_s": 85.826
    }
  ]
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `score` | 余弦相似度；用于同一次查询内排序，不应当作为跨查询的绝对置信度 |
| `node_id` | DSG 物体节点标识，可用于在 Rerun 中定位/核对 |
| `position_m` | 场景图坐标系中的 `[x, y, z]`，单位为米 |
| `first_observed_s` / `last_observed_s` | 自采集开始的相对观测时间，单位为秒 |

调用示例：

```bash
curl -sS http://127.0.0.1:8765/v1/query/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"white ceiling light","top_k":3}'
```

### `POST /v1/query/ask`

受证据约束的自然语言问答，需在启动服务前配置 `DAAAM_KEY`。流程为：

1. `qwen3.7-plus` 将问题转成简短的英文视觉检索短语；
2. 服务在本地场景图中执行 Sentence-T5 检索；
3. `qwen3.7-plus` 只根据返回的节点描述、坐标和时间生成回答，并引用节点 ID（例如 `[O(52)]`）。

请求：

```json
{
  "query": "白色天花板灯在哪里？",
  "top_k": 5,
  "model": "qwen3.7-plus"
}
```

`model` 可省略，省略时使用服务启动时的默认模型。

调用示例：

```bash
curl -sS http://127.0.0.1:8765/v1/query/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"白色天花板灯在哪里？","top_k":5}'
```

响应包含：

```json
{
  "question": "白色天花板灯在哪里？",
  "retrieval_query": "white ceiling light",
  "model": "qwen3.7-plus",
  "matches": ["与 /v1/query/retrieve 相同的候选节点"],
  "answer": "根据候选节点，…… [O(52)]"
}
```

该接口每次正常请求会产生两次模型调用（检索短语改写、依据证据回答）。问答服务不应将未被 `matches` 支持的内容当作地图事实。

## 4. 外部模块集成

外部模块只需保存服务地址，例如：

```text
DAAAM_QUERY_API_URL=http://127.0.0.1:8765
```

然后向 `${DAAAM_QUERY_API_URL}/v1/query/retrieve` 或 `${DAAAM_QUERY_API_URL}/v1/query/ask` 发送 JSON `POST` 请求。外部模块不应保存或转发 `DAAAM_KEY`。

最小 Python 示例（仅使用标准库）：

```python
import json
from urllib.request import Request, urlopen

base_url = "http://127.0.0.1:8765"
payload = json.dumps({"query": "white ceiling light", "top_k": 3}).encode()
request = Request(
    f"{base_url}/v1/query/retrieve",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=10) as response:
    matches = json.load(response)["matches"]
print(matches[0]["node_id"], matches[0]["position_m"])
```

## 5. 错误码与限制

| HTTP 状态 | 含义与处理 |
| --- | --- |
| `200` | 请求成功 |
| `400` | 请求字段不合法、文本为空，或查询向量与图向量不兼容 |
| `503` | 请求了 `/v1/query/ask`，但服务进程没有 `DAAAM_KEY` |
| `502` | 模型服务不可达、模型名不可用或兼容接口请求失败；检查服务端网络、密钥和模型配置 |

当前地图只有 103/105 个 Hydra 物体拥有可用描述向量；剩余 unknown 物体无法被语义检索命中。room 与 traversability 层没有连接边，因此不要将当前 API 的物体级结果推断为可靠的房间级拓扑问答。

默认服务只绑定 `127.0.0.1`，避免未认证的 HTTP 接口暴露到网络。确需跨机器访问时，可显式使用 `--host 0.0.0.0`，并通过防火墙、反向代理或专用网络限制访问来源。
