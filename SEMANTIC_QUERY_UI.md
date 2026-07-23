# DAAAM 独立语义地图查询界面

该服务只读取已经完成的语义地图并执行本地多语种查询，不导入构建 dashboard，也不能启动、
停止或修改建图流程。查询结果会写到所选地图的 `query_results/` 目录，作为可追溯导出产物。

## 启动

从仓库根目录运行：

```bash
cd /home/user/Code/DAAAM_Origin
source .repro/venv/bin/activate

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
python scripts/serve_semantic_query_ui.py \
  --host 127.0.0.1 \
  --port 8790 \
  --output-root /home/user/Code/DAAAM_Origin/output
```

浏览器访问 <http://127.0.0.1:8790>。默认监听地址和端口与地图构建 dashboard 分离；服务没有
鉴权，不应直接暴露到不可信网络。

## 使用

1. 在顶部输入包含 `dsg_updated.json` 的语义地图输出目录，例如：
   `/home/user/Code/DAAAM_Origin/output/g1_260720_1424_indoor_fast_semantic_10cm`。
   服务也会自动发现 `--output-root` 下第一层可查询地图，输入框会提供候选。
2. 点击“加载地图”。首次加载会初始化多语种编码器，之后同一地图从内存缓存读取。
3. 输入中文或英文物体描述，选择 top-k；“仅 Mesh 实体”开启后会排除 `spatial_only` 记录。
4. 查询完成后，中央显示带物体 ID、得分和证据相机连线的 mesh 俯视图。点击左侧候选可切换
   右侧 FastSAM 证据照片与空间/帧元数据。

界面还提供：

- 鼠标滚轮缩放、拖动平移、双击复位和地图全屏；
- top-k 候选、接受/拒识原因、相似度、几何状态和证据可用性；
- 原始证据图、俯视定位图及 `query_result.json` 下载；
- 浏览器本地查询历史和快捷查询；
- 缺失精确证据时的明确提示，不使用其他对象图片兜底；
- DSG、证据图和导出图片 SHA-256 溯源。

每次查询建立独立目录：

```text
output/<semantic-run>/query_results/<utc-time>_<query-hash>/
├── mesh_topdown_query.png
├── query_result.json
└── rank_XX_<node>_evidence.png
```

## 深链接

可以用 URL 预选地图并自动查询：

```text
http://127.0.0.1:8790/?path=/absolute/output/run&q=黑色电脑显示器&require_mesh=1
```

## 独立 API

- `GET /api/health`：查询服务状态；
- `GET /api/maps`：发现输出根目录下的查询就绪地图；
- `POST /api/map/open`：校验并加载地图；
- `GET /api/map/mesh-preview.png`：生成未标注的 mesh XY 俯视图；
- `POST /api/query`：本地语义查询并输出图片证据与标注俯视图；
- `GET /api/file`：读取所选地图目录内允许的查询产物。

所有地图和文件路径都必须解析到 `--output-root` 内，目录穿越和其他文件类型会被拒绝。
OpenAPI 文档位于 <http://127.0.0.1:8790/docs>。
