# DAAAM 语义地图流程面板

本地流程面板把 [`Pipeline.md`](Pipeline.md) 中的真实构建链路、入口参数和运行产物统一成一个浏览器界面。它不会在 Web 进程内加载 CUDA 模型，而是以受控子进程调用现有 runner，因此仍保留原命令、Conda 环境、日志和产物布局。

## 界面覆盖范围

- **高质量离线链**：`prepare / select / depth / calibrate`，从 `03_geometry` 扇出的 `temporal / odometry / loops`，以及 `optimize / filter / validate / fuse / map`。
- **准实时语义链**：`pose / depth / dynamic / fusion / global` 主链，异步 `FastSAM / BotSort / MapMemory / DAM` 旁路，以及 exact-label postpass、DSG mesh 绑定、durable commit 和质量门。
- **查询资产链**：可选 rebind、多语言 embeddings、FastSAM/RGB-D evidence 和查询服务。目前作为只读参考图展示，因为这些步骤尚未由一个仓库 runner 一键闭合。

界面会显示模块职责、输入输出、关键参数、运行状态、帧进度、吞吐/P95/drop、DAM 交付状态、质量门和最终产物。运行超过默认 5 分钟后会按进度里程碑和周期显示提醒；阈值可在左侧“进度提醒”中调整或关闭。历史运行只扫描 `output/*` 一级目录与已知小型状态文件，不递归遍历大型数据集。

## 安装和启动

使用项目 Python 3.12 环境安装可选依赖：

```bash
.repro/venv/bin/pip install -e '.[dashboard]'
```

启动本地服务：

```bash
.repro/venv/bin/python scripts/serve_mapping_dashboard.py \
  --host 127.0.0.1 \
  --port 8787
```

然后访问 `http://127.0.0.1:8787`。可用 `--output-root` 指定另一个允许读取和写入的运行根目录。

## 推荐使用顺序

1. 选择“准实时动态语义地图”或“高质量离线语义地图”。
2. 选择预设；桌面小物体应使用“桌面小物体语义”，它会同时设置 Pipeline YAML、Hydra YAML 和三个必须显式传入的实体/mesh 距离。
3. 设置数据集与新的 `output/<run-name>` 目录。
4. 保持“仅生成计划”开启，先点击“预览命令”，检查最终 argv 和警告。
5. 启动 dry-run；确认路径、配置和模型后，再关闭 dry-run 发起真实运行。
6. 在流程图中点击节点查看模块指标，在底部查看实时 stdout/stderr 和质量门。

### FoundationStereo 深度预览

点击流程图中的 `FoundationStereo`/`depth` 节点会打开画布内的深度窗口。离线深度阶段或准实时 `foundation-worker` 正在执行时，窗口会自动出现并跟随最新一张完整写入的深度图；点击左右箭头、按键盘 `←/→` 或取消“跟随最新”可暂停自动跟随。估计完成后帧索引会保留，可以继续前后翻阅。

预览器仅接受以下两个本次运行目录内的原始 FoundationStereo 来源：

- 离线：`02_selected/depth/*.png`；
- 准实时 worker：`generated_depth/depth/*.png`。

同名 `depth_metadata/*.json` 是单帧写入完成标记。预计算输入和动态隔离后的 `static_depth` 不会冒充本次 FoundationStereo 估计结果。服务端把 uint16 毫米深度按固定 `0.25 m` 到报告最大深度范围渲染为 Turbo 彩虹图；报告缺失时使用室内默认上限 `5 m`。零值/无效深度显示为浅灰，固定色条保证同一颜色在不同帧中表示同一距离。

面板不会自动勾选 FoundationStereo 研究/非商用许可证、固定标定或 RGB-D 人工预览验收。这些硬门必须由运行者明确确认。

## 状态判定

运行中优先使用面板维护的 PID、退出码和日志；准实时帧进度来自原子更新的 `realtime_checkpoint.json`。结束后切换为精确的 artifact-backed 状态：

- 离线：`mapping_run.json` 与各阶段 report；
- 准实时：`run_manifest.json`、checkpoint、`realtime_metrics.json`、`hydra_semantic_postpass.json`、`semantic_dsg_commit.json`、`quality_report.json` 和 `realtime_run_report.json`。

准实时运行只有同时满足以下条件才显示完整成功：最终 report 为 `complete`、exact-label 覆盖率为 `1.0`、DAM drain/ACK 通过、durable commit 存在且最终 hard quality gates 全部通过。`rejected_no_mesh` 即使不阻断质量 PASS，也会单独显示警告。

旧离线 `map` 的目标输出路由和完成检查存在已知缺陷：文件存在只能证明 Hydra 几何产物被发现，不能证明 DAM correction 已严格 drain、绑定和持久化。因此面板把完成的旧离线 `map` 标成警告，不把它冒充为权威语义提交。

## 安全边界

- 默认只监听 `127.0.0.1`，没有鉴权；不要直接暴露到不可信网络。
- runner 固定来自仓库 `scripts/` allowlist，命令由 argv 数组执行，永不使用 shell 拼接。
- 运行目录必须位于 `--output-root` 下；产物 API 也拒绝目录穿越。
- 停止运行时先向独立进程组发送 `SIGINT`，给 DAM/Hydra drain 和 cleanup 留出时间；超时后才升级为 `SIGTERM`。

## 当前限制

- 面板服务重启后会失去内存中的 PID/日志游标，但仍能从运行产物恢复历史状态。
- runner 当前没有周期 telemetry 文件；运行中的精细 queue/latency 指标要到 `realtime_metrics.json` 落盘后才完整显示。帧进度和日志仍会实时更新。
- 三维 DSG/mesh 继续由现有 Rerun visualizer 负责；该面板聚焦流程、参数、状态和产物，不重复实现一套不兼容的 3D renderer。

## API

主要接口为：

- `GET /api/workflows`
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/depth-frames`
- `GET /api/runs/{run_id}/depth-frames/{frame_index}.png`
- `POST /api/commands/preview`
- `POST /api/runs`
- `GET /api/processes/{process_id}/events`
- `POST /api/processes/{process_id}/stop`

交互式接口说明位于 `/docs`。

修改 Python 后端或新增 API 路由后必须重启仪表盘服务；浏览器刷新只会重新读取静态前端文件，无法更新已经运行在内存中的 FastAPI 路由表。若深度窗口提示后端版本过旧，请重启服务后刷新页面。
