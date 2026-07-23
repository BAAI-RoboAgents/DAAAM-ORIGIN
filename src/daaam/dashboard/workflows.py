"""Declarative DAAAM workflow graphs and their user-facing CLI parameters."""

from __future__ import annotations

from .models import (
    ParameterDefinition as P,
    ParameterGroup as G,
    WorkflowDefinition,
    WorkflowEdge as E,
    WorkflowNode as N,
    WorkflowPreset as Preset,
)


OFFLINE_STAGES = (
    "prepare",
    "select",
    "depth",
    "calibrate",
    "temporal",
    "odometry",
    "loops",
    "optimize",
    "filter",
    "validate",
    "fuse",
    "map",
)

REALTIME_STAGES = ("pose", "depth", "dynamic", "fusion", "global")


def _offline_workflow() -> WorkflowDefinition:
    nodes = (
        N(
            "prepare",
            "双目准备",
            "input",
            "同步 G1 鱼眼双目、重投影为针孔，并把图像、位姿与绝对时间绑定。Prepared Stereo 会跳过重投影。",
            ("原始 G1 双目 / Prepared Stereo", "标定与 pose"),
            ("01_pinhole", "pinhole_preparation_report.json"),
            ("adapter", "max_delta_ms", "horizontal_fov_deg", "stereo_calibration_report"),
            ("配对帧", "最大同步差", "时间契约"),
            "G1 模式必须提供可信双目标定。",
            True,
            20,
            150,
        ),
        N(
            "select",
            "关键帧选择",
            "geometry",
            "结合 pose 与图像内容变化保留时间安全关键帧。",
            ("01_pinhole / Prepared Stereo",),
            ("02_selected", "keyframe_selection_report.json"),
            ("soft_translation_m", "soft_rotation_deg", "hard_translation_m", "hard_rotation_deg", "max_gap_s"),
            ("源帧", "保留帧", "压缩率", "选择原因"),
            x=200,
            y=150,
        ),
        N(
            "depth",
            "FoundationStereo",
            "model",
            "生成度量深度、左右一致性、遮挡与置信度证据。",
            ("左右针孔图像", "内参与 baseline", "checkpoint"),
            ("depth", "depth_confidence", "depth_consistency", "depth_metadata"),
            ("checkpoint", "valid_iters", "depth_profile", "depth_scale", "depth_precision", "accept_license"),
            ("处理/失败帧", "有效率", "一致率", "GPU/RSS"),
            "模型 checkpoint、相邻 cfg.yaml 与非商用许可证确认是硬门。",
            True,
            380,
            150,
        ),
        N(
            "calibrate",
            "地面/坐标校准",
            "geometry",
            "应用固定地面与图像坐标校准，并固定几何最大深度。",
            ("02_selected + depth", "floor calibration report"),
            ("03_geometry",),
            ("floor_calibration_report", "geometry_max_depth_m"),
            ("深度尺度", "有效 baseline", "旋转修正"),
            "G1 链在深度后继续运行时，固定地面标定报告是硬门。",
            True,
            560,
            150,
        ),
        N(
            "temporal",
            "时序深度诊断",
            "quality",
            "诊断相邻深度一致性；结果在全局优化时参与不确定度设置。",
            ("03_geometry",),
            ("04_temporal_input/report.json",),
            metrics=("一致率", "中值误差", "有效覆盖"),
            x=740,
            y=20,
        ),
        N(
            "odometry",
            "局部 RGB-D 里程计",
            "geometry",
            "建立局部视觉/RGB-D 约束和可审计的轨迹细化结果。",
            ("03_geometry",),
            ("05_rgbd_window_graph", "trajectory_refinement.json"),
            ("local_keyframe_distance_m", "local_max_keyframe_gap", "local_neighbor_span", "local_min_inliers"),
            ("关键帧", "视觉约束", "残差", "路径长度"),
            x=740,
            y=150,
        ),
        N(
            "loops",
            "闭环发现",
            "geometry",
            "检索闭环候选并通过几何验证。至少一个 verified loop 才能继续。",
            ("03_geometry",),
            ("06_loop_closures/loop_closure_report.json",),
            ("loop_dense_candidate_count", "max_loop_gravity_residual_deg"),
            ("候选数", "验证数", "ICP RMSE"),
            "verified_count >= 1 是硬门。",
            True,
            740,
            280,
        ),
        N(
            "optimize",
            "全局位姿图优化",
            "geometry",
            "汇合局部里程计、时序诊断和闭环，执行 gravity-SE3 优化。",
            ("03_geometry", "temporal", "odometry", "loops"),
            ("07_global_pose_graph", "global_pose_graph_report.json"),
            ("global_iterations",),
            ("收敛状态", "初末残差", "选中闭环", "路径长度"),
            x=940,
            y=150,
        ),
        N(
            "filter",
            "时序深度过滤",
            "geometry",
            "用多邻帧证据过滤不一致深度，同时保留原始绝对时间。",
            ("07_global_pose_graph", "02_selected depth evidence"),
            ("08_temporal_depth_filtered",),
            ("temporal_filter_neighbor_offsets", "temporal_filter_scale", "temporal_filter_min_judged", "temporal_filter_min_support"),
            ("拒绝像素", "前后有效率", "证据覆盖"),
            x=1130,
            y=150,
        ),
        N(
            "validate",
            "最终深度质量门",
            "quality",
            "检查整体、误差与最差窗口三项时序深度硬门。",
            ("08_temporal_depth_filtered",),
            ("09_temporal_validation/report.json",),
            ("final_min_adjacent_agreement", "final_max_adjacent_median_error_m", "final_min_window_adjacent_agreement"),
            ("整体一致率", "中值误差", "最差窗口"),
            "三项自动质量门必须全部通过。",
            True,
            1310,
            70,
        ),
        N(
            "fuse",
            "直接 RGB-D 预览",
            "artifact",
            "生成独立于 Hydra 的稠密融合 PLY 与预览，供人工检查地面、墙体和物体。",
            ("08_temporal_depth_filtered",),
            ("10_direct_rgbd_fusion", "preview.png", "fusion.ply"),
            ("fusion_frame_step", "fusion_pixel_step", "fusion_voxel_size_m", "accept_preview"),
            ("采样帧", "点数", "场景边界"),
            "必须由用户显式接受预览，界面不会自动替用户确认。",
            True,
            1310,
            230,
        ),
        N(
            "map",
            "DAAAM + Hydra DSG",
            "semantic",
            "FastSAM、BotSort/ReID、代表帧分配、DAM-3B 与 Hydra/Khronos 共同生成网格和 DSG。",
            ("08_temporal_depth_filtered", "pipeline YAML", "Hydra YAML"),
            ("mesh.ply", "dsg.json", "dsg_with_mesh.json", "corrections.yaml"),
            ("pipeline_config", "hydra_config_path", "depth_lb", "depth_ub", "fps", "query_interval_frames"),
            ("DSG 节点/边", "对象", "DAM correction", "处理耗时"),
            "旧离线 map 的输出路由和完成判定存在已知缺陷，不能视作严格语义提交证明。",
            True,
            1510,
            150,
        ),
    )
    edges = (
        E("prepare", "select"),
        E("select", "depth"),
        E("depth", "calibrate"),
        E("calibrate", "temporal", "branch"),
        E("calibrate", "odometry", "branch"),
        E("calibrate", "loops", "branch"),
        E("temporal", "optimize", "join"),
        E("odometry", "optimize", "join"),
        E("loops", "optimize", "join"),
        E("optimize", "filter"),
        E("filter", "validate", "gate"),
        E("filter", "fuse", "branch"),
        E("validate", "map", "gate"),
        E("fuse", "map", "gate"),
    )
    groups = (
        G(
            "run",
            "运行与输入",
            "本次运行的输入、输出和执行边界。默认启用 dry-run。",
            (
                P("src", "输入数据", "--src", "path", required=True, placeholder="/path/to/g1-or-prepared-stereo"),
                P("run_dir", "运行目录", "--run-dir", "path", "output/dashboard_offline", True),
                P("adapter", "输入适配器", "--adapter", "choice", "g1-fisheye", choices=("g1-fisheye", "prepared-stereo")),
                P("stop_after", "停止阶段", "--stop-after", "choice", "map", choices=OFFLINE_STAGES),
                P("dry_run", "仅生成计划", "--dry-run", "boolean", True, help="不启动 GPU 作业，先验证命令与路径。"),
                P("resume", "恢复已有运行", "--resume", "boolean", False),
                P("overwrite", "允许覆盖阶段产物", "--overwrite", "boolean", False, advanced=True),
            ),
        ),
        G(
            "prepare",
            "G1 双目准备",
            "Prepared Stereo 模式下部分参数不会生效。",
            (
                P("sequence", "序列号", "--sequence", default="000000"),
                P("max_delta_ms", "最大同步差 (ms)", "--max-delta-ms", "number", 10.0, minimum=0, step=0.1),
                P("horizontal_fov_deg", "水平 FOV (°)", "--horizontal-fov-deg", "number", 100.0),
                P("down_fov_deg", "向下 FOV (°)", "--down-fov-deg", "number", 28.0),
                P("stereo_calibration_report", "双目标定报告", "--stereo-calibration-report", "path", "", hard_gate=True),
                P("recommended_max_depth_m", "推荐最大深度 (m)", "--recommended-max-depth-m", "number", 5.0, minimum=0.1),
            ),
        ),
        G(
            "selection",
            "关键帧选择",
            "控制 pose 运动、视觉事件与 watchdog 保留策略。",
            (
                P("soft_translation_m", "软平移阈值 (m)", "--soft-translation-m", "number", 0.06, minimum=0),
                P("soft_rotation_deg", "软旋转阈值 (°)", "--soft-rotation-deg", "number", 5.0, minimum=0),
                P("hard_translation_m", "硬平移阈值 (m)", "--hard-translation-m", "number", 0.15, minimum=0),
                P("hard_rotation_deg", "硬旋转阈值 (°)", "--hard-rotation-deg", "number", 12.0, minimum=0),
                P("max_gap_s", "最大留帧间隔 (s)", "--max-gap-s", "number", 1.5, minimum=0),
            ),
        ),
        G(
            "depth",
            "FoundationStereo",
            "推理 profile、模型权重和许可门。",
            (
                P("checkpoint", "模型 checkpoint", "--checkpoint", "path", "", hard_gate=True),
                P("foundation_stereo_env", "Conda 环境", "--foundation-stereo-env", default="foundation_stereo", advanced=True),
                P("depth_profile", "深度 profile", "--depth-profile", "choice", "refine", choices=("online", "refine", "custom")),
                P("valid_iters", "迭代次数", "--valid-iters", "integer", 32, minimum=1),
                P("depth_scale", "推理分辨率比例", "--depth-scale", "number", None, minimum=0.05, maximum=1.0, advanced=True),
                P("depth_precision", "精度", "--depth-precision", "choice", "fp16", choices=("fp32", "fp16", "bf16")),
                P("max_depth_m", "深度上限 (m)", "--max-depth-m", "number", 5.0, minimum=0.1),
                P("accept_license", "确认 NVIDIA 研究/非商用许可", "--accept-foundation-stereo-noncommercial-license", "boolean", False, hard_gate=True),
            ),
        ),
        G(
            "geometry",
            "几何与全局优化",
            "地面标定、局部约束、闭环和优化。",
            (
                P("floor_calibration_report", "固定地面标定报告", "--floor-calibration-report", "path", "", hard_gate=True),
                P("geometry_max_depth_m", "几何最大深度 (m)", "--geometry-max-depth-m", "number", 5.0),
                P("local_keyframe_distance_m", "局部关键帧距离 (m)", "--local-keyframe-distance-m", "number", 0.10, advanced=True),
                P("local_max_keyframe_gap", "局部最大帧间隔", "--local-max-keyframe-gap", "integer", 8, advanced=True),
                P("local_neighbor_span", "邻域跨度", "--local-neighbor-span", "integer", 6, advanced=True),
                P("local_min_inliers", "最少 inliers", "--local-min-inliers", "integer", 80, advanced=True),
                P("loop_dense_candidate_count", "闭环稠密候选数", "--loop-dense-candidate-count", "integer", 80),
                P("max_loop_gravity_residual_deg", "闭环重力残差上限 (°)", "--max-loop-gravity-residual-deg", "number", 8.0),
                P("global_iterations", "全局优化迭代", "--global-iterations", "integer", 250),
            ),
        ),
        G(
            "quality",
            "时序过滤与质量门",
            "最终时序质量门和人工融合预览。",
            (
                P("temporal_filter_neighbor_offsets", "邻帧 offsets", "--temporal-filter-neighbor-offsets", default="1,2,3"),
                P("temporal_filter_scale", "过滤尺度", "--temporal-filter-scale", "number", 0.5),
                P("temporal_filter_min_judged", "最少判断邻帧", "--temporal-filter-min-judged", "integer", 3),
                P("temporal_filter_min_support", "最小支持率", "--temporal-filter-min-support", "number", 0.5),
                P("final_min_adjacent_agreement", "最小相邻一致率", "--final-min-adjacent-agreement", "number", 0.85),
                P("final_max_adjacent_median_error_m", "最大相邻中值误差 (m)", "--final-max-adjacent-median-error-m", "number", 0.035),
                P("final_min_window_adjacent_agreement", "最差窗口最小一致率", "--final-min-window-adjacent-agreement", "number", 0.80),
                P("fusion_frame_step", "融合帧步长", "--fusion-frame-step", "integer", 10),
                P("fusion_pixel_step", "融合像素步长", "--fusion-pixel-step", "integer", 8),
                P("fusion_voxel_size_m", "融合 voxel (m)", "--fusion-voxel-size-m", "number", 0.035),
                P("accept_preview", "已人工验收 RGB-D 预览", "--accept-direct-fusion-preview", "boolean", False, hard_gate=True),
            ),
        ),
        G(
            "mapping",
            "DAAAM / Hydra",
            "Pipeline YAML 与 Hydra YAML 是独立参数层。",
            (
                P("pipeline_config", "Pipeline 配置", "--pipeline-config", "path", "config/pipeline_config.yaml"),
                P("hydra_config_path", "Hydra 配置", "--hydra-config-path", "path", "config/hydra_g1_high_quality.yaml"),
                P("labelspace_path", "Labelspace", "--labelspace-path", "path", "", advanced=True),
                P("labelspace_colors", "Labelspace 颜色", "--labelspace-colors", "path", "", advanced=True),
                P("depth_lb", "对象深度下限 (m)", "--depth-lb", "number", 0.25),
                P("depth_ub", "对象深度上限 (m)", "--depth-ub", "number", 5.0),
                P("fps", "Pipeline FPS", "--fps", "number", 10.0),
                P("query_interval_frames", "语义查询间隔", "--query-interval-frames", "integer", 90),
            ),
        ),
    )
    presets = (
        Preset(
            "offline_g1",
            "G1 高质量离线",
            "完整 12-stage 流程；默认 dry-run，许可、标定和人工预览均需显式确认。",
            {"adapter": "g1-fisheye", "stop_after": "map", "dry_run": True},
        ),
        Preset(
            "offline_prepared",
            "Prepared Stereo 离线",
            "跳过鱼眼重投影和固定地面校准的输入准备部分。",
            {"adapter": "prepared-stereo", "stop_after": "fuse", "dry_run": True},
        ),
    )
    return WorkflowDefinition(
        "offline_hq",
        "高质量离线语义地图",
        "原始 G1 或 Prepared Stereo 经内容安全关键帧、深度、几何优化、质量门和 DAAAM/Hydra 构图。",
        "scripts/run_stereo_mapping.py",
        "run_dir",
        nodes,
        edges,
        groups,
        presets,
        (
            "离线 map 当前存在已知输出路由缺陷，mesh/DSG 文件存在也不等于 DAM 已严格 drain/commit。",
            "许可证、固定标定、verified loop 与人工融合验收均是硬门。",
        ),
    )


def _realtime_workflow() -> WorkflowDefinition:
    nodes = (
        N("scheduler", "回放与调度", "input", "按绝对时间读取 Prepared Stereo，并进入有界多速率队列。", ("tick_index", "RGB/stereo/depth/pose"), ("RealtimeEnvelope",), ("rate_hz", "queue_capacity", "checkpoint_interval_frames"), ("请求/派发/完成帧", "drop", "进度"), x=20, y=120),
        N("pose", "Pose 验证", "geometry", "验证数据集提供的 world_T_camera、时间间隔和固定协方差；不执行 VIO。", outputs=("PoseEstimate",), parameters=("pose_position_std_m", "pose_rotation_std_deg"), metrics=("processed", "最大相邻平移/旋转", "errors"), x=210, y=120),
        N("depth", "深度", "model", "读取预计算深度，或调用独立 FoundationStereo worker。", outputs=("DepthFrame", "generated_depth"), parameters=("depth_backend", "checkpoint", "depth_profile", "depth_valid_iters", "depth_scale"), metrics=("有效率", "LR/时序一致", "P95", "GPU/RSS"), x=390, y=120),
        N("dynamic", "运动与动态层", "dynamic", "估计类别无关运动 mask，维护动态实体、生命周期和轨迹。", outputs=("dynamic_masks", "unknown_masks", "dynamic entities"), parameters=("motion_analysis_width", "minimum_dynamic_pixels"), metrics=("dynamic/unknown 比例", "active/expired", "P95"), x=570, y=120),
        N("fusion", "静态深度隔离", "geometry", "清零 dynamic、unknown 和低置信像素，生成只供静态地图融合的深度。", outputs=("static_depth",), metrics=("污染率", "unknown 比例", "完成帧"), x=750, y=120),
        N("global", "Hydra / Global", "geometry", "维护 submap/path bookkeeping，并把静态 RGB-D 融合到 Hydra live pass。", outputs=("hydra_realtime", "canonical_paths"), parameters=("static_map_backend", "hydra_config_path", "submap_frames"), metrics=("submap/path", "Hydra nodes/edges", "P95"), x=930, y=120),
        N("semantic_frontend", "FastSAM + BotSort", "semantic", "与几何主链异步运行实例分割、跟踪和逐帧标签持久化。", inputs=("DepthFrame",), outputs=("label_frames", "MapMemory observations"), parameters=("semantic_mode", "semantic_config", "segmentation_rate_hz", "semantic_queue_capacity"), metrics=("segmentation calls", "detections", "tracked instances", "propagation"), x=570, y=330),
        N("dam", "MapMemory + DAM-3B", "semantic", "对满足真实观测门的实体生成开放词汇描述，并管理 correction、mesh 绑定和 ACK。", outputs=("semantic corrections", "map_memory.sqlite3"), parameters=("semantic_minimum_observations", "entity_merge_distance_m", "binding_center_distance_m", "binding_aabb_gap_m"), metrics=("prompt", "correction", "pending", "rejected_no_mesh", "worker health"), status_hint="worker ready、prompt catch-up、FIFO 与 correction drain 是硬门。", gate=True, x=780, y=330),
        N("postpass", "Exact-label Postpass", "semantic", "关闭 live Hydra 后，在隔离进程用全序列静态深度和精确帧标签重放最终语义 Hydra。", inputs=("static_depth", "label_frames"), outputs=("hydra_semantic_postpass.json", "final Hydra"), metrics=("expected/replayed", "label coverage", "missing frames"), status_hint="label_coverage 必须为 1.0。", gate=True, x=1110, y=230),
        N("commit", "DSG 绑定与提交", "artifact", "把 DAM correction 绑定到真实 object mesh，持久化、reload 并校验最终 DSG hash。", outputs=("mesh.ply", "dsg_with_mesh.json", "semantic_dsg_commit.json"), metrics=("applied/pending/unmapped", "hash verified", "rejected_no_mesh"), status_hint="pending/unmapped/error 必须为 0；rejected_no_mesh 单独告警。", gate=True, x=1300, y=230),
        N("quality", "最终质量门", "quality", "汇总 time、depth、pose、dynamic、runtime、map 和 semantic 硬门。", outputs=("quality_report.json", "realtime_run_report.json"), metrics=("PASS/WARN/FAIL", "hard failures", "authoritative"), gate=True, x=1490, y=230),
    )
    edges = (
        E("scheduler", "pose"), E("pose", "depth"), E("depth", "dynamic"), E("dynamic", "fusion"), E("fusion", "global"),
        E("depth", "semantic_frontend", "async", "异步旁路"), E("semantic_frontend", "dam", "async"),
        E("semantic_frontend", "postpass", "join", "精确帧标签"), E("global", "postpass", "join", "静态深度/几何"),
        E("dam", "commit", "join", "corrections"), E("postpass", "commit", "gate"), E("commit", "quality", "gate"),
    )
    groups = (
        G("run", "运行与输入", "Prepared Stereo 数据、运行目录与执行范围。", (
            P("dataset", "Prepared Stereo 数据集", "--dataset", "path", required=True, placeholder="/path/to/prepared-dataset"),
            P("run_dir", "运行目录", "--run-dir", "path", "output/dashboard_realtime", True),
            P("dry_run", "仅生成计划", "--dry-run", "boolean", True),
            P("resume", "恢复 checkpoint", "--resume", "boolean", False),
            P("overwrite", "覆盖运行目录", "--overwrite", "boolean", False, advanced=True),
            P("max_frames", "最多帧数", "--max-frames", "integer", None, minimum=1),
            P("stop_after", "主链终点", "--stop-after", "choice", "global", choices=REALTIME_STAGES),
        )),
        G("scheduler", "调度与 checkpoint", "墙钟派发不改变原始 sensor_time_ns。", (
            P("rate_hz", "最大派发率 (Hz)", "--rate-hz", "number", 1.0, minimum=0.01),
            P("no_throttle", "禁用墙钟限速", "--no-throttle", "boolean", False, advanced=True),
            P("queue_capacity", "主队列容量", "--queue-capacity", "integer", 8, minimum=1),
            P("stage_deadline_ms", "阶段 deadline (ms)", "--stage-deadline-ms", "number", None, minimum=0, advanced=True),
            P("drain_timeout_s", "主链排空超时 (s)", "--drain-timeout-s", "number", 30.0, minimum=0.1),
            P("checkpoint_interval_frames", "Checkpoint 间隔帧", "--checkpoint-interval-frames", "integer", 30, minimum=1),
            P("submap_frames", "子地图帧数", "--submap-frames", "integer", 30, minimum=1),
        )),
        G("pose_dynamic", "Pose 与动态隔离", "Pose 协方差、光流分析和动态像素门。", (
            P("pose_position_std_m", "Pose 位置 std (m)", "--pose-position-std-m", "number", 0.05, minimum=0),
            P("pose_rotation_std_deg", "Pose 旋转 std (°)", "--pose-rotation-std-deg", "number", 2.0, minimum=0),
            P("minimum_dynamic_pixels", "最少动态像素", "--minimum-dynamic-pixels", "integer", 40, minimum=1),
            P("motion_analysis_width", "运动分析宽度", "--motion-analysis-width", "integer", 160, minimum=48),
        )),
        G("depth", "深度后端", "预计算深度不会被 profile 参数重新生成。", (
            P("depth_backend", "深度后端", "--depth-backend", "choice", "precomputed", choices=("precomputed", "foundation-worker")),
            P("checkpoint", "FoundationStereo checkpoint", "--checkpoint", "path", ""),
            P("foundation_stereo_env", "Conda 环境", "--foundation-stereo-env", default="foundation_stereo", advanced=True),
            P("foundation_stereo_python", "Worker Python", "--foundation-stereo-python", "path", "", advanced=True),
            P("depth_profile", "深度 profile", "--depth-profile", "choice", "online", choices=("online", "refine", "custom")),
            P("depth_valid_iters", "迭代次数", "--depth-valid-iters", "integer", None, minimum=1),
            P("depth_scale", "推理分辨率比例", "--depth-scale", "number", None, minimum=0.05, maximum=1.0),
            P("depth_precision", "精度", "--depth-precision", "choice", "fp16", choices=("fp32", "fp16", "bf16")),
            P("depth_lr_interval", "左右验证间隔", "--depth-lr-interval", "integer", 3, minimum=1),
        )),
        G("hydra", "静态地图与 Hydra", "global 节点只有选择 hydra 才输出最终 mesh/DSG。", (
            P("static_map_backend", "静态地图后端", "--static-map-backend", "choice", "hydra", choices=("submaps", "hydra")),
            P("hydra_config_path", "Hydra 配置", "--hydra-config-path", "path", "config/hydra_g1_high_quality.yaml"),
            P("hydra_labelspace_path", "Hydra labelspace", "--hydra-labelspace-path", "path", "", advanced=True),
            P("hydra_labelspace_colors", "Hydra 颜色", "--hydra-labelspace-colors", "path", "", advanced=True),
        )),
        G("semantic", "FastSAM / BotSort / DAM", "语义旁路不阻塞几何，但启动与最终 drain 是同步硬门。", (
            P("semantic_mode", "语义模式", "--semantic-mode", "choice", "dam", choices=("disabled", "frontend", "dam")),
            P("semantic_config", "Pipeline 配置", "--semantic-config", "path", "config/pipeline_config_realtime.yaml"),
            P("segmentation_rate_hz", "FastSAM 上限 (Hz)", "--segmentation-rate-hz", "number", 5.0, minimum=0.01),
            P("semantic_frontend_rate_hz", "语义旁路上限 (Hz)", "--semantic-frontend-rate-hz", "number", 10.0, minimum=0.01),
            P("semantic_queue_capacity", "语义队列容量", "--semantic-queue-capacity", "integer", 2, minimum=1),
            P("semantic_minimum_observations", "DAM 最少真实观测", "--semantic-minimum-observations", "integer", 5, minimum=1),
            P("entity_merge_distance_m", "实体合并距离 (m)", "--entity-merge-distance-m", "number", 0.50, minimum=0),
            P("binding_center_distance_m", "Mesh 中心绑定距离 (m)", "--object-binding-maximum-center-distance-m", "number", 0.75, minimum=0),
            P("binding_aabb_gap_m", "Mesh AABB gap (m)", "--object-binding-maximum-aabb-gap-m", "number", 0.15, minimum=0),
            P("semantic_startup_timeout_s", "DAM 启动超时 (s)", "--semantic-startup-timeout-s", "number", 120.0),
            P("semantic_drain_timeout_s", "DAM 排空超时 (s)", "--semantic-drain-timeout-s", "number", 60.0),
            P("gpu_sharing_mode", "GPU 协调", "--gpu-sharing-mode", "choice", "staggered", choices=("staggered", "unmanaged")),
        )),
        G("quality", "质量与审计", "阈值来自独立质量 YAML。", (
            P("quality_config", "质量门配置", "--quality-config", "path", "config/realtime_quality_gates.yaml"),
            P("quality_report_only", "质量失败只报告", "--quality-report-only", "boolean", False, advanced=True),
            P("no_write_fusion_products", "不写逐帧融合证据", "--no-write-fusion-products", "boolean", False, advanced=True),
        )),
    )
    presets = (
        Preset("realtime_dam", "准实时语义 Hydra", "预计算深度、1 Hz、DAM、exact-label postpass 和最终质量门。", {"dry_run": True, "stop_after": "global", "static_map_backend": "hydra", "semantic_mode": "dam"}),
        Preset("realtime_geometry", "准实时静态几何", "不启动语义旁路，只验证动态隔离和 Hydra 几何。", {"dry_run": True, "stop_after": "global", "static_map_backend": "hydra", "semantic_mode": "disabled"}),
        Preset("tabletop_dam", "桌面小物体语义", "使用桌面 FastSAM/Hydra 配置及显式的紧实体合并/mesh 绑定距离。", {
            "dry_run": True, "stop_after": "global", "static_map_backend": "hydra", "semantic_mode": "dam",
            "semantic_config": "config/pipeline_config_tabletop.yaml", "hydra_config_path": "config/hydra_g1_tabletop.yaml",
            "entity_merge_distance_m": 0.075, "binding_center_distance_m": 0.10, "binding_aabb_gap_m": 0.025,
        }),
    )
    return WorkflowDefinition(
        "realtime_semantic",
        "准实时动态语义地图",
        "有界几何主链、类别无关动态隔离、异步 FastSAM/BotSort/DAM 与 exact-label Hydra postpass。",
        "scripts/run_realtime_mapping.py",
        "run_dir",
        nodes,
        edges,
        groups,
        presets,
        (
            "当前入口回放已准备的数据集，不直接订阅 ROS 相机/IMU，也不执行在线 VIO。",
            "最终语义地图必须同时通过 postpass、DAM drain/ACK、durable commit 与 hard quality gates。",
        ),
    )


def _query_workflow() -> WorkflowDefinition:
    nodes = (
        N("final_dsg", "最终语义 DSG", "input", "来自通过严格语义提交与质量门的最终 dsg_with_mesh.json。", x=20, y=100),
        N("rebind", "语义 Rebind（可选）", "semantic", "把语义迁移到另一份候选几何，并保留 spatial-only 实体。", outputs=("dsg_rebound.json", "binding audit"), x=230, y=100),
        N("embeddings", "多语言 Embeddings", "model", "生成 checksum-bound 多语言对象向量和 manifest。", outputs=("dsg_updated.json", "manifest", "semantic sidecar"), x=450, y=100),
        N("evidence", "FastSAM / RGB-D 证据", "artifact", "从真实 FastSAM 调用帧生成 overlay、cutout 和可选 3D mask cloud。", outputs=("evidence manifest", "overlays", "point clouds"), x=680, y=100),
        N("query", "中英文查询", "service", "按相同 embedding 模型执行本地 open-set 检索，或启动 query API。", outputs=("matches", "grounded answer"), x=910, y=100),
    )
    return WorkflowDefinition(
        "query_assets",
        "可查询地图后处理",
        "最终语义 DSG 到 rebind、checksum-bound embeddings、真实图像证据与查询服务。首版作为只读流程参考。",
        None,
        None,
        nodes,
        (E("final_dsg", "rebind", "optional"), E("rebind", "embeddings"), E("final_dsg", "embeddings", "optional", "无需 rebind"), E("embeddings", "evidence", "optional"), E("embeddings", "query"), E("evidence", "query")),
        warnings=("这些节点不是两个主 runner 的内置 stage；首版面板只展示已有产物状态，不会把它们伪装成一键闭合流程。",),
    )


WORKFLOWS = {
    workflow.id: workflow
    for workflow in (_offline_workflow(), _realtime_workflow(), _query_workflow())
}

DEFAULT_WORKFLOW_ID = "realtime_semantic"


def get_workflow(workflow_id: str) -> WorkflowDefinition:
    try:
        return WORKFLOWS[workflow_id]
    except KeyError as error:
        raise KeyError(f"Unknown workflow: {workflow_id}") from error


def list_workflows() -> tuple[WorkflowDefinition, ...]:
    return tuple(WORKFLOWS.values())
