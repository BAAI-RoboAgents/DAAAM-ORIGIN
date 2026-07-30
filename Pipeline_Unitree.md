# Unitree 头部 RGB-D 语义地图 Pipeline

本文定义 `daaam_g1_hardware_v1` 采集布局中，以头部 D455（`cam0`）为主相机的
严格建图流程。它不把 `cam0/cam1` 当作双目：两者分别是头部 D455 和胸部 D435i，
不是一个经过联合标定的左右目相机对。

本流程遵守一个原则：采集中没有明确记录或没有验证来源的量，不使用型号默认值、
经验安装尺寸、坐标系别名或视觉观感补齐。

## 1. 本批数据的已验证状态

数据集：

```text
/home/user/datasets/UniTree_daaam_20260727_174310
```

权威审计报告：

```text
output/unitree_20260727_174310_audit/unitree_data_audit.json
```

当前结论为 `status=blocked`、`mapping_ready=false`。这里的 `blocked` 表示建图契约
缺失，不表示已保存的数据载荷损坏。

### 1.1 已存在且已验证

| 数据 | 状态 |
| --- | --- |
| manifest/quality/逐帧索引 | 214 条，tick 为 `0..213`，计数一致 |
| 头部 RGB | 214/214 可解码，`640x480`、`uint8`、214 个唯一文件 |
| 头部深度 | 214/214 可读取，`640x480`、`float32`、与 RGB 尺寸一致 |
| 头部深度有效率 | 每帧 `80.27%..90.05%`，均值 `87.29%`；与 manifest 逐帧记录一致 |
| LiDAR | quality report、manifest 和文件计数均为 214 |
| 机器人轨迹 | 214 条严格递增的 `local_T_body` |
| 文件绑定 | RGB、深度、LiDAR、odom、aux pose 与 tick 一一对应 |
| 相机/odom 时间序列 | 各自严格递增 |

数据集中有 1103 个 `._*` AppleDouble 元数据文件。它们不是传感器帧，审计和转换会
忽略，不计为缺帧。

### 1.2 存在，但当前数据不能独立证明

| 项目 | 已有证据 | 仍不能证明的部分 |
| --- | --- | --- |
| RGB-D 像素对齐 | capture metadata 声明 `cam0_depth_aligned_to_rgb=true`，RGB/深度尺寸一致 | 没有带身份和验证来源的 RGB-depth registration 标定 |
| 头部相机内参 | `fx=fy=386, cx=320, cy=240, 640x480` | 没有畸变模型/系数、相机序列号、标定时间和验证 provenance |
| 相机时间 | `timestamp_ns` 和 `host_ns` 都严格递增 | 没有证明相机与 odom 使用同一时基 |
| RGB/深度同步 | 同一 tick 下各有一份 RGB 和深度 | 深度记录没有独立时间戳，也没有“同次采集”的验证契约 |
| 深度 `65.535 m` | 2 帧共 97 个像素达到该值 | 没有说明它是有效测量、饱和值还是无效哨兵 |

相机 sensor time 相对同 tick odom time 的差为 `4.826..195.096 ms`，中位数
`76.873 ms`；host time 的差为 `48.720..202.855 ms`。这些只是观测值，不能据此选择
哪个时间字段，也不能据此估计固定时钟偏移。

### 1.3 明确缺失

以下两项是当前硬阻断：

1. 完整、已验证的头部 D455 标定 sidecar：
   `body_T_head_D455_color_optical_frame`、针孔/零畸变契约、RGB-D 对齐证据、深度
   有效/无效值策略、相机身份和标定 provenance。
2. 已验证的 RGB-D 时间 sidecar：
   使用 sensor time 或 host time、相机与 odom 的共享时基证据、RGB/深度同次采集证据、
   pose 插值最大间隔和无 pose bracket 帧的明确处理策略。

此外，采集中只有 `local_T_body`，没有 `map_T_body`。因此当前流程只能构建以 `local`
为世界坐标系的局部语义地图。若交付要求世界坐标系必须叫 `map`，需要另行提供真实
`map_T_body`；不能把 `local` 改名成 `map`。

采集目标为 10 Hz，实际估计为 5.531 Hz；355 次尝试中接受 214 帧、跳过 141 次
（39.72%）。接受的 214 帧内部完整，但本次采集没有达到目标采样率。

## 2. 新 Pipeline

```text
原始 Unitree capture
  -> 只读数据审计
  -> 头部 D455 标定硬门
  -> RGB-D/odom 时间契约硬门
  -> 在明确相机时间上插值 local_T_body
  -> local_T_camera = local_T_body @ body_T_camera
  -> RGB 原字节复制 + float32 m 深度转 uint16 mm
  -> RGB-D 有效率与时序一致性质量门
  -> 类别无关运动/不确定区域隔离
  -> Hydra 静态几何融合
  -> FastSAM + BotSort + DAM
  -> exact-frame semantic postpass
  -> 静态语义 Hydra DSG
```

入口：

```text
scripts/run_unitree_mapping.py
```

底层审计/转换器：

```text
scripts/prepare_unitree_head_rgbd_dataset.py
```

Unitree RGB-D 质量门：

```text
config/unitree_rgbd_quality_gates.yaml
```

该质量门不要求左右一致性，因为输入是 D455 已对齐 RGB-D，不是双目模型输出；它仍要求
深度有效率、跨帧时序一致性、pose 连续性、动态隔离、运行时、网格和语义提交通过。

## 3. 两份必需 sidecar

下面仅说明字段结构。`null` 必须替换为真实测量/验证结果；转换器不会接受模板值。

### 3.1 头部 D455 标定

```json
{
  "schema": "unitree_head_rgbd_calibration_v1",
  "sensor": "HEAD_D455",
  "target_frame": "body",
  "source_frame": null,
  "target_T_camera": null,
  "intrinsics": {
    "model": "pinhole",
    "fx": null,
    "fy": null,
    "cx": null,
    "cy": null,
    "width": 640,
    "height": 480,
    "distortion": {
      "model": "none",
      "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0]
    }
  },
  "depth": {
    "aligned_to_color": true,
    "unit": "meter",
    "minimum_valid_depth_m": null,
    "maximum_valid_depth_m": null,
    "invalid_values_m": null
  },
  "provenance": {
    "validated": true,
    "method": null,
    "source": null,
    "timestamp": null
  }
}
```

`target_T_camera` 必须是 4x4 的 `body_T_camera`。本流程不接受
`base_link_T_camera` 后自动假定 `base_link == body`。

### 3.2 时间契约

```json
{
  "schema": "unitree_rgbd_time_contract_v1",
  "camera": "cam0",
  "camera_time_source": null,
  "shared_timebase_verified": true,
  "rgb_depth_same_capture_verified": true,
  "verification_method": null,
  "maximum_pose_interpolation_gap_ms": null,
  "allow_drop_unbracketed_frames": false,
  "maximum_dropped_frames": 0
}
```

`camera_time_source` 只能显式选择 `sensor` 或 `host`。如果允许丢弃无法被 odom 时间
包围的帧，必须把 `allow_drop_unbracketed_frames` 改为 `true`，并明确
`maximum_dropped_frames`；转换报告会保存被丢弃的源 tick。

## 4. 运行

### 4.1 当前可执行：审计

```bash
python scripts/run_unitree_mapping.py \
  --src /home/user/datasets/UniTree_daaam_20260727_174310 \
  --run-dir output/unitree_20260727_174310 \
  --stop-after audit
```

### 4.2 补齐 sidecar 后：准备 RGB-D

```bash
python scripts/run_unitree_mapping.py \
  --src /home/user/datasets/UniTree_daaam_20260727_174310 \
  --run-dir output/unitree_20260727_174310 \
  --camera-calibration /path/to/validated_head_d455_calibration.json \
  --time-contract /path/to/validated_rgbd_time_contract.json \
  --stop-after prepare
```

准备产物为：

```text
01_prepared_head_rgbd/
  rgb/
  depth/
  depth_confidence/
  depth_metadata/
  pose/poses.txt
  pose/pose_timestamps_ns.txt
  camera_info.json
  tick_index.json
  source_audit.json
  rgbd_preparation_report.json
```

RGB 文件按字节复制并校验 SHA-256；不会重映射、缩放或重新编码。深度根据 sidecar 中
明确的有效范围/无效值策略转成 `uint16` 毫米。

### 4.3 正式语义地图

```bash
python scripts/run_unitree_mapping.py \
  --src /home/user/datasets/UniTree_daaam_20260727_174310 \
  --run-dir output/unitree_20260727_174310 \
  --camera-calibration /path/to/validated_head_d455_calibration.json \
  --time-contract /path/to/validated_rgbd_time_contract.json \
  --rate-hz <已确认的单次正式调度频率> \
  --accept-local-world-frame \
  --stop-after map
```

`--accept-local-world-frame` 是显式确认“输出地图使用采集中的 `local` 坐标系”，不是把
它解释成或重命名为 `map`。`--rate-hz` 没有默认猜测值，也不用于 1/3/5 Hz A/B。

## 5. 完成判定

只有同时满足以下条件才称为完成：

1. `unitree_data_audit.json` 为 `mapping_ready=true`；
2. `rgbd_preparation_report.json` 为 `status=complete`，所有保留帧有严格时间和
   `local_T_camera`；
3. realtime checkpoint 覆盖所有请求帧，0 error，drop 符合显式质量门；
4. RGB-D 深度有效率与时序一致性通过；
5. Hydra mesh/DSG 生成；
6. DAM drain、correction ACK 和 exact-frame semantic postpass 完成；
7. 最终 `quality_report.json` 没有 hard failure。

当前数据只完成了第 1 项中的载荷审计，因两份 sidecar 缺失，第 1 项总体仍未通过。
