**Q**：如果要将此项目做成一个实时动态语义地图构建的项目，还需要进行那些方面的优化

目前项目更接近“离线语义建图原型”，距离“实时动态语义地图”还有三个核心缺口：**吞吐不足、动态物体没有与静态地图分层、位姿与闭环仍不可靠**。

以 G1 本次实测为基准：

- FoundationStereo：约 `1.225 s/帧`，`0.82 FPS`。
- DAAAM CV：约 `0.238 s/帧`，`4.19 FPS`。
- Hydra：约 `0.101 s/帧`，`9.89 FPS`。
- 三部分串行约 `0.64 FPS`，距离 `10 Hz` 输入约有 **15.6 倍吞吐差距**。

数据见 [FoundationStereo 运行报告](/home/user/Code/DAAAM_Origin/.repro/datasets/20260713_170500-pinhole-sync10ms/foundation_stereo_run.json) 和 [DAAAM 运行统计](/home/user/Code/DAAAM_Origin/output/g1_20260713_170500_pinhole_smooth3d_full/out_20260715_140849/processing_stats.json)。

**P0：必须先解决**

1. **静态地图与动态对象分层**

当前 `dynamic_labels` 为空，主要还是把所有有效深度向 Hydra 静态地图融合。需要增加：

- 通过“相机运动预测光流”和实际光流/场景流的残差判断物体是否运动，不能只按类别判断。
- 动态掩码不得融合进静态 TSDF。
- 单独维护对象层：`track_id、语义概率、3D位置、速度、尺寸、轨迹、协方差、最后观测时间`。
- 物体移走后，要支持旧几何删除、反积分或时间衰减。
- 物体停止运动后经过连续稳定确认，才能重新进入静态地图。

尤其需要区分：**“pose 不变但图像改变”的帧必须保留给语义和动态跟踪，但不一定应直接融合进静态地图。**

2. **在线位姿、闭环与地图修正**

本次 Hydra 输出没有有效闭环，[loop_closures.csv](/home/user/Code/DAAAM_Origin/output/g1_20260713_170500_pinhole_smooth3d_full/out_20260715_140849/hydra_output/backend/loop_closures.csv) 只有表头。此前 Mesh 也有约 1969 个连通分量，说明这首先是全局一致性问题。

需要：

- 接入实时 stereo VIO、视觉惯性或激光惯性 SLAM，机器人 odom 只作为先验。
- 每个位姿携带 `sensor_time_ns` 和协方差。
- 增加几何验证闭环，不能只依赖外观检索。
- 使用局部子地图；发生闭环修正时重定位子地图、形变或重新积分。
- 在线校验相机、IMU、pose 的时钟漂移和 TF 外参。

3. **把串行循环改为多频率流水线**

当前主循环是 DAAAM 完成后再调用 Hydra，属于串行处理，[runner.py](/home/user/Code/DAAAM_Origin/src/daaam/hydra/runner.py:329)。建议拆成有界队列：

```text
双目/IMU -> 时间对齐 -> 在线位姿
                    |
                    +-> 关键帧/视觉事件筛选
                         |-> 深度
                         |-> 分割/跟踪
                         +-> 动静态判别
                                  |-> 静态子地图
                                  |-> 动态对象图
                                       -> Hydra/语义场景图
```

所有异步结果用 `(sensor_time_ns, track_id, map_revision)` 回填，过期结果不得覆盖新状态。队列过载时优先保留 `pose_motion` 和 `image_event_at_static_pose`，合并普通地图更新，不能无限积压旧帧。

4. **FoundationStereo 实时化**

当前脚本以单帧、全尺寸、`32 iterations` 运行，[run_foundation_stereo_depth.py](/home/user/Code/DAAAM_Origin/scripts/run_foundation_stereo_depth.py:128)。建议保留 FoundationStereo，但分成两种模式：

- 在线模式：关键帧运行，缩小分辨率，测试 `8/16 iterations`，FP16、TensorRT/ONNX 或 `torch.compile`。
- 精修模式：使用原始分辨率和 32 次迭代，对重要关键帧异步更新。
- 输出深度置信度和左右一致性，而不只是 `uint16` 深度 PNG。
- 非关键帧通过光流、位姿和上一帧深度传播。
- 鱼眼矫正映射预计算，避免运行时重复生成。

目标应先达到深度和融合 `3–5 Hz`，不必要求每张相机图像都跑 FoundationStereo。

**P1：语义和跟踪**

- FastSAM 已使用 TensorRT，但目前分割和跟踪仍在主路径串行执行，[orchestrator.py](/home/user/Code/DAAAM_Origin/src/daaam/pipeline/orchestrator.py:423)。
- 建议分割 `3–5 Hz`，2D/3D 跟踪 `10 Hz`，中间帧传播 mask。
- ReID 只在遮挡恢复、轨迹冲突或重新出现时执行。
- 从“2D mask + 中值深度中心点”升级为 3D mask 点云、3D包围盒和运动模型。
- DAM/VLM grounding 保持异步，可延迟数秒，但不能阻塞几何地图。
- 当前配置将 DSG 修正延迟到退出时处理，[pipeline_config.yaml](/home/user/Code/DAAAM_Origin/config/pipeline_config.yaml:65)，实时版本必须改成增量、版本化更新。
- FoundationStereo 和 DAM-3B 会争用显存，最好分 GPU；单 GPU时要设置优先级、显存上限和模型卸载策略。

##  **可编辑记忆**
- 对构建的地图进行个性化编辑、命名等，比如将某个区域定义为零食区，某个桌子定义为餐桌，个性化编辑之后，应该保持其名称不变，后期更新时也保留其名词或者构建别名


## **可更新记忆**
- 不同时间、不同起始位置构建的地图如何更新，如何配准

## **制定地图构建过程中的质检规则**
- 地图构建过程的每一步都严格建立质检规则，对结果进行评价，不通过则别着急进入下个流程

## **路径合并与更新机制**
- 同一个路径，来回采集，合并为一个路径，或者定义为同一个真实回访簇。如，可视化确认是同一张桌子、货架和墙面从不同位置再次观测。机器人在屋子里来回闲逛，会走很多重复的路径。
**建议实时指标**

| 模块 | 建议频率/延迟 |
|---|---|
| 位姿与 TF | 20–50 Hz，P95 `<30 ms` |
| 2D/3D 跟踪 | 10 Hz，P95 `<50 ms` |
| 分割 | 3–5 Hz或事件触发 |
| 深度与局部融合 | 3–5 Hz，P95 `<250 ms` |
| 动态对象地图发布 | 10 Hz |
| VLM 语义修正 | 异步，延迟 `<2–5 s` |

最终还需加入实时回放、GPU过载、丢帧、pose延迟、动态物体移入移出等测试，并监控 P50/P95/P99 延迟、队列年龄、丢帧原因、ATE/RPE、Mesh连通性、IDF1/HOTA 和动态几何污染率。

建议实施顺序是：**在线位姿与闭环 → 多频率调度 → FoundationStereo关键帧化 → 动静态分层 → 3D对象跟踪 → 在线语义修正**。另外 FoundationStereo 当前许可证限制为研究/非商用，[LICENSE](/home/user/Code/DAAAM_Origin/third_party/FoundationStereo/LICENSE)，若后续产品化需要更换后端或单独取得授权。
