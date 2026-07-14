# DAAAM 语义地图构建模型介绍

本次 CODa 语义地图并不是由单一模型端到端生成，而是通过深度估计、
实例分割、跨帧跟踪、开放词汇描述、三维几何融合和区域聚类等多个环节
协同构建。

## 模型与模块

| 环节 | 实际使用 | 作用 |
| --- | --- | --- |
| 深度 | FoundationStereo | 提供每帧深度；结果已预先写入 CODa rosbag |
| 实例分割 | FastSAM-X（TensorRT） | 从 RGB 图像中生成实例掩码 |
| 跨帧跟踪 | BoT-SORT + CLIP ReID `clip_general` | 将不同帧中的同一物体关联为连续轨迹 |
| 语义描述 | NVIDIA DAM-3B | 为物体生成开放词汇自然语言描述 |
| 关键帧特征 | PE-Core-L14-336 | 选择代表帧并提取 1024 维 CLIP 特征 |
| 三维建图 | Hydra + Khronos | 融合 RGB-D、相机位姿和实例掩码，构建三维网格与 Dynamic Scene Graph |
| 语义后处理 | `sentence-transformers/sentence-t5-xl` | 将物体描述转换为 768 维句子嵌入 |
| 房间生成 | Hydra RoomFinder | 根据空间连通性和语义相似度聚类房间 |

## 构建流程

```text
CODa RGB + FoundationStereo 深度 + 相机位姿
                    ↓
              FastSAM 实例分割
                    ↓
        BoT-SORT + CLIP ReID 跨帧跟踪
                    ↓
           DAM-3B 生成物体语义描述
                    ↓
   Hydra/Khronos 融合三维网格和场景图
                    ↓
       sentence-t5-xl 生成语义嵌入
                    ↓
       空间连通性与语义相似度聚类
                    ↓
                 房间层
```

## 各模型的核心职责

- **FastSAM-X** 确定图像中的物体区域，但不负责最终语义命名。
- **BoT-SORT 与 CLIP ReID** 维持物体在时间上的一致身份，避免将每一帧的
  掩码都视为新物体。
- **DAM-3B** 是主要的开放词汇语义模型，负责回答物体是什么以及它具有
  哪些外观属性。
- **PE-Core-L14-336** 提取视觉语义特征，辅助选择具有代表性的观测帧和
  后续语义匹配。
- **Hydra/Khronos** 主要执行几何融合、网格生成和场景图维护，并不是一个
  单独的端到端神经网络。
- **sentence-t5-xl** 将 DAM-3B 生成的描述编码成句子向量，为语义相似度
  计算和区域聚类提供特征。

## 本次运行说明

本次复现实际使用 `dam_multi_image` grounding worker 和本地
`nvidia/DAM-3B` 模型。配置中虽然保留了 `gpt-4.1` 字段，但该字段属于未被
当前 worker 使用的配置项；主流程没有调用 GPT-4.1 或其他 OpenAI 在线模型。

完整运行最终生成了 5310 个场景图节点、8313 条边和 8 个房间。精确模型
参数可参见
[`pipeline_config.yaml`](output/coda/out_20260713_103618/pipeline_config.yaml)，
复现过程和环境记录参见 [`REPRODUCTION_CODA.md`](REPRODUCTION_CODA.md)。
