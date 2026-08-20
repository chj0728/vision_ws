# Pipelines 更新记录

## IoU 追踪与 ArcFace 人脸识别模块拆分 - 2026-08-20

- iou_tracker.hpp
- iou_tracker.cpp
  - 仅负责人体框 IoU 关联。
  - 支持轨迹老化、死亡轨迹短期恢复及过期清理。
  - 写入 PersonMeta.track_id。
  - 不保存人脸识别业务状态。
  
- arcface_pipeline.hpp
- arcface_pipeline.cpp
  - 使用 SCRFD 五点关键点进行标准人脸对齐。
  - 实现人脸质量门控、Embedding 缓冲、数据库识别、自动注册和周期重验。
  - 按 track_id 独立维护识别状态。
  - 写入 FaceRecog 的 UUID、姓名、相似度和 512 维特征。
  - 预留 updatePersonName() 接口，后续可接入 ROS 服务。
- perception_frame_context.hpp
  - 只在单帧 Pipeline 内传递轨迹信息及 SCRFD 五点关键点。
  - 不需要修改 ROS 消息定义，也不会把底层 FaceObject 暴露到 ROS 接口。

```bash
YOLOPipeline
    ↓ 人体框
IouTracker
    ↓ track_id / track_total_frames
SCRFDPipeline
    ↓ 人脸框 / 五点关键点
ArcFacePipeline
    ↓ UUID / 姓名 / 相似度 / Embedding
PerceptionResult
```

## 新增 SCRFD 模块 - 2026-08-19

- 新增 scrfd_pipeline.hpp 和 scrfd_pipeline.cpp。
- 参考旧节点实现人体头肩 ROI 裁剪、SCRFD 推理、最高置信度人脸选择及全图坐标回写。
- 检测结果写入 PersonMeta.face_detection。
- 支持引擎路径、预处理方式、置信度/NMS 阈值、ROI 比例和最大处理人数配置。
- 在 perception_pipeline.cpp 中按 YOLO → SCRFD 顺序执行。
