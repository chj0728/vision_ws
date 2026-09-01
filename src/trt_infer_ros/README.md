# Trt_infer_ros 更新记录

## PerceptionRosComponent 图像保存服务 - 2026-09-01

- 新增三个 `std_srvs/srv/Trigger` 服务，用于保存最近一次完成感知处理的图像快照：

```bash
ros2 service call /save_color_depth std_srvs/srv/Trigger {}
ros2 service call /save_color_bbox std_srvs/srv/Trigger {}
ros2 service call /save_all_images std_srvs/srv/Trigger {}
```

- `/save_color_depth` 保存 `color.png` 和 `depth.png`。
- `/save_color_bbox` 保存包含人体、人脸框和头姿可视化的 `color_bbox.png`。
- `/save_all_images` 同时保存 `color.png`、`depth.png` 和 `color_bbox.png`。
- 每次成功调用都会创建目录 `/home/caohaojie/ws/vision_ws/logs/images/YYYY-MM-DD_HH-MM-SS/`；服务响应的 `message` 字段返回实际保存目录。
- 深度图以毫米为单位保存为 `uint16` 单通道 PNG；服务在尚未获得完整处理帧时返回失败信息。

## ScenePerceptionResult 旧话题兼容 - 2026-08-31

- `PerceptionRosComponent` 新增发布 `ScenePerceptionResult`，默认话题为 `/human_face_fusion/scene_perception`，可通过 `config/ros.yaml` 中的 `engagement_result_topic` 修改。
- 兼容消息复用当前帧的 `PerceptionResult` 和 `InteractionResult`：旧版 `ENGAGED` 对应当前 `TALKING_STATUS`，旧版 `ATTENTION` 对应当前 `ATTENTION_STATUS`。
- 每个人的人体、人脸、头姿和 ArcFace 识别结果均映射至 `PersonPerception`；当前流水线不提供性别结果，因此保持 `GENDER_UNKNOWN`。

## PerceptionRosComponent 图像处理流程优化 - 2026-08-28

- 新增 `processDecodedColorDepth()`，统一处理已经解码完成的 BGR 彩色图和米制浮点深度图。
- 感知 Pipeline 调用、结果发布、交互状态计算、边界框绘制和结果图像发布集中到同一处理路径，减少原始图像和压缩图像分支中的重复代码。
- 压缩图像处理不再执行 `cv::Mat → sensor_msgs/msg/Image → cv::Mat` 的重复转换。
- 彩色压缩图像解码为 BGR `cv::Mat` 后直接进入感知处理流程。
- `16UC1; compressedDepth png` 深度图解码后直接按 `0.001` 比例转换为米制 `CV_32FC1`。
- `32FC1; compressedDepth png` 深度图完成逆深度恢复后直接输出米制 `CV_32FC1`。
- 原始图像分支只负责将 ROS 图像消息转换为 OpenCV 矩阵，然后复用相同的 `processDecodedColorDepth()` 处理流程。
- 对已经是 `BGR8` 的彩色图和 `32FC1` 的深度图保留共享数据视图，避免不必要的图像复制。
- 保留 `cv_bridge::CvImageConstPtr` 的有效生命周期，避免转换生成的 `cv::Mat` 引用已经释放的临时数据。
- `perception_common.hpp` 补充压缩图像解码及深度单位转换所需的标准库头文件，并移除组件中废弃的深度缓存成员和注释代码。
- `CMakeLists.txt` 和 `package.xml` 新增显式的 `std_msgs` 依赖，用于共用处理入口传递图像消息头。

优化后的处理流程：

```text
CompressedImage ──解码──┐
                       ├─> BGR cv::Mat + CV_32FC1 depth ─> processDecodedColorDepth() ─> PerceptionPipeline
sensor_msgs/Image ─转换─┘
```

## PerceptionRosComponent 压缩图像订阅 - 2026-08-26

- `perception_ros_component` 新增压缩 RGB-D 图像订阅模式。
- 支持同步订阅 `sensor_msgs/msg/CompressedImage` 类型的彩色图像和深度图像。
- 彩色压缩图像支持通过 OpenCV 解码 JPEG、PNG 等格式。
- 深度压缩图像支持 `16UC1; compressedDepth png` 和 `32FC1; compressedDepth png` 格式，并统一转换为感知 Pipeline 使用的米制浮点深度图。
- 保留原始 `sensor_msgs/msg/Image` 订阅模式，以及精确同步和近似同步策略。
- 在 `config/ros.yaml` 中新增以下参数：

```yaml
use_compressed_images: false
color_compressed_topic: /camera/color/image_raw/compressed
depth_compressed_topic: /camera/depth/image_raw/compressedDepth
```

- 当 `use_compressed_images` 为 `false` 时，订阅 `color_image_topic` 和 `depth_image_topic` 指定的原始图像话题。
- 当 `use_compressed_images` 为 `true` 时，仅订阅 `color_compressed_topic` 和 `depth_compressed_topic` 指定的压缩图像话题。

## SixDRepNet 头部姿态模块迁移 - 2026-08-21

- 新增 sixdrepnet_pipeline.hpp 和 sixdrepnet_pipeline.cpp。
- 消费 SCRFD 输出的全图人脸框，按配置比例扩展并裁剪人脸 ROI。
- 对满足最小尺寸要求的人脸执行 SixDRepNet 推理，将 yaw、pitch、roll 写入 PersonMeta.head_pose。
- 未启用、无人脸、ROI 尺寸不足或推理失败时，使用 yaw=999、pitch=999、roll=0 表示无效头姿。
- 支持模块开关、引擎文件名、引擎路径、最小人脸尺寸和人脸框扩展比例配置。
- 在总 Pipeline 中按 SCRFD → SixDRepNet → ArcFace 顺序执行，使 ArcFace 可使用当前帧头姿进行质量门控。
- 更新 CMakeLists.txt 和 pipeline.yaml，默认从 models/sixdrepnet 目录加载模型，模块默认关闭。
- `/perception/color_bbox` 图像会在人脸框中心绘制白色头部姿态立方体和局部坐标轴，用于直观观察 SixDRepNet 输出。

### 姿态可视化与坐标约定

图像坐标系遵循相机光学坐标约定：X 轴向右、Y 轴向下、Z 轴指向图像内部。姿态框的局部坐标轴颜色与 RViz TF 保持一致：X 为红色、Y 为绿色、Z 为蓝色。

根据实际测试，以现实世界中人的第一视角为参考：

- yaw：绕 Y 轴旋转，表示左右转头；正值向左转，负值向右转。
- pitch：绕 X 轴旋转，表示上下点头；正值向上抬头，负值向下低头。
- roll：绕 Z 轴旋转，表示左右侧倾；正值向右侧倾，负值向左侧倾。

  `yaw` > 0：人向左转头
  `pitch` > 0：人向上抬头
  `roll` > 0：人向右侧倾

`drawHeadPoseBox()` 按 $R=R_z(roll)R_y(yaw)R_x(pitch)$ 重建 SixDRepNet 输出的旋转矩阵，不额外改变角度符号。三个角度会组合生效，彩色箭头显示的是旋转后的局部坐标轴，其投影方向不应直接解释为人的转头或点头方向。

## Pipeline 执行顺序更新 - 2026-08-21

```bash
YOLOPipeline
  ↓ 人体框
IouTracker
  ↓ track_id / track_total_frames
SCRFDPipeline
  ↓ 人脸框 / 五点关键点
SixDRepNetPipeline
  ↓ yaw / pitch / roll
ArcFacePipeline
  ↓ UUID / 姓名 / 相似度 / Embedding
PerceptionResult
```

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
