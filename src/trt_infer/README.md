# trt_infer

基于 TensorRT、CUDA 和 OpenCV 的视觉推理组件集合，包含人脸分析与目标检测能力。

## 模块说明

| 模块 | 主要类/数据结构 | 功能 |
| --- | --- | --- |
| `scrfd_trt` | `SCRFD_TRT`、`FaceObject` | 使用 SCRFD 检测图像中的人脸，输出人脸框、5 点关键点和置信度；支持 InsightFace 与 Namdvt 两种预处理方式。 |
| `arcface_trt` | `ArcFaceTRT`、`FaceEmbedding` | 对基于关键点对齐后的人脸提取 512 维 L2 归一化特征；提供人脸对齐、特征归一化和余弦相似度计算。 |
| `arcface_trt` | `FaceDatabase`、`PersonRecord`、`FaceMatch` | 使用 SQLite 管理人脸库；加载特征至内存进行快速比对，并支持人员注册、命名、识别和访问记录更新。 |
| `gender_trt` | `GenderAgeTRT`、`GenderResult` | 基于对齐人脸预测性别、置信度和年龄；兼容常见的多种 TensorRT 输出布局。 |
| `sixdrepnet_trt` | `SixDRepNet_TRT`、`HeadPose` | 估计人脸偏航、俯仰和滚转角；支持在图像中绘制头部姿态坐标轴。 |
| `yolo_trt` | `YOLOEngine`、`Detection` | 执行 YOLO 目标检测，输出类别、置信度和边界框；支持 GPU 预处理、NMS 后处理及深度图距离采样。 |

## 人脸分析流程

1. `SCRFD_TRT::detect` 检测人脸，得到 `FaceObject`。
2. `ArcFaceTRT::alignFace` 使用 5 点关键点生成人脸对齐图。
3. `ArcFaceTRT::extractEmbedding` 提取 512 维人脸特征。
4. `FaceDatabase::identify` 按余弦相似度完成人员识别；未识别人员可通过 `registerPerson` 入库。
5. `GenderAgeTRT::predict` 与 `SixDRepNet_TRT::predict` 可复用对齐人脸，分别生成年龄性别和头部姿态结果。

## 目标检测与距离估计

`YOLOEngine::infer` 完成普通目标检测。`YOLOEngine::inferWithDepth` 在同一 GPU 链路中完成图像预处理、推理、后处理和深度图采样，并将距离写入 `Detection::distance`。

## 辅助文件

- `scrfd_gpu_ops.cu/.h`：SCRFD 的 CUDA 图像预处理和候选框处理实现。
- `yolo_gpu_preprocess.cu/.h`：YOLO 的 CUDA 预处理、候选框生成和深度距离计算实现。
- `logging.h`、`macros.h`：TensorRT 日志和通用辅助定义。

## 引用

- TensorRT 实现编译为共享库 libtrt_infer.so。
- 导出 CMake target、公共头文件和依赖，其他 ROS 2 包可直接引用。
- 自动识别平台：
  - aarch64/arm64：按 Jetson AGX Orin 配置，默认 CUDA Compute Capability 为 87。
  - x86_64/amd64：使用 x86_64 TensorRT 库目录，由 CUDA 编译器选择默认 GPU 架构。
- 自动查找 CUDA、TensorRT、OpenCV、SQLite3 和 UUID。
- 安装时只发布 .h、.hpp 公共头文件，不安装 .cpp、.cu。

其他包可按以下方式引用：

```cmake
find_package(trt_infer REQUIRED)

target_link_libraries(your_target 
trt_infer::trt_infer)
```
