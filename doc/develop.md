# 开发指南

现在有以下几个trt 模块：

1.yolo_trt(人体检测+距离) 2.arcface_trt(人脸识别) 3.gender_trt(性别检测) 4. scrfd_trt(人脸检测) 5.sixdrepnet_trt(头部姿态)，其中1同时需要彩色和深度话题数据，2-5只需要彩色话题数据

如果后需要封装ros节点，应该用以下那种方式比较好（后者有其他更好的方式）：

每个模块单独封装节点，再汇总各个节点的数据发布
只分装一个订阅彩色和深度的节点，后续在一个回调里按需处理各个模块

## 节点封装方式选择

你这个场景里，我更推荐 **“1 和 2 的折中方案”**，而不是纯粹选其中一个。

更具体地说：

> **ROS 层只做 1 个彩色+深度同步订阅节点/组件，内部把图像分发给多个 TRT 推理模块；各 TRT 模块保持独立类/组件，不建议每个都单独订阅 ROS 图像话题。**

你的数据关系其实很适合这种架构：

```text
                    RGB Image
                       │
                       ▼
              ┌─────────────────┐
Depth Image ──▶│ Perception Node │
              │  图像同步/预处理   │
              └────────┬────────┘
                       │
          ┌────────────┼───────────────┐
          │            │               │
          ▼            ▼               ▼
       YOLO TRT      SCRFD TRT       其他逻辑
       RGB+Depth       RGB
          │            │
          │       ┌────┼───────────────┐
          │       ▼    ▼               ▼
          │    ArcFace Gender      SixDRepNet
          │     TRT     TRT           TRT
          │
          └──────────────┬───────────────
                         ▼
                  结果融合 / 发布
```

### 为什么不太建议 5 个节点都单独订阅 RGB

假设你是：

```text
/camera/color/image_raw
```

然后：

```text
yolo_node      subscribe RGB
arcface_node   subscribe RGB
gender_node    subscribe RGB
scrfd_node     subscribe RGB
sixdrep_node   subscribe RGB
```

在 ROS2 里虽然逻辑非常清楚，但你会遇到几个实际问题。

首先是**图像重复传输和复制**。如果 RGB 是 1920×1080、30fps：

```text
1920 × 1080 × 3 ≈ 6 MB / frame

30 fps ≈ 180 MB/s
```

5 个节点都收：

```text
RGB：
~180 MB/s × 5
```

即使 ROS2 DDS、共享内存、intra-process 能减少部分 copy，依然会有额外：

* DDS 消息处理
* callback 调度
* cv_bridge
* resize
* normalize
* CPU→GPU memcpy
* CUDA stream 同步

对于 Jetson Orin 这种平台，这些额外开销非常值得避免。

第二个问题是**预处理会大量重复**。例如：

```cpp
sensor_msgs::msg::Image
       ↓
cv::Mat
       ↓
resize
       ↓
BGR -> RGB
       ↓
normalize
       ↓
cudaMemcpy
```

如果 5 个节点都各做一遍，会浪费很多 CPU/GPU 带宽。

---

## 但我也不建议把所有东西直接写在一个 callback 里面

比如这样：

```cpp
void imageCallback(...)
{
    yolo.infer();

    scrfd.infer();

    arcface.infer();

    gender.infer();

    sixdrepnet.infer();
}
```

这种方式初期简单，但以后很容易变成一个非常大的 callback。

尤其你的模块之间其实存在明显的依赖关系：

```text
RGB
 │
 ├──────── YOLO
 │
 └──────── SCRFD
             │
             ├──── ArcFace
             ├──── Gender
             └──── SixDRepNet
```

因为：

* ArcFace 并不需要整张图
* Gender 通常也只需要 face ROI
* SixDRepNet 通常也应该输入 face/head ROI

所以实际上应该是：

```text
SCRFD
  ↓
face bbox
  ↓
crop face
  ├── ArcFace
  ├── Gender
  └── SixDRepNet
```

这也是为什么我更推荐一个**感知 Pipeline 节点**。

---

# 推荐架构

可以设计成：

```text
                 ROS2 Camera
              /              \
           RGB                Depth
            │                   │
            └────── sync ───────┘
                      │
                      ▼
             PerceptionNode
                      │
             ┌────────┴────────┐
             │                 │
          YoloTRT            ScrfdTRT
             │                 │
             │             face bbox
             │                 │
             │        ┌────────┼─────────┐
             │        │        │         │
             │        ▼        ▼         ▼
             │     ArcFace   Gender   SixDRepNet
             │
             ▼
       person + depth
             │
             └─────────────┐
                           ▼
                  PerceptionFusion
                           │
                           ▼
                 ROS2 Result Topic
```

例如代码结构：

```text
perception_ros/
│
├── src/
│   ├── perception_node.cpp
│   ├── perception_pipeline.cpp
│   │
│   └── trt/
│       ├── yolo_trt.cpp
│       ├── scrfd_trt.cpp
│       ├── arcface_trt.cpp
│       ├── gender_trt.cpp
│       └── sixdrepnet_trt.cpp
│
├── include/
│   ├── perception_pipeline.hpp
│   └── trt/
│       ├── yolo_trt.hpp
│       ├── scrfd_trt.hpp
│       ├── arcface_trt.hpp
│       ├── gender_trt.hpp
│       └── sixdrepnet_trt.hpp
```

也就是说：

```cpp
class PerceptionNode : public rclcpp::Node
{
private:

    PerceptionPipeline pipeline_;

    void imageCallback(
        const Image::SharedPtr rgb,
        const Image::SharedPtr depth);
};
```

而：

```cpp
class PerceptionPipeline
{
private:

    YoloTRT yolo_;
    ScrfdTRT scrfd_;
    ArcfaceTRT arcface_;
    GenderTRT gender_;
    SixDRepNetTRT sixdrep_;

public:

    PerceptionResult process(
        const cv::Mat& rgb,
        const cv::Mat& depth);
};
```

这样 ROS 和算法彻底解耦。

---

# 更重要的一点：不要所有模型每帧都跑

这是我认为你这个系统后面性能优化最关键的地方。

假设摄像头：

```text
30 FPS
```

没有必要：

```text
YOLO        30 FPS
SCRFD       30 FPS
ArcFace     30 FPS
Gender      30 FPS
HeadPose    30 FPS
```

可以做成：

```text
YOLO
15~30 FPS

SCRFD
10~15 FPS

ArcFace
只在：
    新人脸出现
    tracking ID 改变
    人脸质量足够好
时运行

Gender
只在人脸第一次稳定出现时运行一次

SixDRepNet
10~30 FPS
```

例如一个人脸 ID：

```text
face_track_id = 7
```

第一次：

```text
SCRFD
 ↓
ArcFace      → 张三
Gender       → male
SixDRepNet   → yaw/pitch/roll
```

之后几十帧根本不用再跑：

```text
ArcFace
Gender
```

只需要：

```text
tracker
+
SixDRepNet
```

这样 GPU 占用可能差很多。

---

# YOLO 和 SCRFD 可以并行

你的两个主要入口模型：

```text
YOLO
SCRFD
```

彼此没有依赖。

因此不要：

```cpp
yolo.infer(rgb);
scrfd.infer(rgb);
```

完全串行。

后续可以设计成两个 CUDA Stream：

```text
RGB
 │
 ├── CUDA Stream 1
 │      └── YOLO
 │
 └── CUDA Stream 2
        └── SCRFD
```

然后：

```text
SCRFD finished
       │
       ▼
 Face ROI
  │   │   │
  ▼   ▼   ▼
Arc Gender Pose
```

甚至 ArcFace / Gender / SixDRepNet 也可以根据 TensorRT context 情况继续并行。

最终：

```text
           Frame N
              │
      ┌───────┴────────┐
      ▼                ▼
    YOLO             SCRFD
      │                │
      │          ┌─────┼──────┐
      │          ▼     ▼      ▼
      │        Arc   Gender   Pose
      │          │     │      │
      └──────────┴─────┴──────┘
                    │
                    ▼
                  Fusion
```

这比简单的：

```cpp
callback()
{
    model1();
    model2();
    model3();
    model4();
    model5();
}
```

扩展性好很多。

---

# 深度图也不要传给所有模块

你的信息关系可以明确限定成：

```text
RGB
 ├── YOLO
 └── SCRFD
      ├── ArcFace
      ├── Gender
      └── SixDRepNet


Depth
 └── YOLO detection result
      ↓
     distance
```

尤其注意：

> YOLO 的 TensorRT 网络本身大概率其实只需要 RGB，真正需要 Depth 的是 YOLO 后面的“距离计算”。

因此更建议进一步拆成：

```text
YOLO TRT
    ↓
person bbox
    ↓
DepthEstimator
    ↓
person distance
```

而不是让：

```cpp
YoloTRT::infer(rgb, depth);
```

把 TensorRT 和深度业务逻辑绑死。

可以定义为：

```cpp
auto persons = yolo_.infer(rgb);

for (auto& person : persons)
{
    person.distance =
        depth_estimator_.estimate(
            depth,
            person.bbox);
}
```

这样以后即使距离来源换成：

```text
RealSense depth
→ stereo
→ 激光雷达
→ 3D camera
```

YOLO 都不用改。

---

# ROS2 层我会这样划分

如果整个系统部署在**同一台 Jetson Orin / GPU 主机**：

### 一个主要感知节点

```text
perception_node
```

订阅：

```text
/camera/color/image_raw
/camera/depth/image_raw
/camera/color/camera_info
```

内部：

```text
YoloTRT
ScrfdTRT
ArcFaceTRT
GenderTRT
SixDRepNetTRT
```

发布：

```text
/perception/persons
/perception/faces
/perception/debug_image
```

例如最终消息不要分散成：

```text
/yolo/results
/scrfd/results
/arcface/results
/gender/results
/head_pose/results
```

再额外做一个节点拼起来。

更建议直接定义类似：

```text
Person[]
```

每个人：

```text
track_id

person_bbox

distance

face_bbox

face_id
face_name
face_similarity

gender
gender_confidence

yaw
pitch
roll
```

即：

```text
Person
 ├── track_id
 ├── bbox
 ├── distance
 │
 └── Face
      ├── bbox
      ├── identity
      ├── gender
      └── head_pose
```

对机器人上层来说会非常舒服。

例如行为节点只关心：

```cpp
for (const auto& person : msg->persons)
{
    if (
        person.distance < 2.0 &&
        abs(person.head_pose.yaw) < 20 &&
        person.face_name == "张三")
    {
        ...
    }
}
```

不需要自己同步 5 个 topic。

---

# 还有一个比普通单节点更适合 ROS2 的选择

如果你既希望：

> 各模块保持“节点级模块化”

又希望：

> RGB 图像不要经过 DDS 重复复制

可以使用：

**ROS2 Components + intra-process communication**

例如：

```text
component_container
│
├── camera_component
├── yolo_component
├── face_component
└── fusion_component
```

开启：

```cpp
rclcpp::NodeOptions()
    .use_intra_process_comms(true)
```

这样可以达到一个不错的折中：

```text
逻辑：
多个 ROS2 Component

进程：
一个 process

通信：
intra-process / zero-copy-ish
```

比：

```text
5 个独立 process
```

性能更好，同时比：

```text
一个 5000 行大节点
```

维护性好。

不过你的这 5 个模型耦合程度其实比较高，特别是：

```text
SCRFD
→ ArcFace
→ Gender
→ SixDRepNet
```

所以我不会一开始就把它们拆成 5 个 ROS Component。

---

# 我建议你最终划成 3 层

这是最适合后续长期维护的方式：

```text
────────────────────────────
        ROS Layer
────────────────────────────

PerceptionNode

RGB / Depth subscriber
message sync
parameter
publisher

           ↓

────────────────────────────
      Pipeline Layer
────────────────────────────

PerceptionPipeline

YOLO scheduling
Face scheduling
tracking
result fusion
frequency control

           ↓

────────────────────────────
       TRT Layer
────────────────────────────

YoloTRT

ScrfdTRT

ArcFaceTRT

GenderTRT

SixDRepNetTRT
```

TRT 层完全不知道 ROS 存在：

```cpp
std::vector<Face>
ScrfdTRT::infer(const cv::Mat& image);
```

而不是：

```cpp
void ScrfdTRT::rosCallback(
    sensor_msgs::msg::Image::SharedPtr msg);
```

这个区别非常重要。

以后你做：

* ROS2
* Python benchmark
* 单元测试
* 视频离线推理
* TensorRT 性能测试
* 服务端部署

都可以直接复用 TRT 类。

---

## 所以你给出的两个方案，我会这样评价

| 方案 | 性能 | 模块化 | 开发难度 | 推荐 |
| ---------------------------------- | --------: | -------: | -------: | ------: |
| 5 个独立 ROS 节点 | ★★ | ★★★★★ | ★★★ | 一般 |
| 一个 callback 串行跑 5 模型 | ★★★ | ★★ | ★★★★★ | 不推荐长期使用 |
| **单 ROS 节点 + 独立 TRT 类 + Pipeline** | **★★★★★** | **★★★★** | **★★★★** | **最推荐** |
| ROS2 Component + Pipeline | ★★★★★ | ★★★★★ | ★★★ | 系统复杂后考虑 |

所以针对你目前的项目，我会直接选：

> **一个 `perception_node` 负责 RGB/Depth 获取与同步，内部一个 `PerceptionPipeline` 管理 5 个 TRT 模块。5 个 TRT 模块保持独立 C++ 类。YOLO 和 SCRFD 并行执行，ArcFace/Gender/SixDRepNet 在 SCRFD 的 ROI 上按需执行。最终统一发布 Person/Face 感知结果。**

而不是简单采用你的方案 2 那种“所有模型全部塞进一个 ROS callback 顺序执行”。这是性能、代码维护性、后续 CUDA Stream 并行和模型调度之间比较好的平衡点。
