# ROS 2 图像话题与压缩格式

不同的 ROS 2 图像话题，主要区别在于**传输格式和压缩方式**：

## 📊 话题对比

### 原始图像话题

| 话题 | 格式 | 用途 |
| ------ | ---------- | ------ |
| `/camera/color/image_raw` | 原始 RGB 图像（未压缩） | 本地处理、计算机视觉、高精度需求 |
| `/camera/depth/image_raw` | 原始深度图（16-bit 单通道，单位 mm） | SLAM、点云生成、精确测距 |

---

### 压缩图像话题

| 话题 | 压缩格式 | 特点 |
| ------ | ---------- | ------ |
| `/camera/color/image_raw/compressed` | **JPEG/PNG** | 有损/无损压缩，兼容性好，网络传输常用 |
| `/camera/depth/image_raw/compressed` | **PNG** | 无损压缩，保持深度精度 |
| `/camera/depth/image_raw/compressedDepth` | **PNG + 深度专用编码** | 专为深度图优化的压缩，效率更高 |
| `/camera/depth/image_raw/theora` | **Theora 视频编码** | 流式视频压缩，适合连续传输 |

---

## 🔍 详细区别

### 1. `image_raw` vs `compressed`

```python
# 原始图像（image_raw）
- 数据大小：1920×1080×3 bytes ≈ 6.2 MB/帧（RGB8）
- 传输带宽：高（100+ Mbps 连续传输）
- 延迟：最低（无编解码开销）
- 质量：完全无损

# 压缩图像（compressed）
- 数据大小：通常 50-500 KB/帧（JPEG 质量 80%）
- 传输带宽：低（10-50 Mbps）
- 延迟：略高（需编解码）
- 质量：取决于压缩参数（JPEG 有损）
```

---

### 2. 深度图特殊压缩

```python
# depth/image_raw/compressed（普通 PNG）
- 使用标准 PNG 压缩
- 保持原始精度（16-bit）
- 压缩率一般（约 2-5 倍）

# depth/image_raw/compressedDepth（深度专用）
- 使用 depth_image_proc 的特殊编码
- 将 16-bit 深度转换为彩色 PNG（更高效压缩）
- 保持毫米级精度
- 压缩率更高（可达 10-30 倍）
```

---

## 🎯 如何选择

### 订阅 `image_raw`（原始图像）

- ✅ 计算机视觉处理（OpenCV、目标检测）
- ✅ 高精度需求（测量、校准）
- ❌ 带宽有限的远程传输
- ❌ 多机器人/多相机系统

### 订阅 `compressed`（压缩图像）

- ✅ Web 可视化（RViz、Web 界面）
- ✅ 远程监控（低带宽网络）
- ✅ 录制 bag 文件（节省磁盘空间）
- ✅ 多话题同时订阅
- ❌ 高精度图像分析

### 订阅 `compressedDepth`（深度专用压缩）

- ✅ 远程传输深度图（如 Web 端 SLAM 可视化）
- ✅ 保存深度 bag 文件（节省 90% 空间）
- ✅ 点云生成（需要解压缩恢复深度值）

---

## 💻 代码示例

### 订阅原始图像（高性能本地处理）

```python
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()

def image_callback(msg):
    # 直接转换为 OpenCV 格式
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    cv2.imshow("Camera", cv_image)
    cv2.waitKey(1)

# 创建订阅
node.create_subscription(Image, '/camera/color/image_raw', image_callback, 10)
```

### 订阅压缩图像（远程传输）

```python
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

def compressed_callback(msg):
    # 解压缩 JPEG 数据
    np_arr = np.frombuffer(msg.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imshow("Camera", cv_image)
    cv2.waitKey(1)

# 订阅压缩话题
node.create_subscription(CompressedImage, '/camera/color/image_raw/compressed', compressed_callback, 10)
```

### 订阅压缩深度图

```python
# 压缩深度图仍然是 sensor_msgs/Image 类型
# 只是经过 depth_image_proc 特殊编码
node.create_subscription(Image, '/camera/depth/image_raw/compressedDepth', depth_callback, 10)
```

---

## ⚡ 性能对比（实测参考）

| 指标 | Raw | Compressed (JPEG 80%) | CompressedDepth |
| ------ | ----- | ---------------------- | ----------------- |
| 单帧大小 (1080p) | 6.2 MB | 200 KB | 120 KB |
| 30 FPS 带宽 | 1490 Mbps | 48 Mbps | 29 Mbps |
| 编码延迟 | 0 ms | 2-5 ms | 3-8 ms |
| 解码延迟 | 0 ms | 1-3 ms | 2-5 ms |
| CPU 占用 | 低 | 中 | 中 |

---

## 📝 建议

1. **本地开发/算法测试**：用 `/image_raw`（零延迟，原汁原味）
2. **多机通信/远程监控**：用 `/compressed`（节省 95% 带宽）
3. **录制 bag**：用 `compressed` 或 `compressedDepth`（节省磁盘）
4. **Web 可视化**：优先用 `compressed`（浏览器友好）
