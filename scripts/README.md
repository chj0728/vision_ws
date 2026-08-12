# vision_ws 构建脚本

`build.sh` 用于统一构建 `vision_ws`，自动处理 ROS 2、CUDA、GPU 架构和 OpenCV 路径，避免不同平台上的环境冲突。

## 已验证平台

1. NVIDIA RTX 5070 Ti、Ubuntu 22.04、CUDA 12.8、ROS 2 Humble、TensorRT 11.1.0。
2. x86_64、Ubuntu 22.04、CUDA 13.3、ROS 2 Humble、TensorRT 11.2.1。
3. Jetson AGX Orin、JetPack 6.2。

## 脚本处理步骤

运行脚本时会依次执行：

1. 确定工作空间根目录。
2. 加载 ROS 2 环境，默认使用 Humble。
3. 搜索完整 CUDA Toolkit，并检查 `bin/nvcc` 和 `nvvm/bin/cicc`。
4. 查找 Ubuntu/ROS 系统 OpenCV，避免误用 `/usr/local` 下由其他 CUDA 版本编译的 OpenCV。
5. 检测 CPU 平台：
   - Jetson AGX Orin 默认使用 CUDA 架构 `87`。
   - x86_64 优先通过 `nvidia-smi` 获取 GPU Compute Capability；读取失败时默认使用 `120`。
6. 临时移除 `src/trt_infer/COLCON_IGNORE`，脚本退出时自动恢复。
7. 将统一的 CUDA、CUDA 架构和 OpenCV 参数传递给工作空间内所有 CMake 功能包。
8. 通过 `colcon build --symlink-install` 编译工作空间。

## 基本使用

在工作空间根目录执行：

```bash
./scripts/build.sh
```

删除 `build`、`install` 和 `log` 后重新构建：

```bash
./scripts/build.sh pure
```

如果脚本没有执行权限：

```bash
chmod +x scripts/build.sh
```

## 手动指定环境

### 指定 CUDA Toolkit

若完整 CUDA Toolkit 位于 `/usr/local/cuda`：

```bash
TRT_CUDA_ROOT=/usr/local/cuda ./scripts/build.sh
```

脚本要求该目录中同时存在：

```text
bin/nvcc
nvvm/bin/cicc
```

仅有 `/usr/bin/nvcc` 不代表 CUDA Toolkit 安装完整。若构建出现 `cicc: not found`，应指定或重新安装完整 Toolkit，不要单独复制 `cicc`。

### 覆盖 CUDA 和 GPU 计算能力

例如 RTX 5070 Ti 使用 CUDA 12.8 和 Compute Capability 12.0：

```bash
TRT_CUDA_ROOT=/usr/local/cuda-12.8 \
CMAKE_CUDA_ARCHITECTURES=120 \
./scripts/build.sh pure
```

Jetson AGX Orin 默认使用 `87`，也可显式指定：

```bash
CMAKE_CUDA_ARCHITECTURES=87 ./scripts/build.sh pure
```

### 指定 ROS 2 版本

脚本默认加载 `/opt/ros/humble/setup.bash`。如需使用其他版本：

```bash
BUILD_ROS_DISTRO=jazzy ./scripts/build.sh pure
```

使用其他 ROS 2 版本前，应确认工作空间中的功能包和依赖均支持该版本。

### 指定 OpenCV

若自动检测不到系统 OpenCV，可显式指定包含 `OpenCVConfig.cmake` 的目录：

```bash
TRT_OPENCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
./scripts/build.sh pure
```

Jetson 上通常可使用：

```bash
TRT_OPENCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4 \
./scripts/build.sh pure
```

## 注意事项

### OpenCV 与 CUDA 版本冲突

如果出现以下错误：

```text
Could NOT find CUDA: Found unsuitable version "12.8", but required is exact version "11.8"
```

通常表示 CMake 选中了 `/usr/local/lib/cmake/opencv4/OpenCVConfig.cmake`，而该 OpenCV 是用 CUDA 11.8 编译的。优先使用 Ubuntu 系统 OpenCV：

```bash
sudo apt install libopencv-dev

TRT_OPENCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
./scripts/build.sh pure
```

不要在同一个构建中混用基于 CUDA 11.8 编译的 OpenCV 和 CUDA 12.8/13.x。

### 切换 CUDA、OpenCV 或 ROS 2 版本

CMake 会缓存编译器和依赖路径。切换环境后应执行纯净构建：

```bash
./scripts/build.sh pure
```

### 当前终端已加载其他 ROS 版本

脚本不直接使用当前环境中的 `ROS_DISTRO`，默认仍加载 Humble。这样可以避免当前终端已加载 Jazzy 时意外构建为 Jazzy。

### 多 GPU 主机

x86_64 平台默认读取 `nvidia-smi` 返回的第一张 GPU。若多张 GPU 的 Compute Capability 不同，应手动指定：

```bash
CMAKE_CUDA_ARCHITECTURES="89;120" ./scripts/build.sh pure
```

### 只构建 TensorRT 相关功能包

当前脚本默认构建整个工作空间。如需单独排查，可直接运行：

```bash
colcon build \
  --symlink-install \
  --packages-select trt_infer trt_infer_msgs trt_infer_ros \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_CUDA_ROOT=/usr/local/cuda-12.8 \
    -DTRT_OPENCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
    -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
    -DCMAKE_CUDA_ARCHITECTURES=120
```
