# vision_ws 脚本说明

本目录提供 `install.sh`、`build.sh`、`run.sh` 和 `deploy.sh`，分别用于安装依赖、构建工作空间、启动视觉栈及配置开机自启动。

## 已验证平台

1. NVIDIA RTX 5070 Ti、Ubuntu 22.04、CUDA 12.8、ROS 2 Humble、TensorRT 11.1.0。
2. x86_64、Ubuntu 22.04、CUDA 13.3、ROS 2 Humble、TensorRT 11.2.1。
3. Jetson AGX Orin、JetPack 6.2。

## 安装

`install.sh` 安装系统依赖，并解压 OrbbecSDK 和 ROS 2 `rcutils` 开发包到工作空间的 `src/third_deps/`。脚本会调用 `sudo apt install`，执行时需要具备 sudo 权限。

在工作空间根目录执行：

```bash
./scripts/install.sh
```

安装前请确认 `third/` 中存在所需的压缩包；若脚本没有执行权限：

```bash
chmod +x scripts/install.sh
```

## 构建

`build.sh` 用于统一构建 `vision_ws`，自动处理 ROS 2、CUDA、GPU 架构和 OpenCV 路径，避免不同平台上的环境冲突。

### 构建流程

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

### 基础构建

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

### 手动指定构建环境

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

### 构建注意事项

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

## 运行脚本说明

  在完成构建后，在工作空间根目录执行：

  ```bash
  ./scripts/run.sh
  ```

  `run.sh` 会加载 ROS 2 和工作空间环境，然后启动：
  `ros2 launch trt_infer_ros start_all_launch.py`

  `start_all_launch.py` 默认将相机与推理组件加载到 `perception_container`。脚本将 launch 进程置于独立进程组中；进程异常退出时会等待 10 秒后自动重启。重复执行脚本时，新的实例会先停止旧实例。

  按 `Ctrl+C` 可停止守护脚本及其启动的全部节点。启动日志写入：

  ```text
  logs/start.launch.log
  ```

  运行状态 PID 文件位于 `.run.pids/`，正常退出时会自动清理。ROS 2 节点所需的临时文件日志也仅存放于该目录并自动清理；持久化日志只保留 `logs/start.launch.log`。ROS 2 控制台颜色输出已关闭，因此新写入的日志不包含 ANSI 颜色控制字符。

### 运行参数

  默认使用 ROS 2 Humble。可通过环境变量调整运行环境和守护行为：

  ```bash
  ROS_DISTRO=jazzy ./scripts/run.sh
  RESTART_DELAY=5 SHUTDOWN_TIMEOUT=10 ./scripts/run.sh
  ```

- `ROS_DISTRO`：ROS 2 发行版，默认 `humble`。
- `RESTART_DELAY`：节点异常退出后的重启等待时间，单位为秒，默认 `10`。
- `SHUTDOWN_TIMEOUT`：优雅停止超时时间，单位为秒，默认 `5`。

## 开机自启动

`deploy.sh` 使用 Supervisor 将 `run.sh` 注册为系统服务。部署后，视觉栈会在开机后自动启动；`run.sh` 或其 launch 进程异常退出时也会自动恢复。

首次部署在工作空间根目录执行：

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh deploy
```

常用管理命令：

```bash
./scripts/deploy.sh status
./scripts/deploy.sh restart
./scripts/deploy.sh stop
./scripts/deploy.sh start
./scripts/deploy.sh remove
./scripts/deploy.sh logs
```

Supervisor 配置位于 `/etc/supervisor/conf.d/vision_stack.conf`。节点主日志仍写入 `logs/start.launch.log`；Supervisor 自身的状态和环境错误写入 `logs/supervisor.log`。执行 `remove` 会停止服务并删除该配置，不会删除日志或工作区文件。
