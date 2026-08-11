# 视觉项目开发

## support platform

| Platform | OS | Support |
| -------- | ------- | ------- |
| X86_64 | Ubuntu 22.04 + ROS2 Humble + CUDA 13.3 | ✅ |
| Jetson AGX Orin | Ubuntu 22.04 + ROS2 Humble + JetPack6.2 | ✅ |

## 依赖安装

```bash
./scripts/install.sh
```

## 编译项目

```bash
./scripts/build.sh

# or before building, remove the build cache and rebuild the project
./scripts/build.sh pure
```

## 运行项目

- 相机节点

```bash
ros2 launch trt_infer_ros trt_infer_camera.launch.py
```

- 推理节点

```bash
ros2 launch trt_infer_ros trt_infer_ros_node.launch.py use_composition:=False
```

### 运行所有节点

```bash
ros2 launch trt_infer_ros start.launch.py 
```

## 注意事项

### 相机部署

参考文档：[相机部署](./doc/camera.md)

### 安装 TensorRT

参考链接：[Debian Package Installation — NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-debian.html#)
