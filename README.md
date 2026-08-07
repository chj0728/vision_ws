# 视觉项目开发

## 相机型号 Gemini 2 L

### Install Dependencies

```bash
sudo apt install libgflags-dev nlohmann-json3-dev \
ros-$ROS_DISTRO-image-transport ros-${ROS_DISTRO}-image-transport-plugins ros-${ROS_DISTRO}-compressed-image-transport \
ros-$ROS_DISTRO-image-publisher ros-$ROS_DISTRO-camera-info-manager \
ros-$ROS_DISTRO-diagnostic-updater ros-$ROS_DISTRO-diagnostic-msgs ros-$ROS_DISTRO-statistics-msgs ros-$ROS_DISTRO-xacro \
ros-$ROS_DISTRO-backward-ros libdw-dev libssl-dev mesa-utils libgl1 libgoogle-glog-dev
```

### Clone && Build OrbbecSDK_ROS2

```bash
git clone https://github.com/orbbec/OrbbecSDK_ROS2.git
cd OrbbecSDK_ROS2
git checkout v2-main

colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install
```

### 查看 Orbbec Description

```bash
source install/setup.bash
ros2 launch orbbec_description view_model.launch.py model:=gemini_2_L.urdf.xacro
```

### 注册脚本（必需）

```bash
cd  .../OrbbecSDK_ROS2/orbbec_camera/scripts
sudo bash install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 启动相机

```bash
ros2 launch orbbec_camera gemini2L.launch.py \
camera_name:=camera \
depth_registration:=true \
uvc_backend:=v4l2 \
enable_ir:=false \
enable_point_cloud:=false
```
