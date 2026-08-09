#!/bin/bash

# ===================== 配置区 =====================
ENABLE_SCRIPT=true
SHELL_DIR=$(dirname $(readlink -f "$0"))
PROJECT_DIR=$(cd "$SHELL_DIR/.." && pwd)
THIRD_DIR=$(cd "$SHELL_DIR/../third" && pwd)

echo "Project Path: $PROJECT_DIR"
echo "Scripts Path: $SHELL_DIR"
echo "Third dependencies Path: $THIRD_DIR"

# ===================== 通用解压安装函数 =====================
# 参数:
#   $1 - zip 文件路径
#   $2 - 目标解压目录（父目录）
#   $3 - 解压后的文件夹名（用于验证，可选，默认从 zip 文件名推断）
extract_and_install() {
    local zip_path="$1"
    local dest_dir="$2"
    local extract_name="$3"

    # 如果未指定解压后文件夹名，从 zip 文件名推断（去掉 .zip）
    if [ -z "$extract_name" ]; then
        extract_name=$(basename "$zip_path" .zip)
    fi

    local target_path="$dest_dir/$extract_name"

    echo "=========================================="
    echo "Installing: $extract_name"
    echo "  Source: $zip_path"
    echo "  Target: $target_path"
    echo "=========================================="

    # 1. 如果目标目录已存在，先删除
    if [ -d "$target_path" ]; then
        echo "Removing existing directory: $target_path"
        rm -rf "$target_path"
    fi

    # 2. 检查 zip 文件是否存在
    if [ ! -f "$zip_path" ]; then
        echo "Error: Zip file not found: $zip_path"
        return 1
    fi

    # 3. 确保目标目录存在
    mkdir -p "$dest_dir"

    # 4. 解压
    echo "Unzipping..."
    unzip -o "$zip_path" -d "$dest_dir"

    # 5. 验证解压结果
    if [ -d "$target_path" ]; then
        echo "✓ Successfully installed to: $target_path"
        return 0
    else
        echo "✗ Extraction failed: $target_path not found!"
        return 1
    fi
}


# ===================== OrbbecSDK 解压安装 =====================
sudo apt install libgflags-dev nlohmann-json3-dev \
ros-$ROS_DISTRO-image-transport ros-${ROS_DISTRO}-image-transport-plugins ros-${ROS_DISTRO}-compressed-image-transport \
ros-$ROS_DISTRO-image-publisher ros-$ROS_DISTRO-camera-info-manager \
ros-$ROS_DISTRO-diagnostic-updater ros-$ROS_DISTRO-diagnostic-msgs ros-$ROS_DISTRO-statistics-msgs ros-$ROS_DISTRO-xacro \
ros-$ROS_DISTRO-backward-ros libdw-dev libssl-dev mesa-utils libgl1 libgoogle-glog-dev

ORBBEC_ZIP="$THIRD_DIR/OrbbecSDK_ROS2-2-main.zip"
ORBBEC_SRC="$PROJECT_DIR/src/third_deps"       # 目标解压目录
ORBBEC_DIR="OrbbecSDK_ROS2-2-main" # 解压后的文件夹名

extract_and_install \
    "$ORBBEC_ZIP" \
    "$ORBBEC_SRC" \
    "$ORBBEC_DIR" || exit 1

# ===================== rcutils-humble-dev.zip 解压安装 =====================
RCUTILS_ZIP="$THIRD_DIR/rcutils-humble-dev.zip"
RCUTILS_SRC="$PROJECT_DIR/src/third_deps"
RCUTILS_DIR="rcutils-humble-dev"

extract_and_install \
    "$RCUTILS_ZIP" \
    "$RCUTILS_SRC" \
    "$RCUTILS_DIR" || exit 1

echo ""
echo "=========================================="
echo "All dependencies installed successfully!"
echo "=========================================="