#!/usr/bin/env bash

set -euo pipefail

# ===================== 工作空间配置 =====================
SHELL_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_DIR=$(cd "$SHELL_DIR/.." && pwd)

BUILD_DIR="$PROJECT_DIR/build"
INSTALL_DIR="$PROJECT_DIR/install"
LOG_DIR="$PROJECT_DIR/log"
ROS_DISTRO_NAME="${BUILD_ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

# ===================== 环境检测 =====================
if [[ ! -f "$ROS_SETUP" ]]; then
    echo "错误：未找到 ROS 2 环境：$ROS_SETUP" >&2
    exit 1
fi
# ROS setup 脚本会读取部分未预先定义的环境变量
set +u
source "$ROS_SETUP"
set -u

# 优先使用显式配置，再搜索系统中的完整 CUDA Toolkit
find_cuda_root() {
    local candidates=()
    local candidate

    [[ -n "${TRT_CUDA_ROOT:-}" ]] && candidates+=("$TRT_CUDA_ROOT")
    [[ -n "${CUDA_HOME:-}" ]] && candidates+=("$CUDA_HOME")
    [[ -n "${CUDA_PATH:-}" ]] && candidates+=("$CUDA_PATH")
    candidates+=("/usr/local/cuda")

    while IFS= read -r candidate; do
        candidates+=("$candidate")
    done < <(find /usr/local -maxdepth 1 -type d -name 'cuda-*' -print 2>/dev/null | sort -Vr)

    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate/bin/nvcc" && -x "$candidate/nvvm/bin/cicc" ]]; then
            readlink -f "$candidate"
            return 0
        fi
    done

    return 1
}

if ! CUDA_ROOT=$(find_cuda_root); then
    echo "错误：未找到完整 CUDA Toolkit，需要同时存在 bin/nvcc 和 nvvm/bin/cicc。" >&2
    echo "可通过 TRT_CUDA_ROOT=/usr/local/cuda-12.8 指定安装目录。" >&2
    exit 1
fi

# 固定使用系统 OpenCV，避免误选 /usr/local 下由其他 CUDA 版本编译的 OpenCV
find_opencv_dir() {
    local multiarch
    local candidates=()
    local candidate

    [[ -n "${TRT_OPENCV_DIR:-}" ]] && candidates+=("$TRT_OPENCV_DIR")
    multiarch=$(gcc -print-multiarch 2>/dev/null || true)
    [[ -n "$multiarch" ]] && candidates+=("/usr/lib/$multiarch/cmake/opencv4")
    candidates+=("/usr/lib/cmake/opencv4" "/usr/share/OpenCV")

    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate/OpenCVConfig.cmake" ]]; then
            readlink -f "$candidate"
            return 0
        fi
    done

    return 1
}

if ! OPENCV_DIR=$(find_opencv_dir); then
    echo "错误：未找到 Ubuntu/ROS 系统 OpenCVConfig.cmake。" >&2
    echo "请安装 libopencv-dev，或通过 TRT_OPENCV_DIR 指定兼容当前 CUDA 的 OpenCV。" >&2
    exit 1
fi

# Orin 使用 SM 8.7，x86 优先从 NVIDIA 驱动读取实际计算能力
case "$(uname -m)" in
    aarch64|arm64)
        CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-87}"
        PLATFORM="Jetson AGX Orin"
        ;;
    x86_64|amd64)
        if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
            CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES"
        elif command -v nvidia-smi >/dev/null 2>&1; then
            GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:].')
            CUDA_ARCHITECTURES="${GPU_COMPUTE_CAP:-120}"
        else
            CUDA_ARCHITECTURES=120
        fi
        PLATFORM="x86_64"
        ;;
    *)
        echo "错误：不支持的平台架构：$(uname -m)" >&2
        exit 1
        ;;
esac

# ===================== 清理构建 =====================
pure_build() {
    echo "清理工作空间的 build、install 和 log 目录。"
    rm -rf "$BUILD_DIR" "$INSTALL_DIR" "$LOG_DIR"
}

# ===================== 编译安装 =====================
if [[ "${1:-}" == "pure" ]]; then
    pure_build
fi

echo "平台：$PLATFORM ($(uname -m))"
echo "CUDA：$CUDA_ROOT"
echo "CUDA 架构：$CUDA_ARCHITECTURES"
echo "OpenCV：$OPENCV_DIR"
echo "ROS 2：$ROS_DISTRO_NAME"

# 构建期间启用 trt_infer，脚本退出时恢复原有忽略状态
TRT_COLCON_IGNORE="$PROJECT_DIR/src/trt_infer/COLCON_IGNORE"
TRT_COLCON_IGNORE_BACKUP="$PROJECT_DIR/src/trt_infer/COLCON_IGNORE.build-disabled"
if [[ -e "$TRT_COLCON_IGNORE" ]]; then
    if [[ -e "$TRT_COLCON_IGNORE_BACKUP" ]]; then
        echo "错误：临时文件已存在：$TRT_COLCON_IGNORE_BACKUP" >&2
        exit 1
    fi
    mv "$TRT_COLCON_IGNORE" "$TRT_COLCON_IGNORE_BACKUP"
    trap 'mv "$TRT_COLCON_IGNORE_BACKUP" "$TRT_COLCON_IGNORE"' EXIT
fi

# 统一向工作空间内所有 CMake 功能包传递 CUDA 和 OpenCV 配置
cd "$PROJECT_DIR"
colcon build \
    --event-handlers console_direct+ \
    --symlink-install \
    --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DTRT_CUDA_ROOT="$CUDA_ROOT" \
        -DTRT_OPENCV_DIR="$OPENCV_DIR" \
        -DOpenCV_DIR="$OPENCV_DIR" \
        -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"