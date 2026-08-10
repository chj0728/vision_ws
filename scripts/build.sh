
#!/bin/bash

# ===================== 配置区 =====================
ENABLE_SCRIPT=true
SHELL_DIR=$(dirname $(readlink -f "$0"))
PROJECT_DIR=$(cd "$SHELL_DIR/.." && pwd)
THIRD_DIR=$(cd "$SHELL_DIR/../third" && pwd)

BUILD_DIR="$PROJECT_DIR/build"
INSTALL_DIR="$PROJECT_DIR/install"
LOG_DIR="$PROJECT_DIR/log"

# if ./build.sh pure, remove build and install, log directory first
pure_build() {
    echo "Performing pure build: removing build, install, and log directories."
    if [ -d "$BUILD_DIR" ]; then
        echo "Removing existing build directory: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi
    if [ -d "$INSTALL_DIR" ]; then
        echo "Removing existing install directory: $INSTALL_DIR"
        rm -rf "$INSTALL_DIR"
    fi
    if [ -d "$LOG_DIR" ]; then
        echo "Removing existing log directory: $LOG_DIR"
        rm -rf "$LOG_DIR"
    fi
}

# ===================== 编译安装 =====================
cd $PROJECT_DIR
# 移除旧的构建和安装目录
if [ "$1" == "pure" ]; then
    pure_build
fi
colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install