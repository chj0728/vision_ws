
#!/bin/bash

# ===================== 配置区 =====================
ENABLE_SCRIPT=true
SHELL_DIR=$(dirname $(readlink -f "$0"))
PROJECT_DIR=$(cd "$SHELL_DIR/.." && pwd)
THIRD_DIR=$(cd "$SHELL_DIR/../third" && pwd)

BUILD_DIR="$PROJECT_DIR/build"
INSTALL_DIR="$PROJECT_DIR/install"

cd $PROJECT_DIR
colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install