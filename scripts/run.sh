#!/usr/bin/env bash
set -euo pipefail

# 视觉栈守护脚本：start.launch.py 启动相机，并将推理节点加载到相机组件容器。

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
WORK_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
LOGS_DIR="$WORK_DIR/logs"
PID_DIR="$WORK_DIR/.run.pids"
SUPERVISOR_PID_FILE="$PID_DIR/run.pid"
LAUNCH_PID_FILE="$PID_DIR/start_launch.pid"

ROS_DISTRO_NAME=${ROS_DISTRO:-humble}
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
RESTART_DELAY=${RESTART_DELAY:-10}
SHUTDOWN_TIMEOUT=${SHUTDOWN_TIMEOUT:-5}
LAUNCH_COMMAND=(ros2 launch trt_infer_ros start.launch.py)

LAUNCH_PID=""

is_process_group_running() {
    kill -0 -- "-$1" 2>/dev/null
}

stop_process_group() {
    local pid=$1
    local deadline=$((SECONDS + SHUTDOWN_TIMEOUT))

    is_process_group_running "$pid" || return 0
    kill -INT -- "-$pid" 2>/dev/null || true
    while is_process_group_running "$pid" && (( SECONDS < deadline )); do
        sleep 0.1
    done
    if is_process_group_running "$pid"; then
        echo "[WARN] Shutdown timed out after ${SHUTDOWN_TIMEOUT}s, forcing exit"
        kill -KILL -- "-$pid" 2>/dev/null || true
    fi
}

cleanup_previous_launch() {
    [[ -f "$LAUNCH_PID_FILE" ]] || return 0

    local old_pid old_cmd
    old_pid=$(<"$LAUNCH_PID_FILE")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        old_cmd=$(ps -p "$old_pid" -o args= 2>/dev/null || true)
        if [[ "$old_cmd" == *"start.launch.py"* ]]; then
            echo "[INFO] Stopping previous visual stack (PGID=$old_pid)"
            stop_process_group "$old_pid"
        else
            echo "[WARN] Ignoring stale launch PID file (PID=$old_pid)"
        fi
    fi
    rm -f "$LAUNCH_PID_FILE"
}

stop_previous_supervisor() {
    [[ -f "$SUPERVISOR_PID_FILE" ]] || return 0

    local old_pid old_cmd
    old_pid=$(<"$SUPERVISOR_PID_FILE")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && [[ "$old_pid" != "$$" ]] && kill -0 "$old_pid" 2>/dev/null; then
        old_cmd=$(ps -p "$old_pid" -o args= 2>/dev/null || true)
        if [[ "$old_cmd" == *"scripts/run.sh"* ]]; then
            echo "[INFO] Stopping previous run.sh (PID=$old_pid)"
            kill -INT "$old_pid" 2>/dev/null || true
        fi
    fi
}

cleanup_pid_files() {
    if [[ -f "$SUPERVISOR_PID_FILE" ]] && [[ "$(<"$SUPERVISOR_PID_FILE")" == "$$" ]]; then
        rm -f "$SUPERVISOR_PID_FILE" "$LAUNCH_PID_FILE"
    fi
    return 0
}

shutdown() {
    echo "[INFO] Shutdown requested"
    trap - SIGINT SIGTERM EXIT
    [[ -n "$LAUNCH_PID" ]] && stop_process_group "$LAUNCH_PID"
    rm -rf "$PID_DIR"
    echo "[INFO] Exit"
    exit 0
}

prepare_environment() {
    [[ -f "$ROS_SETUP" ]] || { echo "[ERROR] ROS 2 setup file not found: $ROS_SETUP" >&2; return 1; }
    [[ -f "$WORK_DIR/install/setup.bash" ]] || { echo "[ERROR] Workspace setup file not found: $WORK_DIR/install/setup.bash" >&2; return 1; }

    cd "$WORK_DIR"
    set +u
    source "$ROS_SETUP"
    source "$WORK_DIR/install/setup.bash"
    set -u
    command -v ros2 >/dev/null 2>&1 || { echo "[ERROR] ros2 was not found after loading the environment" >&2; return 1; }

    mkdir -p "$LOGS_DIR" "$PID_DIR"
    echo "[INFO] ROS 2 environment ready: distro=$ROS_DISTRO_NAME"
}

supervise_launch() {
    local exit_code

    while true; do
        echo "[INFO] Starting visual stack..."
        setsid env --default-signal=INT,QUIT "${LAUNCH_COMMAND[@]}" >> "$LOGS_DIR/start.launch.log" 2>&1 &
        LAUNCH_PID=$!
        printf '%s\n' "$LAUNCH_PID" > "$LAUNCH_PID_FILE"
        echo "[INFO] Launch log: $LOGS_DIR/start.launch.log"

        if wait "$LAUNCH_PID"; then exit_code=0; else exit_code=$?; fi
        rm -f "$LAUNCH_PID_FILE"
        LAUNCH_PID=""
        echo "[WARN] Visual stack exited (code=$exit_code), restarting in ${RESTART_DELAY}s..."
        sleep "$RESTART_DELAY"
    done
}

main() {
    trap cleanup_pid_files EXIT
    trap shutdown SIGINT SIGTERM

    mkdir -p "$PID_DIR"
    stop_previous_supervisor
    cleanup_previous_launch
    printf '%s\n' "$$" > "$SUPERVISOR_PID_FILE"
    prepare_environment
    supervise_launch
}

[[ "${BASH_SOURCE[0]}" == "$0" ]] && main "$@"

