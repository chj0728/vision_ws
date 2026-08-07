#!/usr/bin/env bash
set -eo pipefail
export ROS_DOMAIN_ID=25
set +u
source /opt/ros/humble/setup.bash
source /home/orinagx/ros2_infer_ws/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/orinagx/ros2_infer_ws/config/fastdds_no_shm.xml
set -u

# SHOW_WINDOW=true 时需要桌面（登录后）；默认 false 可无头开机
SHOW_WINDOW="${SHOW_WINDOW:-false}"
if [[ "${SHOW_WINDOW}" == "true" || "${SHOW_WINDOW}" == "1" ]]; then
  export DISPLAY="${DISPLAY:-:1}"
  for _auth in /run/user/1000/gdm/Xauthority /home/orinagx/.Xauthority; do
    if [[ -f "${_auth}" ]]; then
      export XAUTHORITY="${_auth}"
      break
    fi
  done
  echo "Waiting for display ${DISPLAY}..."
  display_timeout=120
  display_elapsed=0
  x_sock="/tmp/.X11-unix/X${DISPLAY#:}"
  while [[ ! -S "${x_sock}" ]]; do
    sleep 2
    display_elapsed=$((display_elapsed + 2))
    if [[ "${display_elapsed}" -ge "${display_timeout}" ]]; then
      echo "ERROR: display ${DISPLAY} not ready after ${display_timeout}s"
      exit 1
    fi
  done
  echo "Display ready: ${DISPLAY}"
else
  echo "Headless mode (SHOW_WINDOW=false), skip display wait"
fi

echo "Waiting for camera topics..."
timeout_sec=120
elapsed=0
while ! ros2 topic list 2>/dev/null | grep -q '/camera/color/image_raw'; do
  sleep 2
  elapsed=$((elapsed + 2))
  if [[ "${elapsed}" -ge "${timeout_sec}" ]]; then
    echo "ERROR: /camera/color/image_raw not available after ${timeout_sec}s"
    exit 1
  fi
done
echo "Camera topics ready, starting face fusion (show_window=${SHOW_WINDOW})..."
exec ros2 launch human_face_fusion_ros human_face_fusion_with_trt.launch.py \
    model_path:=/home/orinagx/ros2_infer_ws/models/yolo26m_fp16.engine \
    engine_path:=/home/orinagx/ros2_infer_ws/models/scrfd_2.5g_bnkps_shape640x640.trt \
    image_topic:=/camera/color/image_raw \
    depth_topic:=/camera/depth/image_raw \
    head_pose_enable:=true \
    head_pose_engine_path:=/home/orinagx/ros2_infer_ws/models/sixdrepnet_fp16.trt \
    body_log_frame_timing:=true \
    fusion_log_frame_timing:=true \
    show_window:=${SHOW_WINDOW} \
    face_recog_enable:=true \
    face_recog_engine_path:=/home/orinagx/ros2_infer_ws/models/w600k_r50_b16_gpu0_fp16.engine \
    face_db_path:=/home/orinagx/ros2_infer_ws/data/face_db.sqlite3 \
    gender_enable:=true \
    gender_engine_path:=/home/orinagx/ros2_infer_ws/models/genderage.engine \
    color_sub_qos:=reliable
