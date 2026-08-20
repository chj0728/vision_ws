#pragma once

#include "scrfd_trt/face_detector.h"

#include <set>
#include <vector>

// 仅在单帧 Pipeline 内传递，不进入 ROS 消息。
struct PersonFrameContext {
  int track_id{-1};
  int track_total_frames{0};
  bool has_face{false};
  FaceObject face{};
};

struct PerceptionFrameContext {
  int frame_number{0};
  std::vector<PersonFrameContext> persons;
  std::set<int> retained_track_ids;
};
