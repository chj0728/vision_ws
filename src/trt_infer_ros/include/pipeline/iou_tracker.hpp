#pragma once

#include "pipeline/perception_frame_context.hpp"

#include <chrono>
#include <map>
#include <set>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "trt_infer_msgs/msg/perception_result.hpp"

// 仅根据人体框维护轨迹，不保存任何人脸识别业务状态。
class IouTracker {
public:
  explicit IouTracker(const YAML::Node &config);

  void loadParameters(const YAML::Node &config);
  void process(trt_infer_msgs::msg::PerceptionResult &perception_result,
               PerceptionFrameContext &frame_context);

  bool isEnabled() const { return enabled_; }
  int liveCount() const;

private:
  struct Track {
    int track_id{-1};
    cv::Rect body_bbox;
    int age_frames{0};
    int total_frames{0};
    bool is_dead{false};
    std::chrono::steady_clock::time_point dead_since{};

    float iouWith(const cv::Rect &other) const;
  };

  Track *tryRevive(const cv::Rect &detection);
  void ageTracks();
  void retireAndPurgeTracks();
  std::set<int> retainedTrackIds() const;

  bool enabled_{true};
  float iou_threshold_{0.30f};
  int max_age_frames_{30};
  int reid_window_seconds_{30};
  float revive_iou_scale_{0.40f};
  int next_id_{0};
  std::map<int, Track> tracks_;
};
