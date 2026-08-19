#pragma once

#include "scrfd_trt/scrfd_trt.h"

#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "trt_infer_msgs/msg/perception_result.hpp"

class SCRFDPipeline {
public:
  explicit SCRFDPipeline(const YAML::Node &config);
  ~SCRFDPipeline();

  void loadParameters(const YAML::Node &config);
  void initialize();
  void process(const cv::Mat &rgb,
               trt_infer_msgs::msg::PerceptionResult &perception_result);

  bool isEnabled() const { return enabled_; }
  std::string getEnginePath() const { return scrfd_engine_path_; }

private:
  std::unique_ptr<SCRFD_TRT> scrfd_engine_ptr_;

  bool enabled_{false};
  std::string scrfd_engine_name_;
  std::string scrfd_engine_path_;
  std::string preprocess_;
  float prob_threshold_{0.38f};
  float nms_threshold_{0.45f};
  double person_head_height_ratio_{0.58};
  double person_head_width_pad_ratio_{0.24};
  double person_head_top_expand_ratio_{0.14};
  int face_roi_min_side_{48};
  int max_person_rois_{8};
};
