/**
 * @file perception_pipeline.hpp
 * @brief Perception pipeline header file.
 * @author Cao Haojie
 * @date 2024-06-20
 */

#ifndef PERCEPTION_PIPELINE_HPP
#define PERCEPTION_PIPELINE_HPP

#include <memory>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "pipeline/arcface_pipeline.hpp"
#include "pipeline/iou_tracker.hpp"
#include "pipeline/scrfd_pipeline.hpp"
#include "pipeline/yolo_pipeline.hpp"

#include "trt_infer_msgs/msg/perception_result.hpp"
using trt_infer_msgs::msg::PerceptionResult;

class PerceptionPipeline {

private:
  YAML::Node config_;
  std::unique_ptr<YOLOPipeline> yolo_pipeline_ptr_;
  std::unique_ptr<IouTracker> iou_tracker_ptr_;
  std::unique_ptr<SCRFDPipeline> scrfd_pipeline_ptr_;
  std::unique_ptr<ArcFacePipeline> arcface_pipeline_ptr_;
  int frame_number_{0};

public:
  PerceptionPipeline(YAML::Node &config);
  ~PerceptionPipeline();

  /**
   * @brief 初始化感知模块
   */
  void initialize();

  /**
   * @brief 处理RGB和深度图像，生成感知结果
   *
   * @param rgb
   * @param depth
   * @param perception_result
   */
  void process(const cv::Mat &rgb, const cv::Mat &depth,
               PerceptionResult &perception_result);
};

#endif // PERCEPTION_PIPELINE_HPP