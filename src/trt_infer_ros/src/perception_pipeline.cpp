#include "perception_pipeline.hpp"
#include <iostream>
PerceptionPipeline::PerceptionPipeline(YAML::Node &config) : config_(config) {

  initialize();

  std::cout << "[PerceptionPipelines] is initialized successfully."
            << std::endl;
}

PerceptionPipeline::~PerceptionPipeline() = default;

void PerceptionPipeline::initialize() {

  // Initialize the YOLO pipeline with the loaded parameters
  yolo_pipeline_ptr_ = std::make_unique<YOLOPipeline>(config_);
  std::cout << "[YOLOPipeline] is initialized successfully." << std::endl;
}

void PerceptionPipeline::process(const cv::Mat &rgb, const cv::Mat &depth,
                                 PerceptionResult &perception_result) {
  // Process the RGB and depth images using the YOLO engine

  std::cout << "[PerceptionPipeline] Processing RGB and depth images..."
            << std::endl;
  if (yolo_pipeline_ptr_) {
    yolo_pipeline_ptr_->process(rgb, depth, perception_result);
  }
} // namespace perception_pipeline