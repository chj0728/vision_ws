#include "perception_pipeline.hpp"
#include <iostream>
PerceptionPipeline::PerceptionPipeline(YAML::Node & config):
  config_(config)
{

  initialize();

  std::cout << "PerceptionPipeline constructed." << std::endl;
}

PerceptionPipeline::~PerceptionPipeline() = default;

void PerceptionPipeline::initialize() {

  // Initialize the YOLO pipeline with the loaded parameters
  yolo_pipeline_ptr_ = std::make_unique<YOLOPipeline>(config_);
  std::cout << "YOLOPipeline is initialized successfully." << std::endl;

}

void PerceptionPipeline::process(
    const cv::Mat& rgb,
    const cv::Mat& depth,
    PerceptionResult& perception_result) {
  // Process the RGB and depth images using the YOLO engine
  
    std::cout << "Processing RGB and depth images..." << std::endl;
} // namespace perception_pipeline