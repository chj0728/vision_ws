#include "perception_pipeline.hpp"
#include <iostream>
PerceptionPipeline::PerceptionPipeline(YAML::Node &config) : config_(config) {

  initialize();

  std::cout << "[All PerceptionPipelines]" << std::endl;
  std::cout << "|--> initialized successfully." << std::endl;
  std::cout << "|____________________________" << std::endl;
}

PerceptionPipeline::~PerceptionPipeline() = default;

void PerceptionPipeline::initialize() {

  // Initialize the YOLO pipeline
  yolo_pipeline_ptr_ = std::make_unique<YOLOPipeline>(config_);
  std::cout << "[YOLOPipeline]:" << std::endl;
  std::cout << "|-->loaded engine from: " << yolo_pipeline_ptr_->getEnginePath()
            << std::endl;
  std::cout << "|--> is initialized successfully." << std::endl;
  std::cout << "|____________________________" << std::endl;

  scrfd_pipeline_ptr_ = std::make_unique<SCRFDPipeline>(config_);
  std::cout << "[SCRFDPipeline]:" << std::endl;
  if (scrfd_pipeline_ptr_->isEnabled()) {
    std::cout << "|-->loaded engine from: "
              << scrfd_pipeline_ptr_->getEnginePath() << std::endl;
    std::cout << "|--> is initialized successfully." << std::endl;
  } else {
    std::cout << "|--> disabled." << std::endl;
  }
  std::cout << "|____________________________" << std::endl;
}

void PerceptionPipeline::process(const cv::Mat &rgb, const cv::Mat &depth,
                                 PerceptionResult &perception_result) {
  // Process the RGB and depth images using the YOLO engine

  std::cout << "[PerceptionPipeline] Processing RGB and depth images..."
            << std::endl;
  if (yolo_pipeline_ptr_) {
    yolo_pipeline_ptr_->process(rgb, depth, perception_result);
  }
  if (scrfd_pipeline_ptr_) {
    scrfd_pipeline_ptr_->process(rgb, perception_result);
  }
} // namespace perception_pipeline
