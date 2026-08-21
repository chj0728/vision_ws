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
  if (yolo_pipeline_ptr_->isEnabled()) {
    std::cout << "|-->loaded engine from: "
              << yolo_pipeline_ptr_->getEnginePath() << std::endl;
    std::cout << "|--> is initialized successfully." << std::endl;
  } else {
    std::cout << "|--> disabled." << std::endl;
  }
  std::cout << "|____________________________" << std::endl;

  iou_tracker_ptr_ = std::make_unique<IouTracker>(config_);
  std::cout << "[IouTracker]:" << std::endl;
  std::cout << (iou_tracker_ptr_->isEnabled() ? "|--> enabled."
                                              : "|--> disabled.")
            << std::endl;
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

  sixdrepnet_pipeline_ptr_ = std::make_unique<SixDRepNetPipeline>(config_);
  std::cout << "[SixDRepNetPipeline]:" << std::endl;
  if (sixdrepnet_pipeline_ptr_->isEnabled()) {
    std::cout << "|-->loaded engine from: "
              << sixdrepnet_pipeline_ptr_->getEnginePath() << std::endl;
    std::cout << "|--> is initialized successfully." << std::endl;
  } else {
    std::cout << "|--> disabled." << std::endl;
  }
  std::cout << "|____________________________" << std::endl;

  arcface_pipeline_ptr_ = std::make_unique<ArcFacePipeline>(config_);
  std::cout << "[ArcFacePipeline]:" << std::endl;
  if (arcface_pipeline_ptr_->isEnabled()) {
    std::cout << "|-->loaded engine from: "
              << arcface_pipeline_ptr_->getEnginePath() << std::endl;
    std::cout << "|-->opened database: "
              << arcface_pipeline_ptr_->getDatabasePath() << std::endl;
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
  perception_result.persons.clear();
  perception_result.body_pipeline_ms = 0.0f;
  if (yolo_pipeline_ptr_) {
    yolo_pipeline_ptr_->process(rgb, depth, perception_result);
  }

  PerceptionFrameContext frame_context;
  frame_context.frame_number = ++frame_number_;
  if (iou_tracker_ptr_) {
    iou_tracker_ptr_->process(perception_result, frame_context);
  }
  if (scrfd_pipeline_ptr_) {
    scrfd_pipeline_ptr_->process(rgb, perception_result, frame_context);
  }
  if (sixdrepnet_pipeline_ptr_) {
    sixdrepnet_pipeline_ptr_->process(rgb, frame_context, perception_result);
  }
  if (arcface_pipeline_ptr_) {
    arcface_pipeline_ptr_->process(rgb, frame_context, perception_result);
  }
} // namespace perception_pipeline
