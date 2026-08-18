#include "pipeline/yolo_pipeline.hpp"
#include <trt_infer_msgs/msg/detail/person_meta__struct.hpp>

#include <iostream>

YOLOPipeline::YOLOPipeline(YAML::Node &config) {
  this->loadParameters(config);
  this->initialize();
}

YOLOPipeline::~YOLOPipeline() {}

void YOLOPipeline::loadParameters(YAML::Node &config) {

  config = config["yolo_pipeline"];

  yolo_engine_path = config["yolo_engine_path"].as<std::string>(
      "/home/caohaojie/ws/vision_ws/models/yolo/yolo26m_fp16.engine");

  conf_ = config["conf_threshold"].as<float>(0.45f);
  max_distance_m_ = config["max_distance_m"].as<float>(5.0f);
  depth_scale_to_meters_ = config["depth_scale_to_meters"].as<float>(0.001f);
  min_depth_m_ = config["min_depth_m"].as<float>(0.08f);
  max_depth_read_m_ = config["max_depth_read_m"].as<float>(25.0f);
  only_human_class_ = config["only_human_class"].as<bool>(true);
  distance_ema_enable_ = config["distance_ema_enable"].as<bool>(false);
  human_class_id_ = config["human_class_id"].as<int>(0);
  depth_roi_y0_ = config["depth_roi_y0"].as<double>(0.52);
  depth_roi_y1_ = config["depth_roi_y1"].as<double>(0.98);
  depth_roi_x_margin_ = config["depth_roi_x_margin"].as<double>(0.2);
  depth_percentile_ = config["depth_percentile"].as<double>(0.5);
  depth_trim_close_ratio_ = config["depth_trim_close_ratio"].as<double>(0.0);
  distance_ema_alpha_ = config["distance_ema_alpha"].as<float>(0.35f);

  bbox_space = config["bbox_coord_space"].as<std::string>("letterbox");
  bbox_in_original_space = (bbox_space == "original" || bbox_space == "orig");
  engine_input_height = config["engine_input_h"].as<int>(0);
  engine_input_width = config["engine_input_w"].as<int>(0);
}

void YOLOPipeline::initialize() {
  // Initialize the YOLO pipeline with the loaded parameters
  yolo_engine_ptr_ = std::make_unique<YOLOEngine>(
      yolo_engine_path, conf_, bbox_in_original_space, engine_input_height,
      engine_input_width);
}

void YOLOPipeline::process(const cv::Mat &rgb, const cv::Mat &depth,
                           PerceptionResult &perception_result) {
  // Process the RGB and depth images using the YOLO engine
  if (yolo_engine_ptr_) {

    auto start_time_ = std::chrono::high_resolution_clock::now();

    const auto detections = yolo_engine_ptr_->inferWithDepth(
        rgb, depth, conf_, min_depth_m_, max_depth_read_m_,
        static_cast<float>(depth_roi_y0_), static_cast<float>(depth_roi_y1_),
        static_cast<float>(depth_roi_x_margin_),
        static_cast<float>(depth_trim_close_ratio_),
        static_cast<float>(depth_percentile_));

    if (detections.size() != distance_ema_state_.size()) {
      distance_ema_state_.resize(detections.size());
      distance_ema_valid_.assign(detections.size(), 0);
    }

    for (size_t index = 0; index < detections.size(); ++index) {
      const Detection &detection = detections[index];
      if (only_human_class_ && detection.class_id != human_class_id_)
        continue;

      float distance = detection.distance;
      const bool depth_valid = distance > 0.0f && distance <= max_distance_m_;
      if (depth_valid && distance_ema_enable_) {
        if (!distance_ema_valid_[index]) {
          distance_ema_state_[index] = distance;
          distance_ema_valid_[index] = 1;
        } else {
          distance_ema_state_[index] =
              distance_ema_alpha_ * distance +
              (1.0f - distance_ema_alpha_) * distance_ema_state_[index];
        }
        distance = distance_ema_state_[index];
      }

      trt_infer_msgs::msg::PersonMeta person;
      person.body_detection.body_bbox.x = detection.x;
      person.body_detection.body_bbox.y = detection.y;
      person.body_detection.body_bbox.w = detection.w;
      person.body_detection.body_bbox.h = detection.h;

      person.body_detection.body_confidence = detection.conf;
      person.body_detection.body_distance =
          depth_valid ? std::round(distance * 1000.0f) / 1000.0f
                      : -1.0f; // Round to 3 decimal places
      perception_result.persons.push_back(std::move(person));
    }

    std::chrono::duration<float, std::milli> pipeline_duration =
        std::chrono::high_resolution_clock::now() - start_time_;
    std::cout << "[YOLOPipeline] Processing time: " << pipeline_duration.count()
              << " ms" << std::endl;
    perception_result.body_pipeline_ms = pipeline_duration.count();
  }
}