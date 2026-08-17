#include "pipeline/yolo_pipeline.hpp"

YOLOPipeline::YOLOPipeline(YAML::Node & config) {
  this->loadParameters(config);
  this->initialize();
}

YOLOPipeline::~YOLOPipeline() {
}

void YOLOPipeline::loadParameters(YAML::Node & config) {

  config = config["yolo_pipeline"];
  
  yolo_engine_path = config["yolo_engine_path"].as<std::string>("/home/caohaojie/ws/vision_ws/models/yolo/yolo26m_fp16.engine");
  
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
  yolo_engine_ptr_ = std::make_unique<YOLOEngine>(yolo_engine_path, conf_, bbox_in_original_space, engine_input_height, engine_input_width);
}