#pragma once
#include "yolo_engine.hpp"

#include <yaml-cpp/yaml.h>

#include "trt_infer_msgs/msg/perception_result.hpp"
using trt_infer_msgs::msg::PerceptionResult;

class YOLOPipeline {
public:
  YOLOPipeline(const YAML::Node &config);
  ~YOLOPipeline();

  /**
   * @brief Load parameters from a YAML configuration node.
   *
   * @param config The YAML node containing the configuration parameters.
   */
  void loadParameters(const YAML::Node &config);

  /**
   * @brief Initialize the YOLO pipeline with the loaded parameters.
   */
  void initialize();

  /**
   * @brief Process RGB and depth images using the YOLO engine.
   *
   * @param rgb The input RGB image (cv::Mat).
   * @param depth The input depth image (cv::Mat).
   * @param perception_result The output perception result message to be filled
   * with detection results.
   */
  void process(const cv::Mat &rgb, const cv::Mat &depth,
               PerceptionResult &perception_result);

  /**
   * @brief Get the Engine Path object
   *
   * @return std::string
   */
  std::string getEnginePath() const { return yolo_engine_path_; }

  /**
   * @brief Check if the YOLO pipeline is enabled.
   *
   * @return true if enabled, false otherwise.
   */
  bool isEnabled() const { return enabled_; }

private:
  std::unique_ptr<YOLOEngine> yolo_engine_ptr_;

  bool enabled_{false};

  std::string yolo_engine_name_;
  std::string yolo_engine_path_;

  float conf_{0.45f};          // Confidence threshold for YOLO detections
  float max_distance_m_{5.0f}; // 最大距离阈值，单位为米
  float depth_scale_to_meters_{
      0.001f};                    // 深度图像的缩放因子，将深度值从毫米转换为米
  float min_depth_m_{0.08f};      // 最小深度阈值，单位为米
  float max_depth_read_m_{25.0f}; // 最大深度读取阈值，单位为米
  bool only_human_class_{true};   // 是否只检测人体类别
  bool distance_ema_enable_{false}; // 是否启用距离的指数移动平均（EMA）滤波
  int human_class_id_{0};           // 人体类别的ID
  double depth_roi_y0_{0.52};       // 深度图像的ROI区域的上边界归一化坐标
  double depth_roi_y1_{0.98};       // 深度图像的ROI区域的下边界归一化坐标
  double depth_roi_x_margin_{0.2};  // 深度图像的ROI区域的左右边界的归一化边距
  double depth_percentile_{0.5};    // 深度图像的百分位数，用于深度值的统计分析
  double depth_trim_close_ratio_{
      0.0}; // 深度图像的裁剪比例，用于去除过近的深度值
  float distance_ema_alpha_{
      0.35f}; // 指数移动平均（EMA）滤波的平滑系数，用于距离值的平滑处理

  std::string bbox_space; // Bounding box coordinate space, can be "letterbox"
                          // or "original"
  bool bbox_in_original_space; // Whether the bounding box coordinates are in
                               // the original image space
  int engine_input_height;     // Input height for the YOLO engine
  int engine_input_width;      // Input width for the YOLO engine

  std::vector<float>
      distance_ema_state_; // 指数移动平均（EMA）滤波的状态，用于存储每个目标的平滑距离值
  std::vector<uint8_t>
      distance_ema_valid_; // 指数移动平均（EMA）滤波的有效性标志，用于指示每个目标的距离值是否有效
};