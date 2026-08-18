#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "trt_infer_msgs/msg/scene_perception_result.hpp"
#include "yolo_engine.hpp"

namespace trt_infer_ros {

/** Synchronizes RGB-D frames and publishes YOLO person detections with distance
 * estimates. */
class TrtInferComponent : public rclcpp::Node {
public:
  explicit TrtInferComponent(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  using Image = sensor_msgs::msg::Image;
  using ExactSyncPolicy =
      message_filters::sync_policies::ExactTime<Image, Image>;
  using ApproximateSyncPolicy =
      message_filters::sync_policies::ApproximateTime<Image, Image>;

  bool decodeToFloatMeters(const Image::ConstSharedPtr &depth_msg,
                           cv::Mat &depth_meters);
  void onSyncedColorDepth(const Image::ConstSharedPtr &color_msg,
                          const Image::ConstSharedPtr &depth_msg);

  std::string model_path_;
  std::string image_topic_;
  std::string depth_topic_;
  std::string out_topic_;
  float conf_{0.45f};
  float max_distance_m_{5.0f};
  float depth_scale_to_meters_{0.001f};
  float min_depth_m_{0.08f};
  float max_depth_read_m_{25.0f};
  bool only_human_class_{true};
  bool depth_sync_exact_{false};
  bool distance_ema_enable_{false};
  int human_class_id_{0};
  uint32_t sync_queue_size_{40};
  double depth_roi_y0_{0.52};
  double depth_roi_y1_{0.98};
  double depth_roi_x_margin_{0.2};
  double depth_percentile_{0.5};
  double depth_trim_close_ratio_{0.0};
  float distance_ema_alpha_{0.35f};

  std::vector<float> distance_ema_state_;
  std::vector<uint8_t> distance_ema_valid_;
  cv::Mat depth_f32_buf_;
  std::unique_ptr<YOLOEngine> engine_;
  rclcpp::Publisher<trt_infer_msgs::msg::ScenePerceptionResult>::SharedPtr pub_;
  message_filters::Subscriber<Image> color_sub_;
  message_filters::Subscriber<Image> depth_sub_;
  std::unique_ptr<message_filters::Synchronizer<ExactSyncPolicy>> sync_exact_;
  std::unique_ptr<message_filters::Synchronizer<ApproximateSyncPolicy>>
      sync_approx_;
};

} // namespace trt_infer_ros
