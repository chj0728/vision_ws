/**
 * @file perception_ros_component.hpp
 * @author your name (you@domain.com)
 * @brief ROS2 component for perception pipeline that synchronizes RGB-D frames
 * @version 0.1
 * @date 2026-08-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifndef PERCEPTION_ROS_COMPONENT_HPP
#define PERCEPTION_ROS_COMPONENT_HPP

#include <cctype>
#include <mutex>
#include <string>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trt_infer_msgs/msg/detail/person_meta__struct.hpp>
#include <trt_infer_msgs/msg/interaction_result.hpp>
#include <trt_infer_msgs/msg/perception_result.hpp>
#include <trt_infer_msgs/msg/scene_perception_result.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <yaml-cpp/yaml.h>

#include "perception_pipeline.hpp"

struct InteractionStruct {
  InteractionStruct() = default;
  InteractionStruct(const float &ayaw, const float &apitch,
                    const float &amax_dist, const float &amin_dist,
                    const float &tyaw, const float &tpitch,
                    const float &tmax_dist, const float &tmin_dist)
      : attention_yaw_deg(ayaw), attention_pitch_deg(apitch),
        attention_max_distance_m(amax_dist),
        attention_min_distance_m(amin_dist), talking_yaw_deg(tyaw),
        talking_pitch_deg(tpitch), talking_max_distance_m(tmax_dist),
        talking_min_distance_m(tmin_dist) {}
  float attention_yaw_deg{0.f};
  float attention_pitch_deg{0.f};
  float attention_max_distance_m{0.f};
  float attention_min_distance_m{0.f};

  float talking_yaw_deg{0.f};
  float talking_pitch_deg{0.f};
  float talking_max_distance_m{0.f};
  float talking_min_distance_m{0.f};

  void update(const float &ayaw, const float &apitch, const float &amax_dist,
              const float &amin_dist, const float &tyaw, const float &tpitch,
              const float &tmax_dist, const float &tmin_dist) {
    attention_yaw_deg = ayaw;
    attention_pitch_deg = apitch;
    attention_max_distance_m = amax_dist;
    attention_min_distance_m = amin_dist;
    talking_yaw_deg = tyaw;
    talking_pitch_deg = tpitch;
    talking_max_distance_m = tmax_dist;
    talking_min_distance_m = tmin_dist;
  }

  bool isAttention(const float &yaw, const float &pitch,
                   const float &distance) const {
    if (distance <= 0.f || distance > attention_max_distance_m ||
        distance < attention_min_distance_m)
      return false;
    return std::abs(yaw) <= attention_yaw_deg &&
           std::abs(pitch) <= attention_pitch_deg;
  }

  bool isTalking(const float &yaw, const float &pitch,
                 const float &distance) const {
    if (distance <= 0.f || distance > talking_max_distance_m ||
        distance < talking_min_distance_m)
      return false;
    return std::abs(yaw) <= talking_yaw_deg &&
           std::abs(pitch) <= talking_pitch_deg;
  }

  uint8_t getInteractionStatus(const float &yaw, const float &pitch,
                               const float &distance) const {
    if (isAttention(yaw, pitch, distance))
      return 1; // Attention
    if (isTalking(yaw, pitch, distance))
      return 2; // Talking
    return 0;   // None
  }
};

namespace perception_ros_component {

using Image = sensor_msgs::msg::Image;
using CompressedImage = sensor_msgs::msg::CompressedImage;
using ExactSyncPolicy = message_filters::sync_policies::ExactTime<Image, Image>;
using ApproximateSyncPolicy =
    message_filters::sync_policies::ApproximateTime<Image, Image>;
using CompressedExactSyncPolicy =
    message_filters::sync_policies::ExactTime<CompressedImage, CompressedImage>;
using CompressedApproximateSyncPolicy =
    message_filters::sync_policies::ApproximateTime<CompressedImage,
                                                    CompressedImage>;

class PerceptionRosComponent : public rclcpp::Node {
public:
  explicit PerceptionRosComponent(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~PerceptionRosComponent() override;

  /**
   * @brief Declare parameters for the ROS node.
   *
   */
  void declareParameters();

  /**
   * @brief Get parameters for the ROS node.
   *
   */
  void getParameters();

  /**
   * @brief 初始化订阅、发布器以及其他实现细节
   *
   * @note This function should be called after declaring and getting
   * parameters.
   */
  void initImplementation();

  /**
   * @brief 同步RGB和深度图像的回调函数
   *
   * @param color_msg
   * @param depth_msg
   */
  void onSyncedColorDepth(const Image::ConstSharedPtr &color_msg,
                          const Image::ConstSharedPtr &depth_msg);

  /**
   * @brief 同步压缩RGB和深度图像的回调函数
   *
   * @param color_msg
   * @param depth_msg
   */
  void onSyncedCompressedColorDepth(
      const CompressedImage::ConstSharedPtr &color_msg,
      const CompressedImage::ConstSharedPtr &depth_msg);

  /**
   * @brief 同步RGB和深度图像的处理函数，定时器回调
   *
   */
  void processLatestColorDepth();

  /**
   * @brief 处理RGB和深度图像，进行感知推理并发布结果
   *
   * @param color_msg
   * @param depth_msg
   */
  void processColorDepth(const Image::ConstSharedPtr &color_msg,
                         const Image::ConstSharedPtr &depth_msg);

  /**
   * @brief 处理压缩RGB和深度图像，进行感知推理并发布结果
   *
   * @param color_msg
   * @param depth_msg
   */
  void
  processCompressedColorDepth(const CompressedImage::ConstSharedPtr &color_msg,
                              const CompressedImage::ConstSharedPtr &depth_msg);

  /**
   * @brief 处理解码后的RGB和深度图像，进行感知推理并发布结果
   *
   * @param header
   * @param color_image
   * @param depth_meters
   */
  void processDecodedColorDepth(const std_msgs::msg::Header &header,
                                cv::Mat color_image,
                                const cv::Mat &depth_meters);

  /**
   * @brief 在图像上绘制感知结果
   *
   * @param image
   * @param person_meta
   */
  void
  drawPerceptionResultOnImage(cv::Mat &image,
                              const trt_infer_msgs::msg::PersonMeta &person);

  /**
   * @brief 更新交互结果消息
   *
   * @param interaction_result
   * @param person
   */
  void updateInteractionResult(
      trt_infer_msgs::msg::InteractionResult &interaction_result,
      const trt_infer_msgs::msg::PersonMeta &person);

  /**
   * @brief 发布兼容旧版 human_face_fusion 的场景感知结果
   *
   * @param perception_result 当前帧感知结果
   * @param interaction_result 当前帧交互状态汇总
   */
  void publishEngagementResult(
      const trt_infer_msgs::msg::PerceptionResult &perception_result,
      const trt_infer_msgs::msg::InteractionResult &interaction_result);

  /**
   * @brief 保存彩色图像和深度图像的服务回调
   *
   * @param
   * @param response
   */
  void
  saveColorDepth(const std::shared_ptr<std_srvs::srv::Trigger::Request> &,
                 std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  /**
   * @brief 保存彩色图像和检测框的服务回调
   *
   * @param request
   * @param response
   */
  void
  saveColorBbox(const std::shared_ptr<std_srvs::srv::Trigger::Request> &,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  /**
   * @brief 保存所有图像的服务回调
   *
   * @param request
   * @param response
   */
  void
  saveAllImages(const std::shared_ptr<std_srvs::srv::Trigger::Request> &,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response);

  /**
   * @brief 打印感知结果到控制台
   *
   * @param result
   */
  void
  printPerceptionResult(const trt_infer_msgs::msg::PerceptionResult &result);

private:
  // config path
  std::string pipeline_config_path_;

  // Pointer to the perception pipeline
  std::unique_ptr<PerceptionPipeline> perception_pipeline_ptr_;

  // synchronization parameters
  bool hard_sync_{false};   // Whether to use exact synchronization or
                            // approximate synchronization
  int sync_queue_size_{10}; // Queue size for message synchronization
  double processing_rate_hz_{
      10.0}; // Processing rate in Hz for the timer callback

  // Publisher for perception results
  std::string perception_result_topic_;
  rclcpp::Publisher<trt_infer_msgs::msg::PerceptionResult>::SharedPtr
      perception_result_pub_;
  std::string engagement_result_topic_;
  rclcpp::Publisher<trt_infer_msgs::msg::ScenePerceptionResult>::SharedPtr
      engagement_result_pub_;
  std::string color_bbox_topic_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr color_bbox_pub_;

  // Subscribers for RGB and depth images
  bool use_compressed_images_{false};
  std::string color_image_topic_;
  std::string depth_image_topic_;
  std::string color_compressed_topic_;
  std::string depth_compressed_topic_;
  message_filters::Subscriber<Image> color_image_sub_;
  message_filters::Subscriber<Image> depth_image_sub_;
  message_filters::Subscriber<CompressedImage> color_compressed_sub_;
  message_filters::Subscriber<CompressedImage> depth_compressed_sub_;
  std::unique_ptr<message_filters::Synchronizer<ExactSyncPolicy>> sync_exact_;
  std::unique_ptr<message_filters::Synchronizer<ApproximateSyncPolicy>>
      sync_approx_;
  std::unique_ptr<message_filters::Synchronizer<CompressedExactSyncPolicy>>
      compressed_sync_exact_;
  std::unique_ptr<
      message_filters::Synchronizer<CompressedApproximateSyncPolicy>>
      compressed_sync_approx_;
  Image::ConstSharedPtr latest_color_msg_;
  Image::ConstSharedPtr latest_depth_msg_;
  CompressedImage::ConstSharedPtr latest_compressed_color_msg_;
  CompressedImage::ConstSharedPtr latest_compressed_depth_msg_;
  std::mutex latest_frames_mutex_;
  rclcpp::TimerBase::SharedPtr processing_timer_;

  std::mutex latest_images_mutex_;
  cv::Mat latest_color_image_;
  cv::Mat latest_depth_meters_;
  cv::Mat latest_color_bbox_image_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_color_depth_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_color_bbox_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_all_images_service_;

  // Interaction status parameters
  std::string interaction_result_topic_;
  rclcpp::Publisher<trt_infer_msgs::msg::InteractionResult>::SharedPtr
      interaction_result_pub_;
  InteractionStruct interaction_struct_; // Stores the parameters for attention
                                         // and talking detection
};

} // namespace perception_ros_component

#endif // PERCEPTION_ROS_COMPONENT_HPP