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

#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <trt_infer_msgs/msg/perception_result.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <yaml-cpp/yaml.h>

#include "perception_pipeline.hpp"

namespace perception_ros_component {

using Image = sensor_msgs::msg::Image;
using ExactSyncPolicy = message_filters::sync_policies::ExactTime<Image, Image>;
using ApproximateSyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;

class PerceptionRosComponent : public rclcpp::Node {
public:
  explicit PerceptionRosComponent(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~PerceptionRosComponent() override;

  /**
   * @brief Load parameters from a YAML configuration file.
   * 
   */
  void loadParameters();

  /**
   * @brief 解码深度图像消息为浮点米单位的深度图像
   * 
   * @param depth_msg 
   * @param depth_meters 
   * @return true 
   * @return false 
   */
	bool decodeToFloatMeters(const Image::ConstSharedPtr& depth_msg, cv::Mat& depth_meters);

  /**
   * @brief 同步RGB和深度图像的回调函数
   * 
   * @param color_msg 
   * @param depth_msg 
   */
  void onSyncedColorDepth(const Image::ConstSharedPtr& color_msg, const Image::ConstSharedPtr& depth_msg);

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
  void processColorDepth(const Image::ConstSharedPtr& color_msg, const Image::ConstSharedPtr& depth_msg);

  /**
   * @brief 将字符串转换为小写
   * 
   * @param value
   * @return std::string 
   */
  std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
  }

  /**
   * @brief 检查图像消息的编码是否为JPEG格式
   * 
   * @param encoding 
   * @return true 
   * @return false 
   */
  bool isJpegInImageMsg(const std::string& encoding) {
    const std::string normalized = toLower(encoding);
    return normalized.find("jpeg") != std::string::npos || normalized.find("jpg") != std::string::npos ||
        normalized.find("mjpeg") != std::string::npos || normalized.find("mjpg") != std::string::npos;
  }

  /**
   * @brief 确保图像为BGR8格式, 
   *        Converts accepted camera image layouts into the BGR8 format required by YOLO.
   * 
   * @param image 
   */
  void ensureBgrU8C3(cv::Mat& image) {
    if (image.empty() || image.cols <= 0 || image.rows <= 0) {
      image.release();
      return;
    }
    try {
      if (image.depth() != CV_8U) {
        cv::Mat u8;
        image.convertTo(u8, CV_8U);
        image = std::move(u8);
      }
      if (image.channels() == 3) return;

      cv::Mat bgr;
      if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
      } else if (image.channels() == 4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
      } else {
        image.release();
        return;
      }
      image = std::move(bgr);
    } catch (const cv::Exception&) {
      image.release();
    }
  }

private:

  // config path
  std::string pipeline_config_path_;

  // Pointer to the perception pipeline
  std::unique_ptr<PerceptionPipeline> perception_pipeline_ptr_;

  // synchronization parameters
  bool hard_sync_{false}; // Whether to use exact synchronization or approximate synchronization
  int sync_queue_size_{10}; // Queue size for message synchronization
  double processing_rate_hz_{10.0}; // Processing rate in Hz for the timer callback

  // Publisher for perception results
  std::string perception_result_topic_;
  rclcpp::Publisher<trt_infer_msgs::msg::PerceptionResult>::SharedPtr perception_result_pub_;

  // Subscribers for RGB and depth images
  std::string color_image_topic_;
  std::string depth_image_topic_;
  message_filters::Subscriber<Image> color_image_sub_;
  message_filters::Subscriber<Image> depth_image_sub_;
  std::unique_ptr<message_filters::Synchronizer<ExactSyncPolicy>> sync_exact_;
  std::unique_ptr<message_filters::Synchronizer<ApproximateSyncPolicy>> sync_approx_;
  Image::ConstSharedPtr latest_color_msg_;
  Image::ConstSharedPtr latest_depth_msg_;
  std::mutex latest_frames_mutex_;
  rclcpp::TimerBase::SharedPtr processing_timer_;

  cv::Mat depth_f32_buf_;               // 
  float depth_scale_to_meters_{0.001f}; // 深度图像的缩放因子，将深度值从毫米转换为米
};

}  // namespace perception_ros_component

#endif // PERCEPTION_ROS_COMPONENT_HPP