#include "trt_infer_ros_component.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <utility>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/synchronizer.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rmw/qos_profiles.h>
#include <sensor_msgs/image_encodings.hpp>

#include "trt_infer_msgs/msg/person_perception.hpp"

namespace {

std::string toLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool isJpegInImageMsg(const std::string &encoding) {
  const std::string normalized = toLower(encoding);
  return normalized.find("jpeg") != std::string::npos ||
         normalized.find("jpg") != std::string::npos ||
         normalized.find("mjpeg") != std::string::npos ||
         normalized.find("mjpg") != std::string::npos;
}

// Converts accepted camera image layouts into the BGR8 format required by YOLO.
void ensureBgrU8C3(cv::Mat &image) {
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
    if (image.channels() == 3)
      return;

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
  } catch (const cv::Exception &) {
    image.release();
  }
}

} // namespace

namespace trt_infer_ros {

TrtInferComponent::TrtInferComponent(const rclcpp::NodeOptions &options)
    : Node("trt_infer_ros_node", options) {
  model_path_ =
      declare_parameter<std::string>("model_path", "yolo26m_fp16.engine");
  conf_ = static_cast<float>(declare_parameter<double>("conf_threshold", 0.45));
  image_topic_ =
      declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
  depth_topic_ =
      declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
  out_topic_ = declare_parameter<std::string>("detections_topic",
                                              "/remote_infer/detections");
  max_distance_m_ =
      static_cast<float>(declare_parameter<double>("max_distance_m", 5.0));
  only_human_class_ = declare_parameter<bool>("only_human_class", true);
  human_class_id_ = declare_parameter<int>("human_class_id", 0);
  depth_sync_exact_ = declare_parameter<bool>("depth_sync_exact", false);
  sync_queue_size_ = static_cast<uint32_t>(
      declare_parameter<int>("depth_sync_queue_size", 40));
  depth_scale_to_meters_ = static_cast<float>(
      declare_parameter<double>("depth_scale_to_meters", 0.001));
  depth_roi_y0_ = declare_parameter<double>("depth_roi_y0", 0.52);
  depth_roi_y1_ = declare_parameter<double>("depth_roi_y1", 0.98);
  depth_roi_x_margin_ = declare_parameter<double>("depth_roi_x_margin", 0.2);
  min_depth_m_ =
      static_cast<float>(declare_parameter<double>("min_depth_m", 0.08));
  max_depth_read_m_ =
      static_cast<float>(declare_parameter<double>("max_depth_read_m", 25.0));
  depth_percentile_ = declare_parameter<double>("depth_percentile", 0.5);
  depth_trim_close_ratio_ =
      declare_parameter<double>("depth_trim_close_ratio", 0.0);
  distance_ema_enable_ = declare_parameter<bool>("distance_ema_enable", false);
  distance_ema_alpha_ =
      static_cast<float>(declare_parameter<double>("distance_ema_alpha", 0.35));

  const std::string bbox_space =
      declare_parameter<std::string>("bbox_coord_space", "letterbox");
  const bool bbox_in_original_space =
      bbox_space == "original" || bbox_space == "orig";
  const int engine_input_height = declare_parameter<int>("engine_input_h", 0);
  const int engine_input_width = declare_parameter<int>("engine_input_w", 0);
  engine_ =
      std::make_unique<YOLOEngine>(model_path_, conf_, bbox_in_original_space,
                                   engine_input_height, engine_input_width);

  pub_ = create_publisher<trt_infer_msgs::msg::ScenePerceptionResult>(
      out_topic_, rclcpp::QoS(2).reliable());

  rclcpp::QoS image_qos(
      rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default),
      rmw_qos_profile_default);
  image_qos.keep_last(6);
  const rmw_qos_profile_t qos_profile = image_qos.get_rmw_qos_profile();
  color_sub_.subscribe(this, image_topic_, qos_profile);
  depth_sub_.subscribe(this, depth_topic_, qos_profile);

  if (depth_sync_exact_) {
    sync_exact_ =
        std::make_unique<message_filters::Synchronizer<ExactSyncPolicy>>(
            ExactSyncPolicy(sync_queue_size_), color_sub_, depth_sub_);
    sync_exact_->registerCallback(
        std::bind(&TrtInferComponent::onSyncedColorDepth, this,
                  std::placeholders::_1, std::placeholders::_2));
  } else {
    sync_approx_ =
        std::make_unique<message_filters::Synchronizer<ApproximateSyncPolicy>>(
            ApproximateSyncPolicy(sync_queue_size_), color_sub_, depth_sub_);
    sync_approx_->registerCallback(
        std::bind(&TrtInferComponent::onSyncedColorDepth, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
}

bool TrtInferComponent::decodeToFloatMeters(
    const Image::ConstSharedPtr &depth_msg, cv::Mat &depth_meters) {
  if (!depth_msg)
    return false;

  const std::string encoding = toLower(depth_msg->encoding);
  if (encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
      encoding == "16uc1") {
    const auto depth = cv_bridge::toCvShare(
        depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    if (depth_f32_buf_.rows != static_cast<int>(depth_msg->height) ||
        depth_f32_buf_.cols != static_cast<int>(depth_msg->width)) {
      depth_f32_buf_.create(depth_msg->height, depth_msg->width, CV_32F);
    }
    depth->image.convertTo(depth_f32_buf_, CV_32F,
                           static_cast<double>(depth_scale_to_meters_));
    depth_meters = depth_f32_buf_;
    return true;
  }
  if (encoding == sensor_msgs::image_encodings::TYPE_32FC1 ||
      encoding == "32fc1") {
    depth_meters = cv_bridge::toCvShare(
                       depth_msg, sensor_msgs::image_encodings::TYPE_32FC1)
                       ->image;
    return true;
  }
  return false;
}

void TrtInferComponent::onSyncedColorDepth(
    const Image::ConstSharedPtr &color_msg,
    const Image::ConstSharedPtr &depth_msg) {
  const auto start_time = std::chrono::steady_clock::now();
  if (!color_msg || !depth_msg)
    return;

  const bool is_jpeg =
      isJpegInImageMsg(color_msg->encoding) ||
      (color_msg->data.size() >= 2 && color_msg->data[0] == 0xff &&
       color_msg->data[1] == 0xd8);
  cv::Mat image;
  if (is_jpeg) {
    image = cv::imdecode(cv::Mat(1, color_msg->data.size(), CV_8UC1,
                                 const_cast<uint8_t *>(color_msg->data.data())),
                         cv::IMREAD_COLOR);
  } else {
    image = cv_bridge::toCvShare(color_msg, sensor_msgs::image_encodings::BGR8)
                ->image;
  }
  ensureBgrU8C3(image);
  if (image.empty())
    return;

  cv::Mat depth_meters;
  if (!decodeToFloatMeters(depth_msg, depth_meters))
    return;

  const auto detections = engine_->inferWithDepth(
      image, depth_meters, conf_, min_depth_m_, max_depth_read_m_,
      static_cast<float>(depth_roi_y0_), static_cast<float>(depth_roi_y1_),
      static_cast<float>(depth_roi_x_margin_),
      static_cast<float>(depth_trim_close_ratio_),
      static_cast<float>(depth_percentile_));

  trt_infer_msgs::msg::ScenePerceptionResult result;
  result.image_width = static_cast<uint32_t>(image.cols);
  result.image_height = static_cast<uint32_t>(image.rows);

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

    trt_infer_msgs::msg::PersonPerception person;
    person.body_x = detection.x;
    person.body_y = detection.y;
    person.body_w = detection.w;
    person.body_h = detection.h;
    person.body_conf = detection.conf;
    person.distance =
        depth_valid ? std::round(distance * 1000.0f) / 1000.0f : -1.0f;
    person.has_face = false;
    result.persons.push_back(std::move(person));
  }

  result.header = color_msg->header;
  result.body_pipeline_ms = std::chrono::duration<float, std::milli>(
                                std::chrono::steady_clock::now() - start_time)
                                .count();
  pub_->publish(result);
}

} // namespace trt_infer_ros

RCLCPP_COMPONENTS_REGISTER_NODE(trt_infer_ros::TrtInferComponent)
