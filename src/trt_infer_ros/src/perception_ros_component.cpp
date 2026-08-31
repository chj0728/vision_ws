#include "perception_ros_component.hpp"
#include "perception_common.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

namespace perception_ros_component {

PerceptionRosComponent::PerceptionRosComponent(
    const rclcpp::NodeOptions &options)
    : Node("perception_ros_component", options) {

  declareParameters();

  getParameters();

  initImplementation();
}

PerceptionRosComponent::~PerceptionRosComponent() = default;

void PerceptionRosComponent::declareParameters() {

  // Load parameters from a YAML configuration file
  this->declare_parameter<std::string>("pipeline_config_path", "pipeline.yaml");

  // Declare perception parameters
  this->declare_parameter<bool>("hard_sync", false);
  this->declare_parameter<int>("sync_queue_size", 10);
  this->declare_parameter<double>("processing_rate_hz", 10.0);

  // Declare topics for RGB and depth images
  this->declare_parameter<std::string>("color_image_topic",
                                       "/camera/color/image_raw");
  this->declare_parameter<std::string>("depth_image_topic",
                                       "/camera/depth/image_raw");
  this->declare_parameter<bool>("use_compressed_images", false);
  this->declare_parameter<std::string>("color_compressed_topic",
                                       "/camera/color/image_raw/compressed");
  this->declare_parameter<std::string>(
      "depth_compressed_topic", "/camera/depth/image_raw/compressedDepth");
  this->declare_parameter<std::string>("perception_result_topic",
                                       "/perception/result");
  this->declare_parameter<std::string>("engagement_result_topic",
                                       "/human_face_fusion/scene_perception");
  this->declare_parameter<std::string>("color_bbox_topic",
                                       "/perception/color_bbox");

  // Declare interaction_result parameters
  this->declare_parameter<std::string>("interaction_result_topic",
                                       "/interaction_result");
  this->declare_parameter<double>("attention_yaw_deg", 55.0);
  this->declare_parameter<double>("attention_pitch_deg", 40.0);
  this->declare_parameter<double>("attention_max_distance_m", 4.0);
  this->declare_parameter<double>("attention_min_distance_m", 0.1);
  this->declare_parameter<double>("talking_yaw_deg", 30.0);
  this->declare_parameter<double>("talking_pitch_deg", 25.0);
  this->declare_parameter<double>("talking_max_distance_m", 2.5);
  this->declare_parameter<double>("talking_min_distance_m", 0.1);
}

void PerceptionRosComponent::getParameters() {

  // pipeline configuration path
  pipeline_config_path_ =
      this->get_parameter("pipeline_config_path").as_string();
  RCLCPP_INFO(this->get_logger(), "Loaded pipelines parameters from %s",
              pipeline_config_path_.c_str());

  // 是否启用严格同步
  hard_sync_ = this->get_parameter("hard_sync").as_bool();
  // 同步队列大小
  sync_queue_size_ = this->get_parameter("sync_queue_size").as_int();
  // 推理频率（Hz）
  processing_rate_hz_ = this->get_parameter("processing_rate_hz").as_double();
  if (processing_rate_hz_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(), "Processing rate must be greater than "
                                    "zero. Using default value 1.0 Hz.");
    processing_rate_hz_ = 1.0;
  }

  // 订阅的图像数据话题名
  color_image_topic_ = this->get_parameter("color_image_topic").as_string();
  depth_image_topic_ = this->get_parameter("depth_image_topic").as_string();
  use_compressed_images_ =
      this->get_parameter("use_compressed_images").as_bool();
  color_compressed_topic_ =
      this->get_parameter("color_compressed_topic").as_string();
  depth_compressed_topic_ =
      this->get_parameter("depth_compressed_topic").as_string();

  // 发布带感知结果的图像话题名
  color_bbox_topic_ = this->get_parameter("color_bbox_topic").as_string();

  // 感知推理结果 和 交互逻辑结果话题名
  perception_result_topic_ =
      this->get_parameter("perception_result_topic").as_string();
  engagement_result_topic_ =
      this->get_parameter("engagement_result_topic").as_string();
  interaction_result_topic_ =
      this->get_parameter("interaction_result_topic").as_string();
  interaction_struct_.update(
      this->get_parameter("attention_yaw_deg").as_double(),
      this->get_parameter("attention_pitch_deg").as_double(),
      this->get_parameter("attention_max_distance_m").as_double(),
      this->get_parameter("attention_min_distance_m").as_double(),
      this->get_parameter("talking_yaw_deg").as_double(),
      this->get_parameter("talking_pitch_deg").as_double(),
      this->get_parameter("talking_max_distance_m").as_double(),
      this->get_parameter("talking_min_distance_m").as_double());
}

void PerceptionRosComponent::initImplementation() {
  // Initialize subscribers, publishers, and other implementation details here

  // Load parameters from the YAML file into the perception pipeline
  try {

    YAML::Node config = YAML::LoadFile(pipeline_config_path_);

    perception_pipeline_ptr_ = std::make_unique<PerceptionPipeline>(config);

  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(),
                 "[PerceptionPipelines] Failed to load parameters from %s: %s",
                 pipeline_config_path_.c_str(), e.what());
  }

  color_bbox_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      color_bbox_topic_, rclcpp::QoS(10).reliable());

  perception_result_pub_ =
      this->create_publisher<trt_infer_msgs::msg::PerceptionResult>(
          perception_result_topic_, rclcpp::QoS(10).reliable());

  engagement_result_pub_ =
      this->create_publisher<trt_infer_msgs::msg::ScenePerceptionResult>(
          engagement_result_topic_, rclcpp::QoS(10).reliable());

  interaction_result_pub_ =
      this->create_publisher<trt_infer_msgs::msg::InteractionResult>(
          interaction_result_topic_, rclcpp::QoS(10).reliable());

  // rclcpp::QoS image_qos(
  //     rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default),
  //     rmw_qos_profile_default);
  // image_qos.keep_last(1);
  // const rmw_qos_profile_t qos_profile = image_qos.get_rmw_qos_profile();

  // SensorDataQoS
  // 历史：KeepLast(5)、可靠性：BestEffort、耐久性：Volatile
  // 高频传感器数据（激光雷达、IMU、摄像头等）
  const rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;

  // default QoS profile for custom use
  // 历史：KeepLast(10)、可靠性：Reliable、耐久性：Volatile
  // const rmw_qos_profile_t qos_profile = rmw_qos_profile_default;

  if (use_compressed_images_) {
    color_compressed_sub_.subscribe(this, color_compressed_topic_, qos_profile);
    depth_compressed_sub_.subscribe(this, depth_compressed_topic_, qos_profile);
    if (hard_sync_) {
      compressed_sync_exact_ = std::make_unique<
          message_filters::Synchronizer<CompressedExactSyncPolicy>>(
          CompressedExactSyncPolicy(sync_queue_size_), color_compressed_sub_,
          depth_compressed_sub_);
      compressed_sync_exact_->registerCallback(
          std::bind(&PerceptionRosComponent::onSyncedCompressedColorDepth, this,
                    std::placeholders::_1, std::placeholders::_2));
    } else {
      compressed_sync_approx_ = std::make_unique<
          message_filters::Synchronizer<CompressedApproximateSyncPolicy>>(
          CompressedApproximateSyncPolicy(sync_queue_size_),
          color_compressed_sub_, depth_compressed_sub_);
      compressed_sync_approx_->registerCallback(
          std::bind(&PerceptionRosComponent::onSyncedCompressedColorDepth, this,
                    std::placeholders::_1, std::placeholders::_2));
    }
  } else {
    color_image_sub_.subscribe(this, color_image_topic_, qos_profile);
    depth_image_sub_.subscribe(this, depth_image_topic_, qos_profile);
    if (hard_sync_) {
      sync_exact_ =
          std::make_unique<message_filters::Synchronizer<ExactSyncPolicy>>(
              ExactSyncPolicy(sync_queue_size_), color_image_sub_,
              depth_image_sub_);
      sync_exact_->registerCallback(
          std::bind(&PerceptionRosComponent::onSyncedColorDepth, this,
                    std::placeholders::_1, std::placeholders::_2));
    } else {
      sync_approx_ = std::make_unique<
          message_filters::Synchronizer<ApproximateSyncPolicy>>(
          ApproximateSyncPolicy(sync_queue_size_), color_image_sub_,
          depth_image_sub_);
      sync_approx_->registerCallback(
          std::bind(&PerceptionRosComponent::onSyncedColorDepth, this,
                    std::placeholders::_1, std::placeholders::_2));
    }
  }

  // 设置定时器以处理最新的RGB和深度图像
  const auto processing_period =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / processing_rate_hz_));
  processing_timer_ = create_wall_timer(
      processing_period,
      std::bind(&PerceptionRosComponent::processLatestColorDepth, this));

  RCLCPP_INFO(this->get_logger(),
              "[PerceptionRosComponent] initialized with %s RGB-D topics.",
              use_compressed_images_ ? "compressed" : "raw");
}

void PerceptionRosComponent::onSyncedColorDepth(
    const Image::ConstSharedPtr &color_msg,
    const Image::ConstSharedPtr &depth_msg) {

  std::lock_guard<std::mutex> lock(latest_frames_mutex_);
  latest_color_msg_ = color_msg;
  latest_depth_msg_ = depth_msg;
}

void PerceptionRosComponent::onSyncedCompressedColorDepth(
    const CompressedImage::ConstSharedPtr &color_msg,
    const CompressedImage::ConstSharedPtr &depth_msg) {
  std::lock_guard<std::mutex> lock(latest_frames_mutex_);
  latest_compressed_color_msg_ = color_msg;
  latest_compressed_depth_msg_ = depth_msg;
}

void PerceptionRosComponent::processLatestColorDepth() {
  CompressedImage::ConstSharedPtr compressed_color_msg;
  CompressedImage::ConstSharedPtr compressed_depth_msg;
  Image::ConstSharedPtr color_msg;
  Image::ConstSharedPtr depth_msg;
  {
    std::lock_guard<std::mutex> lock(latest_frames_mutex_);
    if (use_compressed_images_) {
      compressed_color_msg = latest_compressed_color_msg_;
      compressed_depth_msg = latest_compressed_depth_msg_;
      latest_compressed_color_msg_.reset();
      latest_compressed_depth_msg_.reset();
    } else {
      color_msg = latest_color_msg_;
      depth_msg = latest_depth_msg_;
      latest_color_msg_.reset();
      latest_depth_msg_.reset();
    }
  }

  if (use_compressed_images_ && compressed_color_msg && compressed_depth_msg) {
    processCompressedColorDepth(compressed_color_msg, compressed_depth_msg);
  } else if (!use_compressed_images_ && color_msg && depth_msg) {
    processColorDepth(color_msg, depth_msg);
  }
}

void PerceptionRosComponent::processCompressedColorDepth(
    const CompressedImage::ConstSharedPtr &color_msg,
    const CompressedImage::ConstSharedPtr &depth_msg) {
  if (!color_msg || !depth_msg) {
    return;
  }

  cv::Mat color_image;
  cv::Mat depth_meters;
  if (!decodeCompressedColor(*color_msg, color_image)) {
    RCLCPP_WARN(this->get_logger(), "Failed to decode compressed color image.");
    return;
  }
  if (!decodeCompressedDepthToFloatMeters(*depth_msg, depth_meters)) {
    RCLCPP_WARN(this->get_logger(), "Failed to decode compressed depth image.");
    return;
  }

  processDecodedColorDepth(color_msg->header, std::move(color_image),
                           depth_meters);
}

void PerceptionRosComponent::processColorDepth(
    const Image::ConstSharedPtr &color_msg,
    const Image::ConstSharedPtr &depth_msg) {

  // 检查消息是否为空
  if (!color_msg || !depth_msg) {
    RCLCPP_WARN(this->get_logger(), "Received null color or depth image.");
    return;
  }

  cv::Mat color_image_mat;
  cv_bridge::CvImageConstPtr color_bridge;

  // 检查图像消息的编码是否为JPEG格式
  const bool is_jpeg =
      isJpegInImageMsg(color_msg->encoding) ||
      (color_msg->data.size() >= 2 && color_msg->data[0] == 0xff &&
       color_msg->data[1] == 0xd8);
  if (is_jpeg) {
    color_image_mat =
        cv::imdecode(cv::Mat(1, color_msg->data.size(), CV_8UC1,
                             const_cast<uint8_t *>(color_msg->data.data())),
                     cv::IMREAD_COLOR);
  } else {
    color_bridge =
        cv_bridge::toCvShare(color_msg, sensor_msgs::image_encodings::BGR8);
    color_image_mat = color_bridge->image;
  }

  cv::Mat depth_meters;
  if (!decodeToFloatMeters(depth_msg, depth_meters)) {
    RCLCPP_WARN(this->get_logger(),
                "Failed to decode depth image to float meters.");
    return;
  }

  processDecodedColorDepth(color_msg->header, std::move(color_image_mat),
                           depth_meters);
}

void PerceptionRosComponent::processDecodedColorDepth(
    const std_msgs::msg::Header &header, cv::Mat color_image,
    const cv::Mat &depth_meters) {
  ensureBgrU8C3(color_image);
  if (color_image.empty()) {
    RCLCPP_WARN(this->get_logger(),
                "Failed to convert color image to BGR8 format.");
    return;
  }
  if (depth_meters.empty() || depth_meters.type() != CV_32FC1) {
    RCLCPP_WARN(this->get_logger(), "Depth image must use CV_32FC1 meters.");
    return;
  }

  trt_infer_msgs::msg::PerceptionResult perception_result;
  perception_result.header = header;
  perception_result.image_width = static_cast<uint32_t>(color_image.cols);
  perception_result.image_height = static_cast<uint32_t>(color_image.rows);

  perception_pipeline_ptr_->process(color_image, depth_meters,
                                    perception_result);

  // 打印感知结果到控制台
  if (perception_result.persons.size() > 0) {
    printPerceptionResult(perception_result);
  }

  // 发布感知结果
  if (perception_result_pub_->get_subscription_count() > 0) {

    perception_result_pub_->publish(perception_result);
  }

  cv::Mat color_image_with_bbox = color_image.clone();

  trt_infer_msgs::msg::InteractionResult interaction_result_msg;
  interaction_result_msg.header = header;
  interaction_result_msg.closest_attention_distance = -1.0F;
  interaction_result_msg.closest_talking_distance = -1.0F;

  for (const auto &person : perception_result.persons) {

    drawPerceptionResultOnImage(color_image_with_bbox, person);

    updateInteractionResult(interaction_result_msg, person);
  }

  sensor_msgs::msg::Image::SharedPtr color_bbox_msg =
      cv_bridge::CvImage(header, "bgr8", color_image_with_bbox).toImageMsg();
  if (color_bbox_pub_->get_subscription_count() > 0) {
    color_bbox_pub_->publish(*color_bbox_msg);
  }

  if (interaction_result_pub_->get_subscription_count() > 0) {
    interaction_result_pub_->publish(interaction_result_msg);
  }

  if (engagement_result_pub_->get_subscription_count() > 0) {
    publishEngagementResult(perception_result, interaction_result_msg);
  }
}

void PerceptionRosComponent::drawPerceptionResultOnImage(
    cv::Mat &image, const trt_infer_msgs::msg::PersonMeta &person) {

  // 新增带感知信息的彩色图像发布
  // 图像格式: sensor_msgs::msg::Image
  // 遍历 perception_result.persons
  // 1. 绘制每个人体的bbox
  // 左上角坐标 (x0, y0)，右下角坐标 (x1, y1)
  // x0 = perception_result.persons[i].body_detection.body_bbox.x
  // y0 = perception_result.persons[i].body_detection.body_bbox.y
  // x1 = x0 + perception_result.persons[i].body_detection.body_bbox.w
  // y1 = y0 + perception_result.persons[i].body_detection.body_bbox.h
  // 2. 绘制每个人的人脸的bbox
  // 左上角坐标 (fx0, fy0)，右下角坐标 (fx1, fy1)
  // fx0 = perception_result.persons[i].face_detection.face_bbox.x
  // fy0 = perception_result.persons[i].face_detection.face_bbox.y
  // fx1 = fx0 + perception_result.persons[i].face_detection.face_bbox.w
  // fy1 = fy0 + perception_result.persons[i].face_detection.face_bbox.h
  // 3. 绘制 track_id
  // 绘制在人体bbox的左上角，文本内容为 perception_result.persons[i].track_id
  // 4. 绘制 基于 yaw, pitch, roll 的 box
  // yaw = perception_result.persons[i].head_pose.yaw
  // pitch = perception_result.persons[i].head_pose.pitch
  // roll = perception_result.persons[i].head_pose.roll

  // 1. 绘制人体bbox
  const auto &body_bbox = person.body_detection.body_bbox;
  cv::rectangle(image, cv::Point(body_bbox.x, body_bbox.y),
                cv::Point(body_bbox.x + body_bbox.w, body_bbox.y + body_bbox.h),
                cv::Scalar(0, 255, 0), 4);

  // 2. 绘制人脸bbox
  const auto &face_bbox = person.face_detection.face_bbox;
  if (face_bbox.w > 0 && face_bbox.h > 0) {
    cv::rectangle(
        image, cv::Point(face_bbox.x, face_bbox.y),
        cv::Point(face_bbox.x + face_bbox.w, face_bbox.y + face_bbox.h),
        cv::Scalar(0, 0, 255), 4);
  }

  // 3. 绘制 track_id
  const auto &track_id = person.track_id;
  cv::putText(image, std::to_string(track_id),
              cv::Point(body_bbox.x, body_bbox.y - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 4);

  // 4. 绘制基于 yaw, pitch, roll 的 box
  const auto &head_pose = person.head_pose;
  if (face_bbox.w > 0 && face_bbox.h > 0 && std::isfinite(head_pose.yaw) &&
      std::isfinite(head_pose.pitch) && std::isfinite(head_pose.roll)) {
    drawHeadPoseBox(
        image, cv::Rect2f(face_bbox.x, face_bbox.y, face_bbox.w, face_bbox.h),
        head_pose.yaw, head_pose.pitch, head_pose.roll);
  }
}

void PerceptionRosComponent::updateInteractionResult(
    trt_infer_msgs::msg::InteractionResult &interaction_result,
    const trt_infer_msgs::msg::PersonMeta &person) {

  uint8_t current_status = interaction_struct_.getInteractionStatus(
      person.head_pose.yaw, person.head_pose.pitch,
      person.body_detection.body_distance);

  // 如果当前的status大于interaction_result.best_status，则更新interaction_result.best_status为当前的status
  if (current_status > interaction_result.best_status) {
    interaction_result.best_status = current_status;
  }

  if (current_status ==
      trt_infer_msgs::msg::InteractionResult::INVALID_STATUS) {
    interaction_result.invalid_status_num++;
  }

  if (current_status ==
      trt_infer_msgs::msg::InteractionResult::ATTENTION_STATUS) {
    interaction_result.attention_status_num++;

    // 如果当前的closest_attention_distance小于0.0F，
    // 或者person的body_distance小于当前的closest_attention_distance，则更新closest_attention_distance为person的body_distance
    if (interaction_result.closest_attention_distance < 0.0F ||
        person.body_detection.body_distance <
            interaction_result.closest_attention_distance) {
      interaction_result.closest_attention_distance =
          person.body_detection.body_distance;
    }
  }

  if (current_status ==
      trt_infer_msgs::msg::InteractionResult::TALKING_STATUS) {

    interaction_result.talking_status_num++;

    // 如果当前的closest_talking_distance小于0.0F，
    // 或者person的body_distance小于当前的closest_talking_distance，则更新closest_talking_distance为person的body_distance
    if (interaction_result.closest_talking_distance < 0.0F ||
        person.body_detection.body_distance <
            interaction_result.closest_talking_distance) {
      interaction_result.closest_talking_distance =
          person.body_detection.body_distance;
    }
  }
}

void PerceptionRosComponent::publishEngagementResult(
    const trt_infer_msgs::msg::PerceptionResult &perception_result,
    const trt_infer_msgs::msg::InteractionResult &interaction_result) {
  trt_infer_msgs::msg::ScenePerceptionResult engagement_result;
  engagement_result.header = perception_result.header;
  engagement_result.image_width = perception_result.image_width;
  engagement_result.image_height = perception_result.image_height;
  engagement_result.body_pipeline_ms = perception_result.body_detection_ms;
  engagement_result.best_engagement = interaction_result.best_status;
  engagement_result.engaged_count = interaction_result.talking_status_num;
  engagement_result.attention_count = interaction_result.attention_status_num;
  engagement_result.closest_engaged_distance =
      interaction_result.closest_talking_distance;
  engagement_result.closest_attention_distance =
      interaction_result.closest_attention_distance;
  engagement_result.persons.reserve(perception_result.persons.size());

  for (const auto &person : perception_result.persons) {
    trt_infer_msgs::msg::PersonPerception legacy_person;
    const auto &body_bbox = person.body_detection.body_bbox;
    const auto &face_bbox = person.face_detection.face_bbox;

    legacy_person.track_id = person.track_id;
    legacy_person.body_x = body_bbox.x;
    legacy_person.body_y = body_bbox.y;
    legacy_person.body_w = body_bbox.w;
    legacy_person.body_h = body_bbox.h;
    legacy_person.body_conf = person.body_detection.body_confidence;
    legacy_person.distance = person.body_detection.body_distance;
    legacy_person.has_face = face_bbox.w > 0 && face_bbox.h > 0;
    legacy_person.face_x = face_bbox.x;
    legacy_person.face_y = face_bbox.y;
    legacy_person.face_w = face_bbox.w;
    legacy_person.face_h = face_bbox.h;
    legacy_person.face_conf = person.face_detection.face_confidence;
    legacy_person.yaw = person.head_pose.yaw;
    legacy_person.pitch = person.head_pose.pitch;
    legacy_person.roll = person.head_pose.roll;
    legacy_person.engagement = interaction_struct_.getInteractionStatus(
        legacy_person.yaw, legacy_person.pitch, legacy_person.distance);
    legacy_person.person_uuid = person.face_recog.person_uuid;
    legacy_person.person_name = person.face_recog.person_name;
    legacy_person.face_recog_conf = person.face_recog.face_recog_conf;
    legacy_person.face_embedding = person.face_recog.face_embedding;
    engagement_result.persons.push_back(std::move(legacy_person));
  }

  engagement_result_pub_->publish(engagement_result);
}

void PerceptionRosComponent::printPerceptionResult(
    const trt_infer_msgs::msg::PerceptionResult &result) {
  RCLCPP_INFO(this->get_logger(),
              "Perception Result: \n%zu persons detected, "
              "\nbody_detection_ms: %.4f, \nface_detection_ms: %.4f, "
              "\nhead_pose_ms: %.4f, \nface_recog_ms: %.4f",
              result.persons.size(), result.body_detection_ms,
              result.face_detection_ms, result.head_pose_ms,
              result.face_recog_ms);
  for (const auto &person : result.persons) {
    RCLCPP_INFO(this->get_logger(),
                "Person track_id: %d, \nbody_distance: %.2f m, "
                "\nhead_pose (yaw: %.2f, pitch: %.2f, "
                "roll: %.2f)",
                person.track_id, person.body_detection.body_distance,
                person.head_pose.yaw, person.head_pose.pitch,
                person.head_pose.roll);
  }
}

} // namespace perception_ros_component

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(
    perception_ros_component::PerceptionRosComponent)