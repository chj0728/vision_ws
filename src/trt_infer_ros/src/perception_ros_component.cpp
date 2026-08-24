#include "perception_ros_component.hpp"

#include <array>
#include <cmath>
#include <cstdint>

namespace perception_ros_component {

namespace {

void drawHeadPoseBox(cv::Mat &image, const cv::Rect2f &face_bbox, float yaw,
                     float pitch, float roll) {
  constexpr float kDegreesToRadians = CV_PI / 180.0F;
  const float yaw_rad = yaw * kDegreesToRadians;
  const float pitch_rad = pitch * kDegreesToRadians;
  const float roll_rad = roll * kDegreesToRadians;
  const float side = std::min(face_bbox.width, face_bbox.height) * 0.7F;
  const float focal_length = side * 3.0F;
  const cv::Point2f center(face_bbox.x + face_bbox.width * 0.5F,
                           face_bbox.y + face_bbox.height * 0.5F);

  const float cos_yaw = std::cos(yaw_rad);
  const float sin_yaw = std::sin(yaw_rad);
  const float cos_pitch = std::cos(pitch_rad);
  const float sin_pitch = std::sin(pitch_rad);
  const float cos_roll = std::cos(roll_rad);
  const float sin_roll = std::sin(roll_rad);
  const cv::Matx33f rotation(
      cos_roll * cos_yaw, cos_roll * sin_yaw * sin_pitch - sin_roll * cos_pitch,
      cos_roll * sin_yaw * cos_pitch + sin_roll * sin_pitch, sin_roll * cos_yaw,
      sin_roll * sin_yaw * sin_pitch + cos_roll * cos_pitch,
      sin_roll * sin_yaw * cos_pitch - cos_roll * sin_pitch, -sin_yaw,
      cos_yaw * sin_pitch, cos_yaw * cos_pitch);

  const float half_side = side * 0.5F;
  const auto project = [&](const cv::Vec3f &point) {
    const cv::Vec3f rotated = rotation * point;
    const float scale = focal_length / (focal_length + rotated[2]);
    return cv::Point(cvRound(center.x + rotated[0] * scale),
                     cvRound(center.y + rotated[1] * scale));
  };
  const std::array<cv::Vec3f, 8> vertices = {
      cv::Vec3f{-half_side, -half_side, -half_side},
      cv::Vec3f{half_side, -half_side, -half_side},
      cv::Vec3f{half_side, half_side, -half_side},
      cv::Vec3f{-half_side, half_side, -half_side},
      cv::Vec3f{-half_side, -half_side, half_side},
      cv::Vec3f{half_side, -half_side, half_side},
      cv::Vec3f{half_side, half_side, half_side},
      cv::Vec3f{-half_side, half_side, half_side}};
  std::array<cv::Point, 8> projected_vertices;
  for (std::size_t index = 0; index < vertices.size(); ++index) {
    projected_vertices[index] = project(vertices[index]);
  }

  constexpr std::array<std::array<int, 2>, 12> edges = {{{{0, 1}},
                                                         {{1, 2}},
                                                         {{2, 3}},
                                                         {{3, 0}},
                                                         {{4, 5}},
                                                         {{5, 6}},
                                                         {{6, 7}},
                                                         {{7, 4}},
                                                         {{0, 4}},
                                                         {{1, 5}},
                                                         {{2, 6}},
                                                         {{3, 7}}}};
  for (const auto &edge : edges) {
    cv::line(image, projected_vertices[edge[0]], projected_vertices[edge[1]],
             cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  }

  const cv::Point projected_origin = project(cv::Vec3f(0.0F, 0.0F, 0.0F));
  const float axis_length = side * 0.9F;
  const std::array<cv::Vec3f, 3> axes = {cv::Vec3f(axis_length, 0.0F, 0.0F),
                                         cv::Vec3f(0.0F, axis_length, 0.0F),
                                         cv::Vec3f(0.0F, 0.0F, axis_length)};
  // OpenCV uses BGR: X is red, Y is green, and Z is blue as in RViz TF.
  const std::array<cv::Scalar, 3> axis_colors = {
      cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)};
  for (std::size_t index = 0; index < axes.size(); ++index) {
    cv::arrowedLine(image, projected_origin, project(axes[index]),
                    axis_colors[index], 2, cv::LINE_AA, 0, 0.2);
  }
}

} // namespace

PerceptionRosComponent::PerceptionRosComponent(
    const rclcpp::NodeOptions &options)
    : Node("perception_ros_component", options) {

  // Load parameters from YAML configuration file
  loadParameters();

  // Set up subscribers for RGB and depth images
  rclcpp::QoS image_qos(
      rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default),
      rmw_qos_profile_default);
  image_qos.keep_last(1);
  const rmw_qos_profile_t qos_profile = image_qos.get_rmw_qos_profile();
  color_image_sub_.subscribe(this, color_image_topic_, qos_profile);
  depth_image_sub_.subscribe(this, depth_image_topic_, qos_profile);

  // Set up synchronizer for RGB and depth images
  if (hard_sync_) {
    sync_exact_ =
        std::make_unique<message_filters::Synchronizer<ExactSyncPolicy>>(
            ExactSyncPolicy(sync_queue_size_), color_image_sub_,
            depth_image_sub_);
    sync_exact_->registerCallback(
        std::bind(&PerceptionRosComponent::onSyncedColorDepth, this,
                  std::placeholders::_1, std::placeholders::_2));
  } else {
    sync_approx_ =
        std::make_unique<message_filters::Synchronizer<ApproximateSyncPolicy>>(
            ApproximateSyncPolicy(sync_queue_size_), color_image_sub_,
            depth_image_sub_);
    sync_approx_->registerCallback(
        std::bind(&PerceptionRosComponent::onSyncedColorDepth, this,
                  std::placeholders::_1, std::placeholders::_2));
  }

  // 设置定时器以处理最新的RGB和深度图像
  const auto processing_period =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / processing_rate_hz_));
  processing_timer_ = create_wall_timer(
      processing_period,
      std::bind(&PerceptionRosComponent::processLatestColorDepth, this));

  RCLCPP_INFO(this->get_logger(),
              "[PerceptionRosComponent] is initialized successfully.");
}

PerceptionRosComponent::~PerceptionRosComponent() = default;

void PerceptionRosComponent::loadParameters() {

  // Load parameters from a YAML configuration file
  this->declare_parameter<std::string>("pipeline_config_path", "pipeline.yaml");
  pipeline_config_path_ =
      this->get_parameter("pipeline_config_path").as_string();
  RCLCPP_INFO(this->get_logger(), "Loaded pipelines parameters from %s",
              pipeline_config_path_.c_str());

  // Load parameters from the YAML file into the perception pipeline
  try {

    YAML::Node config = YAML::LoadFile(pipeline_config_path_);

    perception_pipeline_ptr_ = std::make_unique<PerceptionPipeline>(config);

  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(),
                 "[PerceptionPipelines] Failed to load parameters from %s: %s",
                 pipeline_config_path_.c_str(), e.what());
  }

  // Declare and get parameters
  this->declare_parameter<bool>("hard_sync", false);
  this->declare_parameter<int>("sync_queue_size", 10);
  this->declare_parameter<double>("processing_rate_hz", 10.0);

  this->declare_parameter<std::string>("color_image_topic",
                                       "/camera/color/image_raw");
  this->declare_parameter<std::string>("depth_image_topic",
                                       "/camera/depth/image_raw");
  this->declare_parameter<std::string>("perception_result_topic",
                                       "/perception/result");
  this->declare_parameter<std::string>("color_bbox_topic",
                                       "/perception/color_bbox");

  hard_sync_ = this->get_parameter("hard_sync").as_bool();
  sync_queue_size_ = this->get_parameter("sync_queue_size").as_int();
  processing_rate_hz_ = this->get_parameter("processing_rate_hz").as_double();
  if (processing_rate_hz_ <= 0.0) {
    throw std::invalid_argument("processing_rate_hz must be greater than zero");
  }
  color_image_topic_ = this->get_parameter("color_image_topic").as_string();
  depth_image_topic_ = this->get_parameter("depth_image_topic").as_string();

  // Create publisher for perception results
  perception_result_topic_ =
      this->get_parameter("perception_result_topic").as_string();
  perception_result_pub_ =
      create_publisher<trt_infer_msgs::msg::PerceptionResult>(
          perception_result_topic_, rclcpp::QoS(10).reliable());

  // Create publisher for color image with bounding boxes
  color_bbox_topic_ = this->get_parameter("color_bbox_topic").as_string();
  color_bbox_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      color_bbox_topic_, rclcpp::QoS(10));

  // Load interaction status parameters
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

  interaction_result_topic_ =
      this->get_parameter("interaction_result_topic").as_string();
  interaction_result_pub_ =
      create_publisher<trt_infer_msgs::msg::InteractionResult>(
          interaction_result_topic_, rclcpp::QoS(10).reliable());
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

bool PerceptionRosComponent::decodeToFloatMeters(
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

void PerceptionRosComponent::onSyncedColorDepth(
    const Image::ConstSharedPtr &color_msg,
    const Image::ConstSharedPtr &depth_msg) {

  std::lock_guard<std::mutex> lock(latest_frames_mutex_);
  latest_color_msg_ = color_msg;
  latest_depth_msg_ = depth_msg;
}

void PerceptionRosComponent::processLatestColorDepth() {
  Image::ConstSharedPtr color_msg;
  Image::ConstSharedPtr depth_msg;
  {
    std::lock_guard<std::mutex> lock(latest_frames_mutex_);
    color_msg = latest_color_msg_;
    depth_msg = latest_depth_msg_;
    latest_color_msg_.reset();
    latest_depth_msg_.reset();
  }

  if (color_msg && depth_msg) {
    processColorDepth(color_msg, depth_msg);
  }
}

void PerceptionRosComponent::processColorDepth(
    const Image::ConstSharedPtr &color_msg,
    const Image::ConstSharedPtr &depth_msg) {

  // const auto start_time = std::chrono::steady_clock::now();

  // // Check if there are any subscribers for the perception result topic
  // if (perception_result_pub_->get_subscription_count() == 0) {
  //   RCLCPP_WARN(this->get_logger(),
  //               "No subscribers for topic [%s], skipping processing.",
  //               perception_result_topic_.c_str());
  //   return;
  // }

  // 检查消息是否为空
  if (!color_msg || !depth_msg) {
    RCLCPP_WARN(this->get_logger(), "Received null color or depth image.");
    return;
  }

  // RCLCPP_INFO(this->get_logger(), "Received synchronized color and depth
  // images.");

  cv::Mat color_image_mat;

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
    color_image_mat =
        cv_bridge::toCvShare(color_msg, sensor_msgs::image_encodings::BGR8)
            ->image;
  }

  // 确保图像为BGR8格式
  ensureBgrU8C3(color_image_mat);
  if (color_image_mat.empty()) {
    RCLCPP_WARN(this->get_logger(),
                "Failed to convert color image to BGR8 format.");
    return;
  }

  // 将深度图像解码为浮点米表示
  cv::Mat depth_image_mat;
  if (!decodeToFloatMeters(depth_msg, depth_image_mat)) {
    RCLCPP_WARN(this->get_logger(),
                "Failed to decode depth image to float meters.");
    return;
  }

  // 处理图像数据
  trt_infer_msgs::msg::PerceptionResult perception_result;
  perception_result.header = color_msg->header;
  perception_result.image_width = static_cast<uint32_t>(color_image_mat.cols);
  perception_result.image_height = static_cast<uint32_t>(color_image_mat.rows);

  perception_pipeline_ptr_->process(color_image_mat, depth_image_mat,
                                    perception_result);

  // 发布感知结果
  if (perception_result_pub_->get_subscription_count() > 0) {
    RCLCPP_INFO(this->get_logger(),
                "Publishing perception result with %zu persons detected.",
                perception_result.persons.size());
    perception_result_pub_->publish(perception_result);
  } else {
    RCLCPP_WARN(
        this->get_logger(),
        "No subscribers for perception result topic, skipping publish.");
  }

  cv::Mat color_image_with_bbox = color_image_mat.clone();

  trt_infer_msgs::msg::InteractionResult interaction_result_msg;
  interaction_result_msg.header = color_msg->header;
  interaction_result_msg.closest_attention_distance = -1.0F;
  interaction_result_msg.closest_talking_distance = -1.0F;

  for (const auto &person : perception_result.persons) {

    drawPerceptionResultOnImage(color_image_with_bbox, person);

    updateInteractionResult(interaction_result_msg, person);
  }

  sensor_msgs::msg::Image::SharedPtr color_bbox_msg =
      cv_bridge::CvImage(color_msg->header, "bgr8", color_image_with_bbox)
          .toImageMsg();
  if (color_bbox_pub_->get_subscription_count() > 0) {
    color_bbox_pub_->publish(*color_bbox_msg);
  }

  if (interaction_result_pub_->get_subscription_count() > 0) {
    interaction_result_pub_->publish(interaction_result_msg);
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
} // namespace perception_ros_component

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(
    perception_ros_component::PerceptionRosComponent)