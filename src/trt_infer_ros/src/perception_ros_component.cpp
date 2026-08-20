#include "perception_ros_component.hpp"

namespace perception_ros_component {

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

  // Check if there are any subscribers for the perception result topic
  if (perception_result_pub_->get_subscription_count() == 0) {
    RCLCPP_WARN(this->get_logger(),
                "No subscribers for topic [%s], skipping processing.",
                perception_result_topic_.c_str());
    return;
  }

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

  // 处理图像数据并发布感知结果
  trt_infer_msgs::msg::PerceptionResult perception_result;
  perception_result.header = color_msg->header;
  perception_result.image_width = static_cast<uint32_t>(color_image_mat.cols);
  perception_result.image_height = static_cast<uint32_t>(color_image_mat.rows);

  perception_pipeline_ptr_->process(color_image_mat, depth_image_mat,
                                    perception_result);
  perception_result_pub_->publish(perception_result);

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
  if (color_bbox_pub_->get_subscription_count() > 0) {
    cv::Mat color_image_with_bbox = color_image_mat.clone();
    for (const auto &person : perception_result.persons) {

      // 1. 绘制人体bbox
      const auto &body_bbox = person.body_detection.body_bbox;
      cv::rectangle(
          color_image_with_bbox, cv::Point(body_bbox.x, body_bbox.y),
          cv::Point(body_bbox.x + body_bbox.w, body_bbox.y + body_bbox.h),
          cv::Scalar(0, 255, 0), 4);

      // 2. 绘制人脸bbox
      const auto &face_bbox = person.face_detection.face_bbox;
      if (face_bbox.w > 0 && face_bbox.h > 0) {
        cv::rectangle(
            color_image_with_bbox, cv::Point(face_bbox.x, face_bbox.y),
            cv::Point(face_bbox.x + face_bbox.w, face_bbox.y + face_bbox.h),
            cv::Scalar(0, 0, 255), 4);
      }

      // 3. 绘制 track_id
      const auto &track_id = person.track_id;
      cv::putText(color_image_with_bbox, std::to_string(track_id),
                  cv::Point(body_bbox.x, body_bbox.y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 4);
    }
    sensor_msgs::msg::Image::SharedPtr color_bbox_msg =
        cv_bridge::CvImage(color_msg->header, "bgr8", color_image_with_bbox)
            .toImageMsg();
    color_bbox_pub_->publish(*color_bbox_msg);
  }
}

} // namespace perception_ros_component

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(
    perception_ros_component::PerceptionRosComponent)