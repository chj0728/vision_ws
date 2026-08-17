# include "perception_ros_component.hpp"

namespace perception_ros_component {

PerceptionRosComponent::PerceptionRosComponent(const rclcpp::NodeOptions& options)
    : Node("perception_ros_component", options) {

  loadParameters(); // Load parameters from YAML configuration file
  
  // Initialize the perception pipeline

  // perception_pipeline_ptr_ = std::make_unique<PerceptionPipeline>();
  // perception_pipeline_ptr_->initialize();
  // RCLCPP_INFO(this->get_logger(), "Perception pipeline is initialized successfully.");

  // // Declare and get parameters
  // this->declare_parameter<bool>("hard_sync", false);
  // this->declare_parameter<int>("sync_queue_size", 10);

  // this->declare_parameter<std::string>("color_image_topic", "/camera/color/image_raw");
  // this->declare_parameter<std::string>("depth_image_topic", "/camera/depth/image_raw");
  // this->declare_parameter<std::string>("perception_result_topic", "/perception/result");

  // hard_sync_ = this->get_parameter("hard_sync").as_bool();
  // sync_queue_size_ = this->get_parameter("sync_queue_size").as_int();
  // color_image_topic_ = this->get_parameter("color_image_topic").as_string();
  // depth_image_topic_ = this->get_parameter("depth_image_topic").as_string();
  // perception_result_topic_ = this->get_parameter("perception_result_topic").as_string();

  // Create publisher for perception results
  perception_result_pub_ = create_publisher<trt_infer_msgs::msg::PerceptionResult>(perception_result_topic_, rclcpp::QoS(2).reliable());

  // Set up subscribers for RGB and depth images
  rclcpp::QoS image_qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default), rmw_qos_profile_default);
  image_qos.keep_last(6);
  const rmw_qos_profile_t qos_profile = image_qos.get_rmw_qos_profile();
  color_image_sub_.subscribe(this, color_image_topic_, qos_profile);
  depth_image_sub_.subscribe(this, depth_image_topic_, qos_profile);

  // Set up synchronizer for RGB and depth images
  if (hard_sync_) {
    sync_exact_ = std::make_unique<message_filters::Synchronizer<ExactSyncPolicy>>(
      ExactSyncPolicy(sync_queue_size_), color_image_sub_, depth_image_sub_);
    sync_exact_->registerCallback(
      std::bind(&PerceptionRosComponent::onSyncedColorDepth, this, std::placeholders::_1, std::placeholders::_2));
  } else {
    sync_approx_ = std::make_unique<message_filters::Synchronizer<ApproximateSyncPolicy>>(
      ApproximateSyncPolicy(sync_queue_size_), color_image_sub_, depth_image_sub_);
    sync_approx_->registerCallback(
      std::bind(&PerceptionRosComponent::onSyncedColorDepth, this, std::placeholders::_1, std::placeholders::_2));
  }
  
  RCLCPP_INFO(this->get_logger(), "PerceptionRosComponent is initialized successfully.");
}

PerceptionRosComponent::~PerceptionRosComponent() = default;

void PerceptionRosComponent::loadParameters() {
  // Load parameters from a YAML configuration file
  this->declare_parameter<std::string>("config_path", "config/perception_config.yaml");
  config_path_ = this->get_parameter("config_path").as_string();

  try {
    YAML::Node config = YAML::LoadFile(config_path_);
    perception_pipeline_ptr_ = std::make_unique<PerceptionPipeline>(config);
    RCLCPP_INFO(this->get_logger(), "Loaded parameters from %s", config_path_.c_str());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load parameters from %s: %s", config_path_.c_str(), e.what());
  }

  // Declare and get parameters
  this->declare_parameter<bool>("hard_sync", false);
  this->declare_parameter<int>("sync_queue_size", 10);

  this->declare_parameter<std::string>("color_image_topic", "/camera/color/image_raw");
  this->declare_parameter<std::string>("depth_image_topic", "/camera/depth/image_raw");
  this->declare_parameter<std::string>("perception_result_topic", "/perception/result");

  hard_sync_ = this->get_parameter("hard_sync").as_bool();
  sync_queue_size_ = this->get_parameter("sync_queue_size").as_int();
  color_image_topic_ = this->get_parameter("color_image_topic").as_string();
  depth_image_topic_ = this->get_parameter("depth_image_topic").as_string();
  perception_result_topic_ = this->get_parameter("perception_result_topic").as_string();
}

bool PerceptionRosComponent::decodeToFloatMeters(const Image::ConstSharedPtr& depth_msg, cv::Mat& depth_meters) {
	if (!depth_msg) return false;

	const std::string encoding = toLower(depth_msg->encoding);
	if (encoding == sensor_msgs::image_encodings::TYPE_16UC1 || encoding == "16uc1") {
		const auto depth = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
		if (depth_f32_buf_.rows != static_cast<int>(depth_msg->height) ||
			depth_f32_buf_.cols != static_cast<int>(depth_msg->width)) {
			depth_f32_buf_.create(depth_msg->height, depth_msg->width, CV_32F);
		}
		depth->image.convertTo(depth_f32_buf_, CV_32F, static_cast<double>(depth_scale_to_meters_));
		depth_meters = depth_f32_buf_;
		return true;
	}
	if (encoding == sensor_msgs::image_encodings::TYPE_32FC1 || encoding == "32fc1") {
		depth_meters = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
		return true;
	}
	return false;
}

void PerceptionRosComponent::onSyncedColorDepth(const Image::ConstSharedPtr& color_msg,
                        const Image::ConstSharedPtr& depth_msg) {

  // const auto start_time = std::chrono::steady_clock::now();

  // 检查消息是否为空
  if (!color_msg || !depth_msg) {
    RCLCPP_WARN(this->get_logger(), "Received null color or depth image.");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Received synchronized color and depth images.");

  cv::Mat color_image_mat;
  
  // 检查图像消息的编码是否为JPEG格式
  const bool is_jpeg = isJpegInImageMsg(color_msg->encoding) ||
             (color_msg->data.size() >= 2 && color_msg->data[0] == 0xff && color_msg->data[1] == 0xd8);
  if (is_jpeg) {
    color_image_mat = cv::imdecode(cv::Mat(1, color_msg->data.size(), CV_8UC1,
                   const_cast<uint8_t*>(color_msg->data.data())),
               cv::IMREAD_COLOR);
  } else {
    color_image_mat = cv_bridge::toCvShare(color_msg, sensor_msgs::image_encodings::BGR8)->image;
  }

  // 确保图像为BGR8格式
  ensureBgrU8C3(color_image_mat);
  if (color_image_mat.empty()) {
    RCLCPP_WARN(this->get_logger(), "Failed to convert color image to BGR8 format.");
    return;
  }

  // 将深度图像解码为浮点米表示
  cv::Mat depth_image_mat;
  if (!decodeToFloatMeters(depth_msg, depth_image_mat)){
    RCLCPP_WARN(this->get_logger(), "Failed to decode depth image to float meters.");
    return;
  }

  // 处理图像数据并发布感知结果
  trt_infer_msgs::msg::PerceptionResult perception_result;
  perception_result.header = color_msg->header;
  perception_result.image_width = static_cast<uint32_t>(color_image_mat.cols);
  perception_result.image_height = static_cast<uint32_t>(color_image_mat.rows);

  perception_pipeline_ptr_->process(color_image_mat, depth_image_mat, perception_result);

  // 发布感知结果，如果有订阅者
  if (perception_result_pub_->get_subscription_count() > 0) {
    perception_result_pub_->publish(perception_result);
  }
}

} // namespace perception_ros_component

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(perception_ros_component::PerceptionRosComponent)