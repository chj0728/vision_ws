#include "pipeline/sixdrepnet_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

constexpr float kInvalidHeadPoseDeg = 999.0f;

/** @brief 以人脸框中心为基准向四周扩展，并裁剪到图像边界内。 */
cv::Rect expandFaceRect(const cv::Rect_<float> &face_rect, float expand_ratio,
                        int image_width, int image_height) {
  if (face_rect.width <= 0.0f || face_rect.height <= 0.0f || image_width <= 0 ||
      image_height <= 0) {
    return {};
  }

  const float center_x = face_rect.x + face_rect.width * 0.5f;
  const float center_y = face_rect.y + face_rect.height * 0.5f;
  const float expanded_width = face_rect.width * (1.0f + 2.0f * expand_ratio);
  const float expanded_height = face_rect.height * (1.0f + 2.0f * expand_ratio);

  const cv::Rect expanded(
      static_cast<int>(std::floor(center_x - expanded_width * 0.5f)),
      static_cast<int>(std::floor(center_y - expanded_height * 0.5f)),
      static_cast<int>(std::ceil(expanded_width)),
      static_cast<int>(std::ceil(expanded_height)));
  return expanded & cv::Rect(0, 0, image_width, image_height);
}

/** @brief 将头姿消息恢复为未执行或推理失败状态。 */
void clearHeadPose(trt_infer_msgs::msg::HeadPose &head_pose) {
  head_pose.yaw = kInvalidHeadPoseDeg;
  head_pose.pitch = kInvalidHeadPoseDeg;
  head_pose.roll = 0.0f;
}

} // namespace

SixDRepNetPipeline::SixDRepNetPipeline(const YAML::Node &config) {
  loadParameters(config);
  initialize();
}

SixDRepNetPipeline::~SixDRepNetPipeline() = default;

void SixDRepNetPipeline::loadParameters(const YAML::Node &config) {
  const YAML::Node sixdrepnet_config = config["sixdrepnet_pipeline"];

  enabled_ = sixdrepnet_config["enable"].as<bool>(false);
  sixdrepnet_engine_name_ =
      sixdrepnet_config["sixdrepnet_engine_name"].as<std::string>(
          "SixDRepNet.engine");

  const std::string configured_path =
      sixdrepnet_config["sixdrepnet_engine_path"].as<std::string>("");
  if (!configured_path.empty()) {
    const std::filesystem::path path(configured_path);
    sixdrepnet_engine_path_ =
        (path.is_absolute() ? path
                            : std::filesystem::path(TRT_WORKSPACE_ROOT) / path)
            .string();
  } else {
    sixdrepnet_engine_path_ =
        (std::filesystem::path(TRT_WORKSPACE_ROOT) / "models" / "sixdrepnet" /
         sixdrepnet_engine_name_)
            .string();
  }

  min_face_px_ = std::max(1, sixdrepnet_config["min_face_px"].as<int>(36));
  face_expand_ratio_ = std::clamp(
      sixdrepnet_config["face_expand_ratio"].as<float>(0.12f), 0.0f, 0.5f);
}

void SixDRepNetPipeline::initialize() {
  if (!enabled_) {
    return;
  }

  if (!std::filesystem::exists(sixdrepnet_engine_path_)) {
    throw std::runtime_error("SixDRepNet engine not found: " +
                             sixdrepnet_engine_path_);
  }

  sixdrepnet_engine_ptr_ =
      std::make_unique<SixDRepNet_TRT>(sixdrepnet_engine_path_);
}

void SixDRepNetPipeline::process(
    const cv::Mat &rgb, const PerceptionFrameContext &frame_context,
    trt_infer_msgs::msg::PerceptionResult &perception_result) {
  for (auto &person : perception_result.persons) {
    clearHeadPose(person.head_pose);
  }

  if (!enabled_ || !sixdrepnet_engine_ptr_ || rgb.empty()) {
    return;
  }

  const auto start_time = std::chrono::high_resolution_clock::now();
  const std::size_t person_count =
      std::min(frame_context.persons.size(), perception_result.persons.size());

  for (std::size_t index = 0; index < person_count; ++index) {
    const auto &person_context = frame_context.persons[index];
    if (!person_context.has_face) {
      continue;
    }

    const cv::Rect face_roi = expandFaceRect(
        person_context.face.rect, face_expand_ratio_, rgb.cols, rgb.rows);
    if (face_roi.width < min_face_px_ || face_roi.height < min_face_px_) {
      continue;
    }

    try {
      const ::HeadPose pose = sixdrepnet_engine_ptr_->predict(rgb(face_roi));
      if (!std::isfinite(pose.yaw) || !std::isfinite(pose.pitch) ||
          !std::isfinite(pose.roll)) {
        continue;
      }

      auto &head_pose = perception_result.persons[index].head_pose;
      head_pose.yaw = pose.yaw;
      head_pose.pitch = pose.pitch;
      head_pose.roll = pose.roll;
    } catch (const std::exception &exception) {
      std::cerr << "[SixDRepNetPipeline] Prediction failed: "
                << exception.what() << std::endl;
    }
  }

  const std::chrono::duration<float, std::milli> duration =
      std::chrono::high_resolution_clock::now() - start_time;
  std::cout << "[SixDRepNetPipeline] Processing time: " << duration.count()
            << " ms" << std::endl;
}
