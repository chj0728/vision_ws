#include "pipeline/scrfd_pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

} // namespace

SCRFDPipeline::SCRFDPipeline(const YAML::Node &config) {
  loadParameters(config);
  initialize();
}

SCRFDPipeline::~SCRFDPipeline() = default;

void SCRFDPipeline::loadParameters(const YAML::Node &config) {
  // 仅优化内部命名，已有 YAML 键名、默认值和范围限制保持不变。
  const YAML::Node scrfd_config = config["scrfd_pipeline"];

  enabled_ = scrfd_config["enable"].as<bool>(false);

  engine_filename_ = scrfd_config["scrfd_engine_name"].as<std::string>(
      "scrfd_2.5g_bnkps_shape640x640.trt");
  const std::string configured_path =
      scrfd_config["scrfd_engine_path"].as<std::string>("");
  if (!configured_path.empty()) {
    const std::filesystem::path path(configured_path);
    engine_path_ =
        (path.is_absolute() ? path
                            : std::filesystem::path(TRT_WORKSPACE_ROOT) / path)
            .string();
  } else {
    engine_path_ = (std::filesystem::path(TRT_WORKSPACE_ROOT) / "models" /
                    "scrfd" / engine_filename_)
                       .string();
  }

  preprocess_mode_ =
      toLower(scrfd_config["preprocess"].as<std::string>("insightface"));
  face_confidence_threshold_ =
      std::clamp(scrfd_config["prob_threshold"].as<float>(0.38f), 0.08f, 0.95f);
  face_nms_iou_threshold_ =
      std::clamp(scrfd_config["nms_threshold"].as<float>(0.45f), 0.15f, 0.95f);
  head_roi_height_ratio_ = std::clamp(
      scrfd_config["person_head_height_ratio"].as<double>(0.58), 0.18, 0.78);
  head_roi_side_padding_ratio_ = std::clamp(
      scrfd_config["person_head_width_pad_ratio"].as<double>(0.24), 0.0, 0.55);
  head_roi_top_expansion_ratio_ = std::clamp(
      scrfd_config["person_head_top_expand_ratio"].as<double>(0.14), 0.0, 0.5);
  min_head_roi_side_px_ =
      std::max(32, scrfd_config["face_roi_min_side"].as<int>(48));
  max_person_roi_attempts_ =
      std::max(1, scrfd_config["max_person_rois"].as<int>(8));
}

void SCRFDPipeline::initialize() {

  if (!enabled_) {
    return;
  }

  if (!std::filesystem::exists(engine_path_)) {
    throw std::runtime_error("SCRFD engine not found: " + engine_path_);
  }

  const bool use_insightface = preprocess_mode_ == "insightface";
  const bool use_namdvt =
      preprocess_mode_ == "namdvt" || preprocess_mode_ == "upstream";
  if (!use_insightface && !use_namdvt) {
    throw std::invalid_argument("Unsupported SCRFD preprocess mode: " +
                                preprocess_mode_);
  }

  face_detector_ = std::make_unique<SCRFD_TRT>(engine_path_);
  face_detector_->setPreprocess(use_insightface
                                    ? SCRFD_TRT::Preprocess::InsightFacePython
                                    : SCRFD_TRT::Preprocess::NamdvtUpstream);
}

cv::Rect SCRFDPipeline::buildHeadShoulderRoi(
    const trt_infer_msgs::msg::BoundingBox &body,
    const cv::Size &image_size) const {
  const cv::Rect image_bounds(0, 0, image_size.width, image_size.height);
  if (body.w <= 0 || body.h <= 0) {
    return {};
  }
  const cv::Rect body_rect(body.x, body.y, body.w, body.h);
  const cv::Rect clipped_body = body_rect & image_bounds;
  if (clipped_body.width < 4 || clipped_body.height < 4) {
    return {};
  }

  // 使用裁后人体框计算比例，避免越界部分放大头肩区域。
  const double center_x =
      static_cast<double>(clipped_body.x) + 0.5 * clipped_body.width;
  const int roi_width = std::max(
      8, static_cast<int>(std::ceil(
             clipped_body.width * (1.0 + 2.0 * head_roi_side_padding_ratio_))));
  const int expand_up =
      std::min(clipped_body.y,
               static_cast<int>(std::ceil(clipped_body.height *
                                          head_roi_top_expansion_ratio_)));
  const int roi_y = clipped_body.y - expand_up;
  const int base_height =
      std::max(8, static_cast<int>(
                      std::ceil(clipped_body.height * head_roi_height_ratio_)));
  const int roi_height =
      std::min(image_size.height - roi_y, base_height + expand_up);
  const int roi_x = static_cast<int>(std::floor(center_x - 0.5 * roi_width));

  return cv::Rect(roi_x, roi_y, roi_width, roi_height) & image_bounds;
}

bool SCRFDPipeline::detectBestFaceInRoi(const cv::Mat &bgr,
                                        const cv::Rect &head_roi,
                                        FaceObject &roi_face) {
  std::vector<FaceObject> faces;
  face_detector_->detect(bgr(head_roi), faces, face_confidence_threshold_,
                         face_nms_iou_threshold_);
  if (faces.empty()) {
    return false;
  }

  const auto best =
      std::max_element(faces.begin(), faces.end(),
                       [](const FaceObject &lhs, const FaceObject &rhs) {
                         return lhs.prob < rhs.prob;
                       });
  // 置信度相同时仍选择检测结果中先出现的人脸。
  roi_face = *best;
  return true;
}

void SCRFDPipeline::writeFaceResult(const FaceObject &roi_face,
                                    const cv::Rect &head_roi,
                                    const cv::Size &image_size,
                                    trt_infer_msgs::msg::PersonMeta &person,
                                    PersonFrameContext &person_context) {
  // 框和全部关键点必须一起从 ROI 局部坐标平移到彩色原图坐标。
  FaceObject global_face = roi_face;
  global_face.rect.x += static_cast<float>(head_roi.x);
  global_face.rect.y += static_cast<float>(head_roi.y);
  for (cv::Point2f &landmark : global_face.landmark) {
    landmark.x += static_cast<float>(head_roi.x);
    landmark.y += static_cast<float>(head_roi.y);
  }

  // 消息使用裁后整数框；上下文仍保留平移后的浮点框和关键点。
  const cv::Rect face_rect(
      static_cast<int>(std::floor(global_face.rect.x)),
      static_cast<int>(std::floor(global_face.rect.y)),
      static_cast<int>(std::ceil(global_face.rect.width)),
      static_cast<int>(std::ceil(global_face.rect.height)));
  const cv::Rect clipped_face =
      face_rect & cv::Rect(0, 0, image_size.width, image_size.height);
  if (clipped_face.width <= 0 || clipped_face.height <= 0) {
    return;
  }

  auto &face_detection = person.face_detection;

  face_detection.has_face = true;
  face_detection.face_confidence = global_face.prob;

  face_detection.face_bbox.x = clipped_face.x;
  face_detection.face_bbox.y = clipped_face.y;
  face_detection.face_bbox.w = clipped_face.width;
  face_detection.face_bbox.h = clipped_face.height;

  person_context.has_face = true;
  person_context.face = global_face;
}

void SCRFDPipeline::process(
    const cv::Mat &bgr,
    trt_infer_msgs::msg::PerceptionResult &perception_result,
    PerceptionFrameContext &frame_context) {
  // 1. 对齐人员数量并重置当前人脸结果，保留追踪器写入的轨迹信息。
  if (frame_context.persons.size() != perception_result.persons.size()) {
    frame_context.persons.resize(perception_result.persons.size());
  }
  for (std::size_t index = 0; index < perception_result.persons.size();
       ++index) {
    auto &person_context = frame_context.persons[index];
    person_context.has_face = false;
    auto &face_detection = perception_result.persons[index].face_detection;
    // 沿用总流程每帧新建消息的约定，不改变旧接口的 has_face 重置行为。
    face_detection.face_bbox.x = 0;
    face_detection.face_bbox.y = 0;
    face_detection.face_bbox.w = 0;
    face_detection.face_bbox.h = 0;
    face_detection.face_confidence = 0.0f;
  }
  if (!enabled_ || !face_detector_ || bgr.empty()) {
    return;
  }

  const auto start_time = std::chrono::high_resolution_clock::now();
  int attempted_person_rois = 0;

  for (std::size_t index = 0; index < perception_result.persons.size();
       ++index) {
    auto &person = perception_result.persons[index];
    if (attempted_person_rois >= max_person_roi_attempts_) {
      break;
    }

    // 2. 无效或过小 ROI 也占用一次额度，保持原有按人员顺序限额的规则。
    const cv::Rect head_roi =
        buildHeadShoulderRoi(person.body_detection.body_bbox, bgr.size());
    ++attempted_person_rois;
    if (head_roi.width < min_head_roi_side_px_ ||
        head_roi.height < min_head_roi_side_px_) {
      continue;
    }

    // 3. 每个人体只选一张最高置信度人脸，不做跨人体 ROI 去重。
    FaceObject roi_face;
    if (!detectBestFaceInRoi(bgr, head_roi, roi_face)) {
      continue;
    }

    // 4. 校验裁后人脸框，成功后同时写入消息和内部上下文。
    writeFaceResult(roi_face, head_roi, bgr.size(), person,
                    frame_context.persons[index]);
  }

  const std::chrono::duration<float, std::milli> duration =
      std::chrono::high_resolution_clock::now() - start_time;
  // std::cout << "[SCRFDPipeline] Processing time: " << duration.count() << "
  // ms"
  //           << std::endl;
  perception_result.face_detection_ms = duration.count();
}
