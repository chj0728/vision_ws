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

cv::Rect makeHeadRoi(const trt_infer_msgs::msg::BoundingBox &body,
                     int image_width, int image_height, double height_ratio,
                     double width_pad_ratio, double top_expand_ratio) {
  const cv::Rect image_bounds(0, 0, image_width, image_height);
  if (body.w <= 0 || body.h <= 0) {
    return {};
  }
  const cv::Rect body_rect(body.x, body.y, body.w, body.h);
  const cv::Rect clipped_body = body_rect & image_bounds;
  if (clipped_body.width < 4 || clipped_body.height < 4) {
    return {};
  }

  const double center_x =
      static_cast<double>(clipped_body.x) + 0.5 * clipped_body.width;
  const int roi_width =
      std::max(8, static_cast<int>(std::ceil(clipped_body.width *
                                             (1.0 + 2.0 * width_pad_ratio))));
  const int expand_up = std::min(
      clipped_body.y,
      static_cast<int>(std::ceil(clipped_body.height * top_expand_ratio)));
  const int roi_y = clipped_body.y - expand_up;
  const int base_height = std::max(
      8, static_cast<int>(std::ceil(clipped_body.height * height_ratio)));
  const int roi_height =
      std::min(image_height - roi_y, base_height + expand_up);
  const int roi_x = static_cast<int>(std::floor(center_x - 0.5 * roi_width));

  return cv::Rect(roi_x, roi_y, roi_width, roi_height) & image_bounds;
}

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
  const YAML::Node scrfd_config = config["scrfd_pipeline"];

  enabled_ = scrfd_config["enable"].as<bool>(false);

  scrfd_engine_name_ = scrfd_config["scrfd_engine_name"].as<std::string>(
      "scrfd_2.5g_bnkps_shape640x640.trt");
  const std::string configured_path =
      scrfd_config["scrfd_engine_path"].as<std::string>("");
  if (!configured_path.empty()) {
    const std::filesystem::path path(configured_path);
    scrfd_engine_path_ =
        (path.is_absolute() ? path
                            : std::filesystem::path(TRT_WORKSPACE_ROOT) / path)
            .string();
  } else {
    scrfd_engine_path_ = (std::filesystem::path(TRT_WORKSPACE_ROOT) / "models" /
                          "scrfd" / scrfd_engine_name_)
                             .string();
  }

  preprocess_ =
      toLower(scrfd_config["preprocess"].as<std::string>("insightface"));
  prob_threshold_ =
      std::clamp(scrfd_config["prob_threshold"].as<float>(0.38f), 0.08f, 0.95f);
  nms_threshold_ =
      std::clamp(scrfd_config["nms_threshold"].as<float>(0.45f), 0.15f, 0.95f);
  person_head_height_ratio_ = std::clamp(
      scrfd_config["person_head_height_ratio"].as<double>(0.58), 0.18, 0.78);
  person_head_width_pad_ratio_ = std::clamp(
      scrfd_config["person_head_width_pad_ratio"].as<double>(0.24), 0.0, 0.55);
  person_head_top_expand_ratio_ = std::clamp(
      scrfd_config["person_head_top_expand_ratio"].as<double>(0.14), 0.0, 0.5);
  face_roi_min_side_ =
      std::max(32, scrfd_config["face_roi_min_side"].as<int>(48));
  max_person_rois_ = std::max(1, scrfd_config["max_person_rois"].as<int>(8));
}

void SCRFDPipeline::initialize() {

  if (!enabled_) {
    return;
  }

  if (!std::filesystem::exists(scrfd_engine_path_)) {
    throw std::runtime_error("SCRFD engine not found: " + scrfd_engine_path_);
  }

  const bool use_insightface = preprocess_ == "insightface";
  const bool use_namdvt = preprocess_ == "namdvt" || preprocess_ == "upstream";
  if (!use_insightface && !use_namdvt) {
    throw std::invalid_argument("Unsupported SCRFD preprocess mode: " +
                                preprocess_);
  }

  scrfd_engine_ptr_ = std::make_unique<SCRFD_TRT>(scrfd_engine_path_);
  scrfd_engine_ptr_->setPreprocess(
      use_insightface ? SCRFD_TRT::Preprocess::InsightFacePython
                      : SCRFD_TRT::Preprocess::NamdvtUpstream);
}

void SCRFDPipeline::process(
    const cv::Mat &rgb,
    trt_infer_msgs::msg::PerceptionResult &perception_result,
    PerceptionFrameContext &frame_context) {
  if (frame_context.persons.size() != perception_result.persons.size()) {
    frame_context.persons.resize(perception_result.persons.size());
  }
  for (std::size_t index = 0; index < perception_result.persons.size(); ++index) {
    auto &person_context = frame_context.persons[index];
    person_context.has_face = false;
    auto &face_detection = perception_result.persons[index].face_detection;
    face_detection.face_bbox.x = 0;
    face_detection.face_bbox.y = 0;
    face_detection.face_bbox.w = 0;
    face_detection.face_bbox.h = 0;
    face_detection.face_confidence = 0.0f;
  }
  if (!enabled_ || !scrfd_engine_ptr_ || rgb.empty()) {
    return;
  }

  const auto start_time = std::chrono::high_resolution_clock::now();
  int processed_rois = 0;

  for (std::size_t index = 0; index < perception_result.persons.size(); ++index) {
    auto &person = perception_result.persons[index];
    if (processed_rois >= max_person_rois_) {
      break;
    }

    const cv::Rect roi =
        makeHeadRoi(person.body_detection.body_bbox, rgb.cols, rgb.rows,
                    person_head_height_ratio_, person_head_width_pad_ratio_,
                    person_head_top_expand_ratio_);
    ++processed_rois;
    if (roi.width < face_roi_min_side_ || roi.height < face_roi_min_side_) {
      continue;
    }

    std::vector<FaceObject> faces;
    scrfd_engine_ptr_->detect(rgb(roi), faces, prob_threshold_, nms_threshold_);
    if (faces.empty()) {
      continue;
    }

    const auto best =
        std::max_element(faces.begin(), faces.end(),
                         [](const FaceObject &lhs, const FaceObject &rhs) {
                           return lhs.prob < rhs.prob;
                         });
    FaceObject global_face = *best;
    global_face.rect.x += static_cast<float>(roi.x);
    global_face.rect.y += static_cast<float>(roi.y);
    for (cv::Point2f &landmark : global_face.landmark) {
      landmark.x += static_cast<float>(roi.x);
      landmark.y += static_cast<float>(roi.y);
    }

    const cv::Rect face_rect(
        static_cast<int>(std::floor(global_face.rect.x)),
        static_cast<int>(std::floor(global_face.rect.y)),
        static_cast<int>(std::ceil(global_face.rect.width)),
        static_cast<int>(std::ceil(global_face.rect.height)));
    const cv::Rect clipped_face =
        face_rect & cv::Rect(0, 0, rgb.cols, rgb.rows);
    if (clipped_face.width <= 0 || clipped_face.height <= 0) {
      continue;
    }

    auto &face_detection = person.face_detection;
    face_detection.face_bbox.x = clipped_face.x;
    face_detection.face_bbox.y = clipped_face.y;
    face_detection.face_bbox.w = clipped_face.width;
    face_detection.face_bbox.h = clipped_face.height;
    face_detection.face_confidence = global_face.prob;

    auto &person_context = frame_context.persons[index];
    person_context.has_face = true;
    person_context.face = global_face;
  }

  const std::chrono::duration<float, std::milli> duration =
      std::chrono::high_resolution_clock::now() - start_time;
  std::cout << "[SCRFDPipeline] Processing time: " << duration.count() << " ms"
            << std::endl;
}
