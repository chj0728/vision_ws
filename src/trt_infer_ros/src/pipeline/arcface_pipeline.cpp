#include "pipeline/arcface_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <stdexcept>

ArcFacePipeline::ArcFacePipeline(const YAML::Node &config) {
  loadParameters(config);
  initialize();
}

ArcFacePipeline::~ArcFacePipeline() = default;

void ArcFacePipeline::loadParameters(const YAML::Node &config) {
  const YAML::Node arcface_config = config["arcface_pipeline"];

  enabled_ = arcface_config["enable"].as<bool>(false);
  auto_register_ = arcface_config["auto_register"].as<bool>(true);
  require_head_pose_ = arcface_config["require_head_pose"].as<bool>(false);
  arcface_engine_name_ = arcface_config["arcface_engine_name"].as<std::string>(
      "w600k_r50_b16_gpu0_fp16.engine");

  const std::string configured_engine_path =
      arcface_config["arcface_engine_path"].as<std::string>("");
  if (!configured_engine_path.empty()) {
    const std::filesystem::path path(configured_engine_path);
    arcface_engine_path_ =
        (path.is_absolute() ? path
                            : std::filesystem::path(TRT_WORKSPACE_ROOT) / path)
            .string();
  } else {
    arcface_engine_path_ = (std::filesystem::path(TRT_WORKSPACE_ROOT) /
                            "models" / "arcface" / arcface_engine_name_)
                               .string();
  }

  const std::string configured_db_path =
      arcface_config["face_db_path"].as<std::string>("db/face_db.sqlite3");
  const std::filesystem::path db_path(configured_db_path);
  face_db_path_ = (db_path.is_absolute()
                       ? db_path
                       : std::filesystem::path(TRT_WORKSPACE_ROOT) / db_path)
                      .string();

  recog_threshold_ = std::clamp(
      arcface_config["recog_threshold"].as<float>(0.45f), 0.05f, 0.99f);
  min_face_px_ = std::max(16, arcface_config["min_face_px"].as<int>(64));
  min_face_confidence_ = std::clamp(
      arcface_config["min_face_confidence"].as<float>(0.75f), 0.0f, 1.0f);
  max_yaw_deg_ =
      std::clamp(arcface_config["max_yaw_deg"].as<float>(30.0f), 1.0f, 90.0f);
  max_pitch_deg_ =
      std::clamp(arcface_config["max_pitch_deg"].as<float>(25.0f), 1.0f, 90.0f);
  min_track_frames_ =
      std::max(1, arcface_config["min_track_frames"].as<int>(20));
  recheck_interval_frames_ =
      std::max(1, arcface_config["recheck_interval_frames"].as<int>(150));
  embedding_buffer_size_ =
      std::clamp(arcface_config["embedding_buffer_size"].as<int>(5), 1,
                 FaceDatabase::kMaxEmbeddings);
}

void ArcFacePipeline::initialize() {
  if (!enabled_) {
    return;
  }
  if (!std::filesystem::exists(arcface_engine_path_)) {
    throw std::runtime_error("ArcFace engine not found: " +
                             arcface_engine_path_);
  }
  if (face_db_path_.empty()) {
    throw std::runtime_error("ArcFace database path is empty");
  }

  const std::filesystem::path parent =
      std::filesystem::path(face_db_path_).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }

  arcface_engine_ptr_ =
      std::make_unique<ArcFaceTRT>(arcface_engine_path_, recog_threshold_);
  face_database_ptr_ = std::make_unique<FaceDatabase>();
  if (face_database_ptr_->open(face_db_path_) < 0 ||
      !face_database_ptr_->isOpen()) {
    throw std::runtime_error("Failed to open ArcFace database: " +
                             face_db_path_);
  }
}

bool ArcFacePipeline::passesQualityGate(
    const PersonFrameContext &person_context,
    const trt_infer_msgs::msg::PersonMeta &person) const {
  const auto &face_detection = person.face_detection;
  const auto &face_bbox = face_detection.face_bbox;

  if (!person_context.has_face || person_context.track_id < 0 ||
      person_context.track_total_frames < min_track_frames_ ||
      face_detection.face_confidence < min_face_confidence_ ||
      face_bbox.w < min_face_px_ || face_bbox.h < min_face_px_) {
    return false;
  }

  if (!require_head_pose_) {
    return true;
  }
  const float yaw = person.head_pose.yaw;
  const float pitch = person.head_pose.pitch;
  return std::isfinite(yaw) && std::isfinite(pitch) &&
         std::abs(yaw) <= max_yaw_deg_ && std::abs(pitch) <= max_pitch_deg_;
}

bool ArcFacePipeline::extractEmbedding(const cv::Mat &rgb,
                                       const PersonFrameContext &person_context,
                                       FaceEmbedding &embedding) const {
  if (!arcface_engine_ptr_ || !person_context.has_face) {
    return false;
  }
  const cv::Mat aligned =
      ArcFaceTRT::alignFace(rgb, person_context.face.landmark);
  return !aligned.empty() &&
         arcface_engine_ptr_->extractEmbedding(aligned, embedding);
}

void ArcFacePipeline::clearPendingBuffer(RecognitionState &state) {
  state.embedding_buffer.clear();
  state.yaw_buffer.clear();
  state.pitch_buffer.clear();
  state.confidence_buffer.clear();
}

void ArcFacePipeline::processPending(
    RecognitionState &state, const FaceEmbedding &embedding,
    const trt_infer_msgs::msg::PersonMeta &person, int frame_number) {
  state.embedding_buffer.push_back(embedding);
  state.yaw_buffer.push_back(person.head_pose.yaw);
  state.pitch_buffer.push_back(person.head_pose.pitch);
  state.confidence_buffer.push_back(person.face_detection.face_confidence);

  if (static_cast<int>(state.embedding_buffer.size()) <
      embedding_buffer_size_) {
    return;
  }

  const FaceMatch match =
      face_database_ptr_->identify(embedding, recog_threshold_);
  state.last_recog_frame = frame_number;
  if (match.identified) {
    state.status = RecognitionStatus::Identified;
    state.person_uuid = match.uuid;
    state.person_name = match.name;
    state.confidence = match.similarity;
    face_database_ptr_->touchPerson(match.uuid);
  } else if (auto_register_) {
    const std::string uuid = face_database_ptr_->registerPerson(
        "", state.embedding_buffer, state.yaw_buffer, state.pitch_buffer,
        state.confidence_buffer);
    if (!uuid.empty()) {
      state.status = RecognitionStatus::Identified;
      state.person_uuid = uuid;
      state.person_name.clear();
      state.confidence = 0.0f;
    }
  }
  clearPendingBuffer(state);
}

void ArcFacePipeline::processIdentified(RecognitionState &state,
                                        const FaceEmbedding &embedding,
                                        int frame_number) {
  const FaceMatch match =
      face_database_ptr_->identify(embedding, recog_threshold_);
  state.last_recog_frame = frame_number;
  if (match.identified) {
    state.person_uuid = match.uuid;
    state.person_name = match.name;
    state.confidence = match.similarity;
    face_database_ptr_->touchPerson(match.uuid);
    return;
  }

  state.status = RecognitionStatus::Pending;
  state.person_uuid.clear();
  state.person_name.clear();
  state.confidence = 0.0f;
  clearPendingBuffer(state);
}

void ArcFacePipeline::pruneStates(const std::set<int> &retained_track_ids) {
  for (auto it = recognition_states_.begin();
       it != recognition_states_.end();) {
    if (retained_track_ids.count(it->first) == 0U) {
      it = recognition_states_.erase(it);
    } else {
      ++it;
    }
  }
}

void ArcFacePipeline::clearMessage(trt_infer_msgs::msg::FaceRecog &face_recog) {
  face_recog.person_uuid.clear();
  face_recog.person_name.clear();
  face_recog.face_recog_conf = 0.0f;
  face_recog.face_embedding.fill(0.0f);
}

void ArcFacePipeline::writeEmbedding(
    const FaceEmbedding &embedding,
    trt_infer_msgs::msg::FaceRecog &face_recog) {
  std::copy(std::begin(embedding.v), std::end(embedding.v),
            face_recog.face_embedding.begin());
}

void ArcFacePipeline::writeIdentity(
    const RecognitionState &state, trt_infer_msgs::msg::FaceRecog &face_recog) {
  if (state.status != RecognitionStatus::Identified) {
    return;
  }
  face_recog.person_uuid = state.person_uuid;
  face_recog.person_name = state.person_name;
  face_recog.face_recog_conf = std::clamp(state.confidence, 0.0f, 1.0f);
}

void ArcFacePipeline::process(
    const cv::Mat &rgb, const PerceptionFrameContext &frame_context,
    trt_infer_msgs::msg::PerceptionResult &perception_result) {
  for (auto &person : perception_result.persons) {
    clearMessage(person.face_recog);
  }
  if (!enabled_ || !arcface_engine_ptr_ || !face_database_ptr_ || rgb.empty()) {
    return;
  }

  pruneStates(frame_context.retained_track_ids);
  const std::size_t person_count =
      std::min(perception_result.persons.size(), frame_context.persons.size());

  const auto start_time = std::chrono::high_resolution_clock::now();

  for (std::size_t index = 0; index < person_count; ++index) {
    auto &person = perception_result.persons[index];
    const auto &person_context = frame_context.persons[index];
    if (person_context.track_id < 0) {
      continue;
    }

    auto state_it = recognition_states_.find(person_context.track_id);
    if (state_it == recognition_states_.end()) {
      state_it = recognition_states_
                     .emplace(person_context.track_id, RecognitionState{})
                     .first;
    }
    RecognitionState &state = state_it->second;

    bool should_extract = false;
    if (passesQualityGate(person_context, person)) {
      should_extract = state.status == RecognitionStatus::Pending ||
                       (frame_context.frame_number - state.last_recog_frame) >=
                           recheck_interval_frames_;
    }

    if (should_extract) {
      FaceEmbedding embedding{};
      if (extractEmbedding(rgb, person_context, embedding)) {
        std::cout << "Embedding for track ID " << person_context.track_id
                  << ": [";
        for (int i = 0; i < 5; ++i) {
          std::cout << (i == 0 ? "" : ", ") << embedding.v[i];
        }
        std::cout << ", ...]" << std::endl;

        writeEmbedding(embedding, person.face_recog);
        if (state.status == RecognitionStatus::Pending) {
          processPending(state, embedding, person, frame_context.frame_number);
        } else {
          processIdentified(state, embedding, frame_context.frame_number);
        }
      }
    }
    writeIdentity(state, person.face_recog);
  }

  std::chrono::duration<float, std::milli> pipeline_duration =
      std::chrono::high_resolution_clock::now() - start_time;
  std::cout << "[ArcFacePipeline] Processing time: "
            << pipeline_duration.count() << " ms" << std::endl;
}

bool ArcFacePipeline::updatePersonName(const std::string &uuid,
                                       const std::string &name) {
  if (!enabled_ || !face_database_ptr_ || uuid.empty()) {
    return false;
  }
  if (!face_database_ptr_->updateName(uuid, name)) {
    return false;
  }
  for (auto &[track_id, state] : recognition_states_) {
    (void)track_id;
    if (state.person_uuid == uuid) {
      state.person_name = name;
    }
  }
  return true;
}
