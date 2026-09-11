#include "pipeline/arcface_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>

ArcFacePipeline::ArcFacePipeline(const YAML::Node &config) {
  loadParameters(config);
  initialize();
}

ArcFacePipeline::~ArcFacePipeline() = default;

void ArcFacePipeline::loadParameters(const YAML::Node &config) {
  // 保留原有配置键名、默认值和范围限制，部署配置无需迁移。
  const YAML::Node arcface_config = config["arcface_pipeline"];

  enabled_ = arcface_config["enable"].as<bool>(false);
  auto_register_ = arcface_config["auto_register"].as<bool>(true);
  require_head_pose_ = arcface_config["require_head_pose"].as<bool>(false);
  engine_filename_ = arcface_config["arcface_engine_name"].as<std::string>(
      "w600k_r50_b16_gpu0_fp16.engine");

  const std::string configured_engine_path =
      arcface_config["arcface_engine_path"].as<std::string>("");
  if (!configured_engine_path.empty()) {
    const std::filesystem::path path(configured_engine_path);
    engine_path_ =
        (path.is_absolute() ? path
                            : std::filesystem::path(TRT_WORKSPACE_ROOT) / path)
            .string();
  } else {
    engine_path_ = (std::filesystem::path(TRT_WORKSPACE_ROOT) / "models" /
                    "arcface" / engine_filename_)
                       .string();
  }

  const std::string configured_db_path =
      arcface_config["face_db_path"].as<std::string>("db/face_db.sqlite3");
  const std::filesystem::path db_path(configured_db_path);
  database_path_ = (db_path.is_absolute()
                        ? db_path
                        : std::filesystem::path(TRT_WORKSPACE_ROOT) / db_path)
                       .string();

  identity_similarity_threshold_ = std::clamp(
      arcface_config["recog_threshold"].as<float>(0.45f), 0.05f, 0.99f);
  min_face_side_px_ = std::max(16, arcface_config["min_face_px"].as<int>(64));
  min_face_confidence_ = std::clamp(
      arcface_config["min_face_confidence"].as<float>(0.75f), 0.0f, 1.0f);
  max_yaw_deg_ =
      std::clamp(arcface_config["max_yaw_deg"].as<float>(30.0f), 1.0f, 90.0f);
  max_pitch_deg_ =
      std::clamp(arcface_config["max_pitch_deg"].as<float>(25.0f), 1.0f, 90.0f);
  min_track_matched_frames_ =
      std::max(1, arcface_config["min_track_frames"].as<int>(20));
  recheck_interval_frames_ =
      std::max(1, arcface_config["recheck_interval_frames"].as<int>(150));
  required_embedding_count_ =
      std::clamp(arcface_config["embedding_buffer_size"].as<int>(5), 1,
                 FaceDatabase::kMaxEmbeddings);
}

void ArcFacePipeline::initialize() {
  if (!enabled_) {
    return;
  }
  if (!std::filesystem::exists(engine_path_)) {
    throw std::runtime_error("ArcFace engine not found: " + engine_path_);
  }
  if (database_path_.empty()) {
    throw std::runtime_error("ArcFace database path is empty");
  }

  const std::filesystem::path parent =
      std::filesystem::path(database_path_).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }

  embedding_extractor_ = std::make_unique<ArcFaceTRT>(
      engine_path_, identity_similarity_threshold_);
  face_database_ = std::make_unique<FaceDatabase>();
  if (face_database_->open(database_path_) < 0 || !face_database_->isOpen()) {
    throw std::runtime_error("Failed to open ArcFace database: " +
                             database_path_);
  }
}

bool ArcFacePipeline::passesQualityGate(
    const PersonFrameContext &person_context,
    const trt_infer_msgs::msg::PersonMeta &person) const {
  const auto &face_detection = person.face_detection;
  const auto &face_bbox = face_detection.face_bbox;

  if (!person_context.has_face || person_context.track_id < 0 ||
      person_context.track_total_frames < min_track_matched_frames_ ||
      face_detection.face_confidence < min_face_confidence_ ||
      face_bbox.w < min_face_side_px_ || face_bbox.h < min_face_side_px_) {
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

bool ArcFacePipeline::extractAlignedEmbedding(
    const cv::Mat &bgr, const PersonFrameContext &person_context,
    FaceEmbedding &embedding) const {
  if (!embedding_extractor_ || !person_context.has_face) {
    return false;
  }
  const cv::Mat aligned =
      ArcFaceTRT::alignFace(bgr, person_context.face.landmark);
  return !aligned.empty() &&
         embedding_extractor_->extractEmbedding(aligned, embedding);
}

void ArcFacePipeline::clearPendingSamples(TrackIdentityState &state) {
  state.embedding_buffer.clear();
  state.yaw_buffer.clear();
  state.pitch_buffer.clear();
  state.face_confidence_buffer.clear();
}

void ArcFacePipeline::collectAndIdentify(
    TrackIdentityState &state, const FaceEmbedding &embedding,
    const trt_infer_msgs::msg::PersonMeta &person, int frame_number) {
  state.embedding_buffer.push_back(embedding);
  state.yaw_buffer.push_back(person.head_pose.yaw);
  state.pitch_buffer.push_back(person.head_pose.pitch);
  state.face_confidence_buffer.push_back(person.face_detection.face_confidence);

  if (static_cast<int>(state.embedding_buffer.size()) <
      required_embedding_count_) {
    return;
  }

  // 查询使用本次特征；缓冲里的多帧特征仅用于未命中后的注册。
  const FaceMatch match =
      face_database_->identify(embedding, identity_similarity_threshold_);
  state.last_query_frame = frame_number;
  if (match.identified) {
    state.stage = RecognitionStage::Identified;
    state.person_uuid = match.uuid;
    state.person_name = match.name;
    state.similarity = match.similarity;
    face_database_->touchPerson(match.uuid);
  } else if (auto_register_) {
    const std::string uuid = face_database_->registerPerson(
        "", state.embedding_buffer, state.yaw_buffer, state.pitch_buffer,
        state.face_confidence_buffer);
    if (!uuid.empty()) {
      state.stage = RecognitionStage::Identified;
      state.person_uuid = uuid;
      state.person_name.clear();
      state.similarity = 0.0f;
    }
  }
  // 查询后总是清空本轮样本；未命中且未注册时下次重新积累。
  clearPendingSamples(state);
}

void ArcFacePipeline::recheckIdentity(TrackIdentityState &state,
                                      const FaceEmbedding &embedding,
                                      int frame_number) {
  const FaceMatch match =
      face_database_->identify(embedding, identity_similarity_threshold_);
  state.last_query_frame = frame_number;
  if (match.identified) {
    state.person_uuid = match.uuid;
    state.person_name = match.name;
    state.similarity = match.similarity;
    face_database_->touchPerson(match.uuid);
    return;
  }

  // 重验失败退回待识别；本次重验特征不作为新一轮缓冲的首条。
  state.stage = RecognitionStage::Pending;
  state.person_uuid.clear();
  state.person_name.clear();
  state.similarity = 0.0f;
  clearPendingSamples(state);
}

void ArcFacePipeline::removeUnretainedTrackStates(
    const std::set<int> &retained_track_ids) {
  for (auto it = identity_states_by_track_id_.begin();
       it != identity_states_by_track_id_.end();) {
    if (retained_track_ids.count(it->first) == 0U) {
      it = identity_states_by_track_id_.erase(it);
    } else {
      ++it;
    }
  }
}

void ArcFacePipeline::clearIdentityMessage(
    trt_infer_msgs::msg::FaceRecog &face_recog) {
  face_recog.person_uuid.clear();
  face_recog.person_name.clear();
  face_recog.face_recog_conf = 0.0f;
}

void ArcFacePipeline::writeIdentityMessage(
    const TrackIdentityState &state,
    trt_infer_msgs::msg::FaceRecog &face_recog) {
  if (state.stage != RecognitionStage::Identified) {
    return;
  }
  face_recog.person_uuid = state.person_uuid;
  face_recog.person_name = state.person_name;
  face_recog.face_recog_conf = std::clamp(state.similarity, 0.0f, 1.0f);
}

void ArcFacePipeline::updateIdentityIfReady(
    const cv::Mat &bgr, const PersonFrameContext &person_context,
    const trt_infer_msgs::msg::PersonMeta &person, int frame_number,
    TrackIdentityState &state) {
  if (!passesQualityGate(person_context, person)) {
    return;
  }
  // 已识别轨迹只在到期且质量合格时重验，间隔按 Pipeline 帧号计算。
  if (state.stage == RecognitionStage::Identified &&
      frame_number - state.last_query_frame < recheck_interval_frames_) {
    return;
  }

  FaceEmbedding embedding{};
  if (!extractAlignedEmbedding(bgr, person_context, embedding)) {
    return;
  }
  // 提取失败或质量不足不会清空已有缓冲，也不会推进查询帧号。
  if (state.stage == RecognitionStage::Pending) {
    collectAndIdentify(state, embedding, person, frame_number);
  } else {
    recheckIdentity(state, embedding, frame_number);
  }
}

void ArcFacePipeline::process(
    const cv::Mat &bgr, const PerceptionFrameContext &frame_context,
    trt_infer_msgs::msg::PerceptionResult &perception_result) {
  // 1. 先清空当前消息；历史身份仍由轨迹状态保存。
  for (auto &person : perception_result.persons) {
    clearIdentityMessage(person.face_recog);
  }
  if (!enabled_ || !embedding_extractor_ || !face_database_ || bgr.empty()) {
    return;
  }

  // 2. 只移除追踪器不再保留的内存状态，不删除数据库人物。
  removeUnretainedTrackStates(frame_context.retained_track_ids);
  const std::size_t person_count =
      std::min(perception_result.persons.size(), frame_context.persons.size());

  const auto start_time = std::chrono::high_resolution_clock::now();

  for (std::size_t index = 0; index < person_count; ++index) {
    auto &person = perception_result.persons[index];
    const auto &person_context = frame_context.persons[index];
    if (person_context.track_id < 0) {
      continue;
    }

    // 3. 为有效轨迹取得独立状态；质量不足时也保留待识别状态。
    auto [state_it, inserted] =
        identity_states_by_track_id_.try_emplace(person_context.track_id);
    (void)inserted;
    TrackIdentityState &state = state_it->second;
    updateIdentityIfReady(bgr, person_context, person,
                          frame_context.frame_number, state);

    // 4. 无论本帧是否提取特征，都回写已确认身份，允许复用缓存。
    writeIdentityMessage(state, person.face_recog);
  }

  std::chrono::duration<float, std::milli> pipeline_duration =
      std::chrono::high_resolution_clock::now() - start_time;
  perception_result.face_recog_ms = pipeline_duration.count();
}

bool ArcFacePipeline::updatePersonName(const std::string &uuid,
                                       const std::string &name) {
  if (!enabled_ || !face_database_ || uuid.empty()) {
    return false;
  }
  if (!face_database_->updateName(uuid, name)) {
    return false;
  }
  for (auto &[track_id, state] : identity_states_by_track_id_) {
    (void)track_id;
    if (state.person_uuid == uuid) {
      state.person_name = name;
    }
  }
  return true;
}
