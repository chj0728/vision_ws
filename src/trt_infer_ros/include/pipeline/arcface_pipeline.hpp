#pragma once

#include "arcface_trt/arcface_trt.h"
#include "arcface_trt/face_database.h"
#include "pipeline/perception_frame_context.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "trt_infer_msgs/msg/perception_result.hpp"

// 识别状态按 track_id 独立维护，SCRFD 只提供本帧关键点。
class ArcFacePipeline {
public:
  explicit ArcFacePipeline(const YAML::Node &config);
  ~ArcFacePipeline();

  void loadParameters(const YAML::Node &config);
  void initialize();
  void process(const cv::Mat &rgb, const PerceptionFrameContext &frame_context,
               trt_infer_msgs::msg::PerceptionResult &perception_result);

  bool updatePersonName(const std::string &uuid, const std::string &name);
  bool isEnabled() const { return enabled_; }
  std::string getEnginePath() const { return arcface_engine_path_; }
  std::string getDatabasePath() const { return face_db_path_; }

private:
  enum class RecognitionStatus { Pending, Identified };

  struct RecognitionState {
    RecognitionStatus status{RecognitionStatus::Pending};
    std::string person_uuid;
    std::string person_name;
    float confidence{0.0f};
    int last_recog_frame{-1};
    std::vector<FaceEmbedding> embedding_buffer;
    std::vector<float> yaw_buffer;
    std::vector<float> pitch_buffer;
    std::vector<float> confidence_buffer;
  };

  bool passesQualityGate(
      const PersonFrameContext &person_context,
      const trt_infer_msgs::msg::PersonMeta &person) const;
  bool extractEmbedding(const cv::Mat &rgb,
                        const PersonFrameContext &person_context,
                        FaceEmbedding &embedding) const;
  void processPending(RecognitionState &state, const FaceEmbedding &embedding,
                      const trt_infer_msgs::msg::PersonMeta &person,
                      int frame_number);
  void processIdentified(RecognitionState &state,
                         const FaceEmbedding &embedding, int frame_number);
  void clearPendingBuffer(RecognitionState &state);
  void pruneStates(const std::set<int> &retained_track_ids);
  static void clearMessage(trt_infer_msgs::msg::FaceRecog &face_recog);
  static void writeEmbedding(const FaceEmbedding &embedding,
                             trt_infer_msgs::msg::FaceRecog &face_recog);
  static void writeIdentity(const RecognitionState &state,
                            trt_infer_msgs::msg::FaceRecog &face_recog);

  std::unique_ptr<ArcFaceTRT> arcface_engine_ptr_;
  std::unique_ptr<FaceDatabase> face_database_ptr_;
  std::map<int, RecognitionState> recognition_states_;

  bool enabled_{false};
  bool auto_register_{true};
  bool require_head_pose_{false};
  std::string arcface_engine_name_;
  std::string arcface_engine_path_;
  std::string face_db_path_;
  float recog_threshold_{0.45f};
  int min_face_px_{64};
  float min_face_confidence_{0.75f};
  float max_yaw_deg_{30.0f};
  float max_pitch_deg_{25.0f};
  int min_track_frames_{20};
  int recheck_interval_frames_{150};
  int embedding_buffer_size_{5};
};
