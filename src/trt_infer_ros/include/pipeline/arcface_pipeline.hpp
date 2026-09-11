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

/**
 * @brief 基于 ArcFace 特征和 SQLite 人脸库执行人脸识别。
 *
 * 识别状态按 track_id 独立维护，SCRFD 通过帧上下文提供五点关键点。
 */
class ArcFacePipeline {
public:
  /**
   * @brief 创建 ArcFace Pipeline，并完成参数、引擎和数据库初始化。
   *
   * @param config 完整的 Pipeline YAML 配置节点。
   */
  explicit ArcFacePipeline(const YAML::Node &config);
  ~ArcFacePipeline();

  /**
   * @brief 从 arcface_pipeline 配置节点读取识别参数。
   *
   * @param config 完整的 Pipeline YAML 配置节点。
   */
  void loadParameters(const YAML::Node &config);

  /** @brief 加载 ArcFace TensorRT 引擎并打开 SQLite 人脸库。 */
  void initialize();

  /**
   * @brief 对当前帧有效人脸执行对齐、特征提取和身份识别。
   *
   * @param bgr 当前帧 BGR 彩色图像。
   * @param frame_context 当前帧轨迹信息和 SCRFD 五点关键点。
   * @param perception_result 写入 UUID、姓名和相似度；特征只在进程内部使用。
   */
  void process(const cv::Mat &bgr, const PerceptionFrameContext &frame_context,
               trt_infer_msgs::msg::PerceptionResult &perception_result);

  /**
   * @brief 更新数据库及内存识别状态中的人物姓名。
   *
   * @param uuid 需要更新的人物 UUID。
   * @param name 新的人物姓名。
   * @return 更新成功返回 true，否则返回 false。
   */
  bool updatePersonName(const std::string &uuid, const std::string &name);

  /** @brief 返回当前模块是否启用。 */
  bool isEnabled() const { return enabled_; }

  /** @brief 返回最终解析得到的 ArcFace 引擎路径。 */
  std::string getEnginePath() const { return engine_path_; }

  /** @brief 返回最终解析得到的人脸数据库路径。 */
  std::string getDatabasePath() const { return database_path_; }

private:
  /** @brief 单条轨迹当前所处的识别阶段。 */
  enum class RecognitionStage { Pending, Identified };

  /** @brief 与 track_id 绑定的跨帧人脸识别状态。 */
  struct TrackIdentityState {
    RecognitionStage stage{RecognitionStage::Pending}; // 当前识别阶段
    std::string person_uuid;                           // 已识别人物的 UUID
    std::string person_name;                           // 数据库中的人物姓名
    float similarity{0.0f};   // 最近一次识别的余弦相似度
    int last_query_frame{-1}; // 最近一次数据库查询的帧号，不是特征提取帧号
    std::vector<FaceEmbedding> embedding_buffer; // 待识别或注册的特征缓冲
    std::vector<float> yaw_buffer;               // 与特征对应的偏航角元数据
    std::vector<float> pitch_buffer;             // 与特征对应的俯仰角元数据
    std::vector<float> face_confidence_buffer;   // 与特征对应的人脸置信度
  };

  /** @brief 质量和时间条件满足时提取特征，再执行首次识别或身份重验。 */
  void updateIdentityIfReady(const cv::Mat &bgr,
                             const PersonFrameContext &person_context,
                             const trt_infer_msgs::msg::PersonMeta &person,
                             int frame_number, TrackIdentityState &state);

  /** @brief 检查人脸尺寸、置信度、轨迹累计匹配帧数和可选头姿条件。 */
  bool passesQualityGate(const PersonFrameContext &person_context,
                         const trt_infer_msgs::msg::PersonMeta &person) const;

  /** @brief 根据 SCRFD 五点关键点对齐人脸并提取 512 维特征。 */
  bool extractAlignedEmbedding(const cv::Mat &bgr,
                               const PersonFrameContext &person_context,
                               FaceEmbedding &embedding) const;

  /** @brief 累积待识别特征，并在缓冲满足条件后识别或自动注册。 */
  void collectAndIdentify(TrackIdentityState &state,
                          const FaceEmbedding &embedding,
                          const trt_infer_msgs::msg::PersonMeta &person,
                          int frame_number);

  /** @brief 对已识别轨迹执行周期性身份重验。 */
  void recheckIdentity(TrackIdentityState &state,
                       const FaceEmbedding &embedding, int frame_number);

  /** @brief 同步清空待识别特征及配套质量元数据。 */
  void clearPendingSamples(TrackIdentityState &state);

  /** @brief 删除已经超过追踪器保留周期的识别状态。 */
  void removeUnretainedTrackStates(const std::set<int> &retained_track_ids);

  /** @brief 将人脸识别消息恢复为未识别状态。 */
  static void clearIdentityMessage(trt_infer_msgs::msg::FaceRecog &face_recog);

  /** @brief 将已确认的 UUID、姓名和相似度写入 ROS 消息。 */
  static void writeIdentityMessage(const TrackIdentityState &state,
                                   trt_infer_msgs::msg::FaceRecog &face_recog);

  std::unique_ptr<ArcFaceTRT> embedding_extractor_; // ArcFace TensorRT 推理实例
  std::unique_ptr<FaceDatabase> face_database_;     // SQLite 人脸数据库
  std::map<int, TrackIdentityState>
      identity_states_by_track_id_; // track_id 到识别状态

  bool enabled_{false};           // 是否启用人脸识别
  bool auto_register_{true};      // 未匹配到人物时是否自动注册
  bool require_head_pose_{false}; // 质量门控是否检查 yaw/pitch 角度范围
  std::string engine_filename_;   // 默认引擎文件名
  std::string engine_path_;       // 解析后的引擎绝对路径
  std::string database_path_;     // SQLite 人脸数据库绝对路径
  float identity_similarity_threshold_{0.45f}; // 判定为同一人物的最小余弦相似度
  int min_face_side_px_{64};         // 允许提取特征的最小人脸边长，单位为像素
  float min_face_confidence_{0.75f}; // 允许提取特征的最小 SCRFD 置信度
  float max_yaw_deg_{30.0f};         // 质量门控允许的最大偏航角绝对值
  float max_pitch_deg_{25.0f};       // 质量门控允许的最大俯仰角绝对值
  int min_track_matched_frames_{
      20}; // 开始识别要求的最小累计匹配次数，不要求连续
  int recheck_interval_frames_{
      150}; // 已识别人物的重验间隔，按 Pipeline 帧号计算
  int required_embedding_count_{5}; // 首次识别或注册前累计的特征数量
};
