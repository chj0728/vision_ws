#pragma once

#include "pipeline/perception_frame_context.hpp"

#include <chrono>
#include <map>
#include <set>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "trt_infer_msgs/msg/perception_result.hpp"

/**
 * @brief 基于人体框 IoU 的轻量级多目标追踪器。
 *
 * 该模块只负责人体检测与 track_id 的时序关联，不保存人脸识别状态。
 */
class IouTracker {
public:
  /**
   * @brief 创建追踪器并加载 iou_tracker 配置。
   *
   * @param config 完整的 Pipeline YAML 配置节点。
   */
  explicit IouTracker(const YAML::Node &config);

  /**
   * @brief 读取 IoU 阈值、轨迹寿命和短期恢复参数。
   *
   * @param config 完整的 Pipeline YAML 配置节点。
   */
  void loadParameters(const YAML::Node &config);

  /**
   * @brief 使用当前帧人体框更新轨迹并写入 track_id。
   *
   * @param perception_result 输入人体检测结果，并写入每个人的 track_id。
   * @param frame_context 写入累计匹配帧数和当前仍保留的轨迹 ID。
   */
  void process(trt_infer_msgs::msg::PerceptionResult &perception_result,
               PerceptionFrameContext &frame_context);

  /** @brief 返回当前追踪器是否启用。 */
  bool isEnabled() const { return enabled_; }

  /**
   * @brief 获取当前活跃的轨迹数量。
   *
   * @return 活跃轨迹数量。
   */
  int activeTrackCount() const;

  /** @brief 保留原计数接口，含义与 activeTrackCount() 相同。 */
  int liveCount() const { return activeTrackCount(); }

private:
  /** @brief 单条人体轨迹的内部状态。 */
  struct Track {
    int track_id{-1};        // 追踪器实例内单调递增的 ID，恢复时不变
    cv::Rect last_body_bbox; // 最近一次匹配的人体框，不做运动预测
    int missed_frames{0};    // 连续失配帧数；帧首先加一，匹配后清零
    int matched_frames{0};   // 累计匹配帧数，包含新建帧，不要求连续
    bool is_inactive{false}; // 是否已失活并进入短期恢复等待状态
    std::chrono::steady_clock::time_point
        inactive_since{}; // 进入等待状态的时间

    /**
     * @brief 计算当前轨迹框与目标框的 IoU。
     *
     * @param other 待比较的人体框。
     * @return IoU 值，范围为 [0, 1]。
     */
    float computeIou(const cv::Rect &other) const;

    /** @brief 写入匹配框、清零失配帧数并累计命中；恢复时重新激活。 */
    void updateFromDetection(const cv::Rect &body_bbox);
  };

  /** @brief 按 IoU 降序贪心匹配活跃轨迹，输出与检测索引对齐的轨迹指针。 */
  std::vector<Track *>
  matchActiveTracks(const std::vector<cv::Rect> &body_boxes);

  /** @brief 为未匹配的有效检测恢复旧轨迹或创建新轨迹。 */
  void recoverOrCreateTracks(const std::vector<cv::Rect> &body_boxes,
                             std::vector<Track *> &matched_tracks);

  /** @brief 查找窗口内满足宽松阈值的最佳死亡轨迹，只查找、不修改。 */
  Track *findRecoveryCandidate(const cv::Rect &detection);

  /** @brief 增加所有活跃轨迹的未匹配帧数。 */
  void incrementMissedFrames();

  /** @brief 标记超龄轨迹并清理超过恢复窗口的不活跃轨迹。 */
  void retireAndRemoveExpiredTracks();

  /** @brief 返回活跃轨迹和恢复窗口内不活跃轨迹的 ID 集合。 */
  std::set<int> collectRetainedTrackIds() const;

  bool enabled_{true};                // 是否启用人体追踪
  float match_iou_threshold_{0.30f};  // 正常匹配要求 IoU >= 此值
  int max_missed_frames_{30};         // 连续失配超过此值才进入死亡状态
  int recovery_window_seconds_{30};   // 不活跃轨迹恢复窗口，按整数秒比较
  float recovery_iou_scale_{0.40f};   // 恢复要求 IoU > 正常阈值乘此系数
  int next_track_id_{0};              // 下一个新 ID，已清理的 ID 不复用
  std::map<int, Track> tracks_by_id_; // 保存活跃轨迹及恢复窗口内不活跃轨迹
};
