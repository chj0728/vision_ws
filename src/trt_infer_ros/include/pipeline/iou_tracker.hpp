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
   * @param frame_context 写入轨迹总帧数和当前仍保留的轨迹 ID。
   */
  void process(trt_infer_msgs::msg::PerceptionResult &perception_result,
               PerceptionFrameContext &frame_context);

  /** @brief 返回当前追踪器是否启用。 */
  bool isEnabled() const { return enabled_; }

  /**
   * @brief 获取当前未死亡的轨迹数量。
   *
   * @return 活跃轨迹数量。
   */
  int liveCount() const;

private:
  /** @brief 单条人体轨迹的内部状态。 */
  struct Track {
    int track_id{-1};    // 进程生命周期内单调递增的轨迹 ID
    cv::Rect body_bbox;  // 最近一次匹配到的人体框
    int age_frames{0};   // 连续未匹配的帧数
    int total_frames{0}; // 轨迹累计成功匹配的帧数
    bool is_dead{false}; // 是否已进入短期恢复等待状态
    std::chrono::steady_clock::time_point dead_since{}; // 轨迹死亡时间

    /**
     * @brief 计算当前轨迹框与目标框的 IoU。
     *
     * @param other 待比较的人体框。
     * @return IoU 值，范围为 [0, 1]。
     */
    float iouWith(const cv::Rect &other) const;
  };

  /** @brief 尝试使用较宽松的 IoU 阈值恢复死亡轨迹。 */
  Track *tryRevive(const cv::Rect &detection);

  /** @brief 增加所有活跃轨迹的未匹配帧数。 */
  void ageTracks();

  /** @brief 标记超龄轨迹并清理超过恢复窗口的死亡轨迹。 */
  void retireAndPurgeTracks();

  /** @brief 返回活跃轨迹和恢复窗口内死亡轨迹的 ID 集合。 */
  std::set<int> retainedTrackIds() const;

  bool enabled_{true};            // 是否启用人体追踪
  float iou_threshold_{0.30f};    // 活跃轨迹与检测框的最小匹配 IoU
  int max_age_frames_{30};        // 轨迹允许连续丢失的最大帧数
  int reid_window_seconds_{30};   // 死亡轨迹可被恢复的时间窗口，单位为秒
  float revive_iou_scale_{0.40f}; // 恢复阈值相对正常 IoU 阈值的缩放系数
  int next_id_{0};                // 下一条新轨迹使用的 ID
  std::map<int, Track> tracks_;   // 按 track_id 保存全部保留中的轨迹
};
