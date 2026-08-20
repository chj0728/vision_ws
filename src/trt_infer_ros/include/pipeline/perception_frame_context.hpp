#pragma once

#include "scrfd_trt/face_detector.h"

#include <set>
#include <vector>

/**
 * @brief 单个人在当前帧内由各 Pipeline 共享的中间数据。
 *
 * 该结构只在进程内部使用，不作为 ROS 消息发布。
 */
struct PersonFrameContext {
  int track_id{-1};          // IoU 追踪器分配的轨迹 ID
  int track_total_frames{0}; // 当前轨迹累计成功匹配的帧数
  bool has_face{false};      // 当前帧是否检测到有效人脸
  FaceObject face{};         // 全图坐标下的人脸框、关键点和置信度
};

/** @brief 当前帧所有人员及轨迹生命周期信息的内部上下文。 */
struct PerceptionFrameContext {
  int frame_number{0}; // Pipeline 启动后的递增帧编号
  std::vector<PersonFrameContext>
      persons;                      // 与 PerceptionResult.persons 顺序一致
  std::set<int> retained_track_ids; // 活跃或仍处于短期恢复窗口的轨迹 ID
};
