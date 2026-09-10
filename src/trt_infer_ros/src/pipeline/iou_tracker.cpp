#include "pipeline/iou_tracker.hpp"

#include <algorithm>

namespace {

cv::Rect toValidRect(const trt_infer_msgs::msg::BoundingBox &bbox) {
  if (bbox.w <= 0 || bbox.h <= 0) {
    return {};
  }
  return {bbox.x, bbox.y, bbox.w, bbox.h};
}

} // namespace

IouTracker::IouTracker(const YAML::Node &config) { loadParameters(config); }

void IouTracker::loadParameters(const YAML::Node &config) {
  // 保留已有 YAML 键名及默认值，部署配置无需迁移。
  const YAML::Node tracker_config = config["iou_tracker"];
  enabled_ = tracker_config["enable"].as<bool>(true);
  match_iou_threshold_ =
      std::clamp(tracker_config["iou_threshold"].as<float>(0.30f), 0.01f, 1.0f);
  max_missed_frames_ =
      std::max(1, tracker_config["max_age_frames"].as<int>(30));
  recovery_window_seconds_ =
      std::max(0, tracker_config["reid_window_seconds"].as<int>(30));
  recovery_iou_scale_ = std::clamp(
      tracker_config["revive_iou_scale"].as<float>(0.40f), 0.05f, 1.0f);
}

float IouTracker::Track::computeIou(const cv::Rect &other) const {
  const cv::Rect intersection = last_body_bbox & other;
  const float intersection_area = static_cast<float>(intersection.area());
  if (intersection_area <= 0.0f) {
    return 0.0f;
  }
  const float union_area =
      static_cast<float>(last_body_bbox.area() + other.area()) -
      intersection_area;
  return union_area > 1e-6f ? intersection_area / union_area : 0.0f;
}

void IouTracker::Track::updateFromDetection(const cv::Rect &body_bbox) {
  last_body_bbox = body_bbox;
  missed_frames = 0;
  ++matched_frames;
  is_inactive = false;
}

void IouTracker::incrementMissedFrames() {
  for (auto &[track_id, track] : tracks_by_id_) {
    (void)track_id;
    if (!track.is_inactive) {
      ++track.missed_frames;
    }
  }
}

IouTracker::Track *
IouTracker::findRecoveryCandidate(const cv::Rect &detection) {
  const auto now = std::chrono::steady_clock::now();
  Track *best_track = nullptr;
  float best_iou = match_iou_threshold_ * recovery_iou_scale_;

  for (auto &[track_id, track] : tracks_by_id_) {
    (void)track_id;
    if (!track.is_inactive) {
      continue;
    }
    const auto inactive_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(now -
                                                         track.inactive_since)
            .count();
    if (inactive_seconds > recovery_window_seconds_) {
      continue;
    }
    const float iou = track.computeIou(detection);
    if (iou > best_iou) {
      best_iou = iou;
      best_track = &track;
    }
  }
  return best_track;
}

void IouTracker::retireAndRemoveExpiredTracks() {
  const auto now = std::chrono::steady_clock::now();
  for (auto &[track_id, track] : tracks_by_id_) {
    (void)track_id;
    if (!track.is_inactive && track.missed_frames > max_missed_frames_) {
      track.is_inactive = true;
      track.inactive_since = now;
    }
  }

  for (auto it = tracks_by_id_.begin(); it != tracks_by_id_.end();) {
    if (it->second.is_inactive) {
      const auto inactive_seconds =
          std::chrono::duration_cast<std::chrono::seconds>(
              now - it->second.inactive_since)
              .count();
      if (inactive_seconds > recovery_window_seconds_) {
        it = tracks_by_id_.erase(it);
        continue;
      }
    }
    ++it;
  }
}

std::set<int> IouTracker::collectRetainedTrackIds() const {
  std::set<int> ids;
  for (const auto &[track_id, track] : tracks_by_id_) {
    (void)track;
    ids.insert(track_id);
  }
  return ids;
}

std::vector<IouTracker::Track *>
IouTracker::matchActiveTracks(const std::vector<cv::Rect> &body_boxes) {
  // 固定本阶段的活跃轨迹集合；新建和恢复在下一阶段进行。
  std::vector<int> active_track_ids;
  active_track_ids.reserve(tracks_by_id_.size());
  for (const auto &[track_id, track] : tracks_by_id_) {
    if (!track.is_inactive) {
      active_track_ids.push_back(track_id);
    }
  }

  struct MatchCandidate {
    std::size_t detection_index;
    int track_id;
    float iou;
  };
  std::vector<MatchCandidate> candidates;
  for (std::size_t detection_index = 0; detection_index < body_boxes.size();
       ++detection_index) {
    if (body_boxes[detection_index].empty()) {
      continue;
    }
    for (const int track_id : active_track_ids) {
      const float iou =
          tracks_by_id_.at(track_id).computeIou(body_boxes[detection_index]);
      if (iou >= match_iou_threshold_) {
        candidates.push_back({detection_index, track_id, iou});
      }
    }
  }
  // 保留原来的遍历和排序规则；相同 IoU 不增加新的优先级。
  std::sort(candidates.begin(), candidates.end(),
            [](const MatchCandidate &left, const MatchCandidate &right) {
              return left.iou > right.iou;
            });

  std::set<int> used_track_ids;
  std::vector<Track *> matched_tracks(body_boxes.size(), nullptr);

  // 每个检测和每条轨迹至多使用一次。空指针同时表示该检测尚未匹配。
  for (const MatchCandidate &match : candidates) {
    if (matched_tracks[match.detection_index] != nullptr ||
        used_track_ids.count(match.track_id) != 0U) {
      continue;
    }
    Track &track = tracks_by_id_.at(match.track_id);
    track.updateFromDetection(body_boxes[match.detection_index]);
    used_track_ids.insert(match.track_id);
    matched_tracks[match.detection_index] = &track;
  }

  return matched_tracks;
}

void IouTracker::recoverOrCreateTracks(const std::vector<cv::Rect> &body_boxes,
                                       std::vector<Track *> &matched_tracks) {
  for (std::size_t detection_index = 0; detection_index < body_boxes.size();
       ++detection_index) {
    if (matched_tracks[detection_index] != nullptr ||
        body_boxes[detection_index].empty()) {
      continue;
    }

    Track *track = findRecoveryCandidate(body_boxes[detection_index]);
    if (track != nullptr) {
      // 立即重新激活，后面的检测就不能再次恢复同一条轨迹。
      track->updateFromDetection(body_boxes[detection_index]);
    } else {
      Track new_track;
      new_track.track_id = next_track_id_++;
      new_track.updateFromDetection(body_boxes[detection_index]);
      const int track_id = new_track.track_id;
      tracks_by_id_.emplace(track_id, std::move(new_track));
      track = &tracks_by_id_.at(track_id);
    }
    matched_tracks[detection_index] = track;
  }
}

void IouTracker::process(
    trt_infer_msgs::msg::PerceptionResult &perception_result,
    PerceptionFrameContext &frame_context) {
  // 1. 保持消息与上下文索引一致，清除上一帧的人员中间信息。
  frame_context.persons.assign(perception_result.persons.size(), {});

  if (!enabled_) {
    for (auto &person : perception_result.persons) {
      person.track_id = -1;
    }
    frame_context.retained_track_ids.clear();
    return;
  }

  // 2. 先累计失配帧数，后续匹配成功时清零；无人帧也要推进生命周期。
  incrementMissedFrames();

  std::vector<cv::Rect> body_boxes;
  body_boxes.reserve(perception_result.persons.size());
  for (const auto &person : perception_result.persons) {
    body_boxes.push_back(toValidRect(person.body_detection.body_bbox));
  }

  // 3. 先匹配活跃轨迹，再为剩余检测恢复旧轨迹或分配新 ID。
  auto matched_tracks = matchActiveTracks(body_boxes);
  recoverOrCreateTracks(body_boxes, matched_tracks);

  // 4. 必须在匹配之后退休：本帧重新匹配成功的轨迹不应被提前判死。
  retireAndRemoveExpiredTracks();
  frame_context.retained_track_ids = collectRetainedTrackIds();

  // 5. 仅回写本帧检测到的人员，不为暂时丢失的轨迹补框。
  // std::map 插入不使指针失效，清理阶段也不会删除本帧已匹配的轨迹。
  for (std::size_t index = 0; index < perception_result.persons.size();
       ++index) {
    auto &person = perception_result.persons[index];
    auto &person_context = frame_context.persons[index];
    const Track *track = matched_tracks[index];
    if (track == nullptr) {
      person.track_id = -1;
      continue;
    }
    person.track_id = track->track_id;
    person_context.track_id = track->track_id;
    person_context.track_total_frames = track->matched_frames;
  }
}

int IouTracker::activeTrackCount() const {
  return static_cast<int>(std::count_if(
      tracks_by_id_.begin(), tracks_by_id_.end(),
      [](const auto &entry) { return !entry.second.is_inactive; }));
}
