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
  const YAML::Node tracker_config = config["iou_tracker"];
  enabled_ = tracker_config["enable"].as<bool>(true);
  iou_threshold_ =
      std::clamp(tracker_config["iou_threshold"].as<float>(0.30f), 0.01f, 1.0f);
  max_age_frames_ = std::max(1, tracker_config["max_age_frames"].as<int>(30));
  reid_window_seconds_ =
      std::max(0, tracker_config["reid_window_seconds"].as<int>(30));
  revive_iou_scale_ = std::clamp(
      tracker_config["revive_iou_scale"].as<float>(0.40f), 0.05f, 1.0f);
}

float IouTracker::Track::iouWith(const cv::Rect &other) const {
  const cv::Rect intersection = body_bbox & other;
  const float intersection_area = static_cast<float>(intersection.area());
  if (intersection_area <= 0.0f) {
    return 0.0f;
  }
  const float union_area =
      static_cast<float>(body_bbox.area() + other.area()) - intersection_area;
  return union_area > 1e-6f ? intersection_area / union_area : 0.0f;
}

void IouTracker::ageTracks() {
  for (auto &[track_id, track] : tracks_) {
    (void)track_id;
    if (!track.is_dead) {
      ++track.age_frames;
    }
  }
}

IouTracker::Track *IouTracker::tryRevive(const cv::Rect &detection) {
  const auto now = std::chrono::steady_clock::now();
  Track *best_track = nullptr;
  float best_iou = iou_threshold_ * revive_iou_scale_;

  for (auto &[track_id, track] : tracks_) {
    (void)track_id;
    if (!track.is_dead) {
      continue;
    }
    const auto dead_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(now - track.dead_since)
            .count();
    if (dead_seconds > reid_window_seconds_) {
      continue;
    }
    const float iou = track.iouWith(detection);
    if (iou > best_iou) {
      best_iou = iou;
      best_track = &track;
    }
  }
  return best_track;
}

void IouTracker::retireAndPurgeTracks() {
  const auto now = std::chrono::steady_clock::now();
  for (auto &[track_id, track] : tracks_) {
    (void)track_id;
    if (!track.is_dead && track.age_frames > max_age_frames_) {
      track.is_dead = true;
      track.dead_since = now;
    }
  }

  for (auto it = tracks_.begin(); it != tracks_.end();) {
    if (it->second.is_dead) {
      const auto dead_seconds =
          std::chrono::duration_cast<std::chrono::seconds>(
              now - it->second.dead_since)
              .count();
      if (dead_seconds > reid_window_seconds_) {
        it = tracks_.erase(it);
        continue;
      }
    }
    ++it;
  }
}

std::set<int> IouTracker::retainedTrackIds() const {
  std::set<int> ids;
  for (const auto &[track_id, track] : tracks_) {
    (void)track;
    ids.insert(track_id);
  }
  return ids;
}

void IouTracker::process(
    trt_infer_msgs::msg::PerceptionResult &perception_result,
    PerceptionFrameContext &frame_context) {
  frame_context.persons.assign(perception_result.persons.size(), {});

  if (!enabled_) {
    for (auto &person : perception_result.persons) {
      person.track_id = -1;
    }
    frame_context.retained_track_ids.clear();
    return;
  }

  ageTracks();

  std::vector<cv::Rect> detections;
  detections.reserve(perception_result.persons.size());
  for (const auto &person : perception_result.persons) {
    detections.push_back(toValidRect(person.body_detection.body_bbox));
  }

  std::vector<int> live_track_ids;
  live_track_ids.reserve(tracks_.size());
  for (const auto &[track_id, track] : tracks_) {
    if (!track.is_dead) {
      live_track_ids.push_back(track_id);
    }
  }

  struct Match {
    std::size_t detection_index;
    int track_id;
    float iou;
  };
  std::vector<Match> candidates;
  for (std::size_t detection_index = 0; detection_index < detections.size();
       ++detection_index) {
    if (detections[detection_index].empty()) {
      continue;
    }
    for (const int track_id : live_track_ids) {
      const float iou =
          tracks_.at(track_id).iouWith(detections[detection_index]);
      if (iou >= iou_threshold_) {
        candidates.push_back({detection_index, track_id, iou});
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Match &left, const Match &right) {
              return left.iou > right.iou;
            });

  std::vector<bool> detection_matched(detections.size(), false);
  std::set<int> used_track_ids;
  std::vector<Track *> matched_tracks(detections.size(), nullptr);

  for (const Match &match : candidates) {
    if (detection_matched[match.detection_index] ||
        used_track_ids.count(match.track_id) != 0U) {
      continue;
    }
    Track &track = tracks_.at(match.track_id);
    track.body_bbox = detections[match.detection_index];
    track.age_frames = 0;
    ++track.total_frames;
    detection_matched[match.detection_index] = true;
    used_track_ids.insert(match.track_id);
    matched_tracks[match.detection_index] = &track;
  }

  for (std::size_t detection_index = 0; detection_index < detections.size();
       ++detection_index) {
    if (detection_matched[detection_index] ||
        detections[detection_index].empty()) {
      continue;
    }

    Track *track = tryRevive(detections[detection_index]);
    if (track != nullptr) {
      track->body_bbox = detections[detection_index];
      track->age_frames = 0;
      track->is_dead = false;
      ++track->total_frames;
    } else {
      Track new_track;
      new_track.track_id = next_id_++;
      new_track.body_bbox = detections[detection_index];
      new_track.total_frames = 1;
      const int track_id = new_track.track_id;
      tracks_.emplace(track_id, std::move(new_track));
      track = &tracks_.at(track_id);
    }
    matched_tracks[detection_index] = track;
  }

  retireAndPurgeTracks();
  frame_context.retained_track_ids = retainedTrackIds();

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
    person_context.track_total_frames = track->total_frames;
  }
}

int IouTracker::liveCount() const {
  return static_cast<int>(
      std::count_if(tracks_.begin(), tracks_.end(),
                    [](const auto &entry) { return !entry.second.is_dead; }));
}
