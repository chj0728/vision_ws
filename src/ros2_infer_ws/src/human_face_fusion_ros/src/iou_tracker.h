#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <set>
#include <vector>

#include <opencv2/core.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// TrackRecogState — per-track identity state machine
// ─────────────────────────────────────────────────────────────────────────────
enum class TrackRecogState {
    PENDING,     // waiting for quality gate + embedding accumulation
    IDENTIFIED,  // has a person_uuid (either from DB match or auto-registered)
};

// ─────────────────────────────────────────────────────────────────────────────
// FaceTrack — state for one tracked person
// ─────────────────────────────────────────────────────────────────────────────
struct FaceTrack {
    static constexpr int  kEmbDim          = 512;
    static constexpr int  kRegBufferTarget  = 5;    // embeddings to collect before recog
    static constexpr int  kReIdWindowSec    = 30;   // dead-track re-id window (seconds)

    int      track_id{-1};
    cv::Rect body_bbox;
    int      age_frames{0};   // frames since last matched detection
    int      total_frames{0}; // total frames this track has been active

    // ── Identity (persists across frames) ────────────────────────────────────
    TrackRecogState recog_state{TrackRecogState::PENDING};
    std::string     person_uuid;
    std::string     person_name;
    float           recog_conf{0.f};
    int             last_recog_frame{-1};

    // Accumulator: raw embeddings collected while PENDING (before first identify)
    std::vector<std::array<float, kEmbDim>> reg_emb_buffer;
    std::vector<float> reg_yaw_buffer;
    std::vector<float> reg_pitch_buffer;
    std::vector<float> reg_conf_buffer;

    // ── Gender (多帧投票，质量门控后决策) ───────────────────────────────────
    uint8_t gender{0};        // 0=UNKNOWN, 1=MALE, 2=FEMALE
    float   gender_conf{0.f};
    bool    gender_done{false};
    int     gender_votes_male{0};
    int     gender_votes_female{0};
    int     gender_last_eval_frame{-1};
    static constexpr int kGenderVotesNeeded = 5;  // 至少 5 票才确认

    // ── Dead-track metadata (for re-id after brief disappearance) ─────────────
    bool is_dead{false};
    std::chrono::steady_clock::time_point dead_since;

    // ─────────────────────────────────────────────────────────────────────────
    float iouWith(const cv::Rect& other) const {
        const cv::Rect inter = body_bbox & other;
        const float ia = static_cast<float>(inter.area());
        if (ia <= 0.f) return 0.f;
        const float ua = static_cast<float>(body_bbox.area() + other.area()) - ia;
        return ua > 1e-6f ? ia / ua : 0.f;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// IouTracker
//
// Greedy IoU-based multi-person body tracker.  No external dependencies.
//
// update() returns a vector of FaceTrack* in the SAME ORDER as the input
// detections array.  result[i] is the track associated with detections[i].
//
// Dead tracks are kept in the map for kReIdWindowSec seconds to enable
// position-based re-identification (person briefly leaves frame and returns).
// ─────────────────────────────────────────────────────────────────────────────
class IouTracker {
public:
    explicit IouTracker(float iou_threshold = 0.30f, int max_age_frames = 30)
        : iou_threshold_(iou_threshold), max_age_frames_(max_age_frames) {}

    /** Update tracker with new body detections for this frame.
     *  Returns vector of FaceTrack* in same order as detections[].
     *  Pointer is valid until the next call to update(). */
    std::vector<FaceTrack*> update(const std::vector<cv::Rect>& detections,
                                   int frame_number) {
        // Age all live tracks
        for (auto& [id, t] : tracks_)
            if (!t.is_dead) t.age_frames++;

        // Build list of live track IDs
        std::vector<int> live_ids;
        live_ids.reserve(tracks_.size());
        for (auto& [id, t] : tracks_)
            if (!t.is_dead) live_ids.push_back(id);

        // Build candidate matches sorted by IoU descending
        struct Match { int det_idx; int trk_id; float iou; };
        std::vector<Match> candidates;
        for (int di = 0; di < static_cast<int>(detections.size()); ++di) {
            for (int ti : live_ids) {
                const float iou = tracks_[ti].iouWith(detections[static_cast<size_t>(di)]);
                if (iou >= iou_threshold_) candidates.push_back({di, ti, iou});
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Match& a, const Match& b) { return a.iou > b.iou; });

        // Greedy assignment
        std::vector<bool> det_matched(detections.size(), false);
        std::set<int> used_trk;
        // result[i] → pointer into tracks_ for detections[i]
        std::vector<FaceTrack*> result(detections.size(), nullptr);

        for (const auto& m : candidates) {
            if (det_matched[static_cast<size_t>(m.det_idx)]) continue;
            if (used_trk.count(m.trk_id)) continue;
            det_matched[static_cast<size_t>(m.det_idx)] = true;
            used_trk.insert(m.trk_id);
            FaceTrack& t = tracks_[m.trk_id];
            t.body_bbox   = detections[static_cast<size_t>(m.det_idx)];
            t.age_frames  = 0;
            t.total_frames++;
            result[static_cast<size_t>(m.det_idx)] = &t;
        }

        // Unmatched detections → try reviving a dead track, else create new
        for (int di = 0; di < static_cast<int>(detections.size()); ++di) {
            if (det_matched[static_cast<size_t>(di)]) continue;
            const cv::Rect& det = detections[static_cast<size_t>(di)];

            FaceTrack* revived = tryRevive(det);
            if (revived) {
                revived->body_bbox   = det;
                revived->age_frames  = 0;
                revived->is_dead     = false;
                revived->total_frames++;
                result[static_cast<size_t>(di)] = revived;
            } else {
                FaceTrack t;
                t.track_id    = next_id_++;
                t.body_bbox   = det;
                t.age_frames  = 0;
                t.total_frames = 1;
                const int id  = t.track_id;
                tracks_[id]   = std::move(t);
                result[static_cast<size_t>(di)] = &tracks_[id];
            }
        }

        // Kill stale tracks
        const auto now = std::chrono::steady_clock::now();
        for (auto& [id, t] : tracks_) {
            if (!t.is_dead && t.age_frames > max_age_frames_) {
                t.is_dead   = true;
                t.dead_since = now;
            }
        }

        // Purge expired dead tracks
        for (auto it = tracks_.begin(); it != tracks_.end(); ) {
            if (it->second.is_dead) {
                const auto age_s = std::chrono::duration_cast<std::chrono::seconds>(
                    now - it->second.dead_since).count();
                if (age_s > FaceTrack::kReIdWindowSec) {
                    it = tracks_.erase(it);
                    continue;
                }
            }
            ++it;
        }

        (void)frame_number;
        return result;
    }

    FaceTrack* getTrack(int track_id) {
        auto it = tracks_.find(track_id);
        return (it != tracks_.end() && !it->second.is_dead) ? &it->second : nullptr;
    }

    /** Update person_name for all tracks (live or dead) matching the given UUID.
     *  Returns the number of tracks updated. */
    int updatePersonNameByUuid(const std::string& uuid, const std::string& name) {
        int count = 0;
        for (auto& [id, t] : tracks_) {
            if (t.person_uuid == uuid) {
                t.person_name = name;
                ++count;
            }
        }
        return count;
    }

    int liveCount() const {
        int n = 0;
        for (const auto& [id, t] : tracks_) if (!t.is_dead) ++n;
        return n;
    }

private:
    float iou_threshold_{0.30f};
    int   max_age_frames_{30};
    int   next_id_{0};
    std::map<int, FaceTrack> tracks_;

    FaceTrack* tryRevive(const cv::Rect& det) {
        const auto now = std::chrono::steady_clock::now();
        FaceTrack* best = nullptr;
        float best_iou = iou_threshold_ * 0.4f;  // looser threshold for revival
        for (auto& [id, t] : tracks_) {
            if (!t.is_dead) continue;
            const auto age_s = std::chrono::duration_cast<std::chrono::seconds>(
                now - t.dead_since).count();
            if (age_s > FaceTrack::kReIdWindowSec) continue;
            const float iou = t.iouWith(det);
            if (iou > best_iou) {
                best_iou = iou;
                best = &t;
            }
        }
        return best;
    }
};
