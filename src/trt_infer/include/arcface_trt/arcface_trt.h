#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "face_detector.h" // FaceObject with landmark[5]

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/** 512-dim L2-normalised face embedding. */
struct FaceEmbedding {
  static constexpr int DIM = 512;
  float v[DIM];
};

// ─────────────────────────────────────────────────────────────────────────────
// ArcFaceTRT
//
// Pure TRT inference wrapper.  Gallery management is handled by FaceDatabase.
//
// Workflow:
//   1. Construct with a .trt engine path
//   2. Per-frame: alignFace() → extractEmbedding()
//   3. Use FaceDatabase::identify() for recognition
//
// Thread-safety: NOT thread-safe. Use one instance per thread.
// ─────────────────────────────────────────────────────────────────────────────
class ArcFaceTRT {
public:
  explicit ArcFaceTRT(const std::string &engine_path,
                      float match_threshold = 0.40f);
  ~ArcFaceTRT();

  ArcFaceTRT(const ArcFaceTRT &) = delete;
  ArcFaceTRT &operator=(const ArcFaceTRT &) = delete;

  // ── Face alignment ────────────────────────────────────────────────────────
  /** Warp `bgr` into a 112×112 aligned face image using the 5 SCRFD landmarks.
   *  Uses the InsightFace/ArcFace standard template.  Returns empty Mat on
   * failure. */
  static cv::Mat alignFace(const cv::Mat &bgr, const cv::Point2f landmark[5]);

  // ── Embedding extraction ──────────────────────────────────────────────────
  /** Run TRT inference on a 112×112 BGR aligned face.
   *  Fills `out_emb` (512 floats, L2-normalised).
   *  Returns false if the engine is not loaded or `aligned` is empty. */
  bool extractEmbedding(const cv::Mat &aligned_112,
                        FaceEmbedding &out_emb) const;

  // ── Utility ────────────────────────────────────────────────────────────────
  /** L2-normalise a finite, non-zero 512-dim vector in-place. */
  static bool l2Normalize(FaceEmbedding &emb);

  /** Return true only when all 512 values are finite. */
  static bool isFiniteEmbedding(const FaceEmbedding &emb);

  /** Cosine similarity (dot-product of two L2-normalised vectors). */
  static float cosineSim(const FaceEmbedding &a, const FaceEmbedding &b);

  bool isReady() const { return engine_ != nullptr; }
  float threshold() const { return match_threshold_; }
  void setThreshold(float t) { match_threshold_ = t; }

private:
  void initEngine(const std::string &path);

  nvinfer1::IRuntime *runtime_{nullptr};
  nvinfer1::ICudaEngine *engine_{nullptr};
  nvinfer1::IExecutionContext *context_{nullptr};

  void *d_input_{nullptr};
  void *d_output_{nullptr};

  static constexpr int kInputSize = 1 * 3 * 112 * 112;
  static constexpr int kOutputSize = 512;

  mutable float h_input_[kInputSize];
  mutable float h_output_[kOutputSize];
  mutable std::vector<__half> h_input_half_;
  mutable std::vector<__half> h_output_half_;

  float match_threshold_{0.40f};

  bool inputShapeDynamic_{false};
  std::string inputTensorName_;
  std::string outputTensorName_;
  nvinfer1::DataType inputDataType_{nvinfer1::DataType::kFLOAT};
  nvinfer1::DataType outputDataType_{nvinfer1::DataType::kFLOAT};
  std::size_t inputBytes_{0};
  std::size_t outputBytes_{0};

  struct Logger : nvinfer1::ILogger {
    void log(Severity s, const char *msg) noexcept override {
      if (s <= Severity::kWARNING)
        std::fprintf(stderr, "[ArcFace_TRT] %s\n", msg);
    }
  } logger_;
};
