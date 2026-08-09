#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// GenderResult
// ─────────────────────────────────────────────────────────────────────────────
struct GenderResult {
    static constexpr uint8_t UNKNOWN = 0;
    static constexpr uint8_t MALE    = 1;
    static constexpr uint8_t FEMALE  = 2;

    uint8_t gender{UNKNOWN};  // UNKNOWN / MALE / FEMALE
    float   conf{0.f};        // confidence ∈ [0,1]
    float   age{-1.f};        // age in years; -1 = not available
};

// ─────────────────────────────────────────────────────────────────────────────
// GenderAgeTRT
//
// Wraps the InsightFace buffalo_l genderage TensorRT engine.
//
// Input  : aligned face image (BGR), resized to engine input size (96×96 typical).
//          Pass the same cv::Mat returned by ArcFaceTRT::alignFace() — no extra
//          alignment needed, just resize if the engine uses a different size.
// Output : GenderResult { gender, conf, age }
//
// Engine output convention (auto-detected by tensor name + size):
//   size == 3  → [female_logit, male_logit, age_norm]
//                gender = argmax(logits[0:2]); age = logits[2] * 100
//   size == 2  → [gender_logit, age_norm]
//                或 [female_logit, male_logit]（年龄由独立 size==1 输出）
//   size == 1  → [gender_logit]
//                gender = (logit > 0 → MALE)
//
// Thread-safety: NOT thread-safe.  Use one instance per thread.
// ─────────────────────────────────────────────────────────────────────────────
class GenderAgeTRT {
public:
    explicit GenderAgeTRT(const std::string& engine_path);
    ~GenderAgeTRT();

    GenderAgeTRT(const GenderAgeTRT&)            = delete;
    GenderAgeTRT& operator=(const GenderAgeTRT&) = delete;

    /** Predict gender and age from an aligned BGR face image.
     *  Internally resizes to engine input resolution if needed.
     *  Returns { UNKNOWN, 0, -1 } on failure. */
    GenderResult predict(const cv::Mat& aligned_bgr) const;

    bool isReady()     const { return engine_ != nullptr; }
    int  inputWidth()  const { return input_w_; }
    int  inputHeight() const { return input_h_; }

private:
    void initEngine(const std::string& path);
    static int tensorVolumeNoBatch(const nvinfer1::Dims& dims);

    nvinfer1::IRuntime*          runtime_{nullptr};
    nvinfer1::ICudaEngine*       engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};

    void* d_input_{nullptr};

    int input_w_{96};
    int input_h_{96};

    mutable std::vector<float> h_input_;
    mutable bool               debug_printed_{false};

    struct OutputTensor {
        std::string name;
        nvinfer1::Dims dims{};
        int volume{0};   // without batch dim
        void* d_ptr{nullptr};
        mutable std::vector<float> h_data;
    };
    mutable std::vector<OutputTensor> outputs_;

    bool        inputBatchDynamic_{false};
    std::string inputTensorName_;

    struct Logger : nvinfer1::ILogger {
        void log(Severity s, const char* msg) noexcept override {
            if (s <= Severity::kWARNING)
                std::fprintf(stderr, "[GenderAge_TRT] %s\n", msg);
        }
    } logger_;
};
