#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <NvInfer.h>
#include "yolo_gpu_preprocess.h"

struct Detection {
    int class_id;
    float conf;
    int x, y, w, h;
    float distance = -1.0f;
};

class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class YOLOEngine {
public:
    YOLOEngine(const std::string& enginePath, float confThreshold, float nmsThreshold);
    YOLOEngine(const std::string& enginePath, float confThreshold, bool bboxInOriginalSpace, int engineInH,
               int engineInW);
    ~YOLOEngine();

    std::vector<Detection> infer(const cv::Mat& img, float conf);

    /** 全 GPU 链路：图像预处理 + 推理 + 后处理 + 深度图采样 */
    std::vector<Detection> inferWithDepth(
        const cv::Mat& img, const cv::Mat& depth_m, float conf,
        float min_depth, float max_depth,
        float roi_y0_n, float roi_y1_n, float x_margin_n,
        float trim_close_ratio, float percentile);

    float getLastPreprocessMs() const { return lastPreprocessMs_; }
    float getLastInferMs() const { return lastInferMs_; }
    float getLastPostprocessMs() const { return lastPostprocessMs_; }
    int getInputH() const { return inputH_; }
    int getInputW() const { return inputW_; }

private:
    TensorRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    int inputH_ = 640, inputW_ = 640;
    size_t inputSize_ = 0;
    size_t outputNumFloats_ = 0;
    int inputIndex_ = 0, outputIndex_ = 0;
    std::vector<void*> buffers_;
    std::vector<float> outputHost_;
    int numClasses_ = 0;
    int numAnchors_ = 0;
    bool outputChannelsFirst_ = true;
    bool isYOLO26_ = false;

    float confThreshold_ = 0.45f;
    float nmsThreshold_ = 0.5f;
    float lastPreprocessMs_ = 0.f;
    float lastInferMs_ = 0.f;
    float lastPostprocessMs_ = 0.f;

    void* d_input_bgr_u8_ = nullptr;
    size_t d_input_bgr_cap_ = 0;
    
    // New GPU buffers for acceleration
    float* d_depth_f32_ = nullptr;
    size_t d_depth_cap_ = 0;
    YoloProposal* d_proposals_ = nullptr;
    int* d_proposals_count_ = nullptr;
    const int MAX_PROPOSALS = 100;

    cudaStream_t preprocessStream_ = nullptr;

    bool bbox_in_original_space_{false};
    int engine_override_h_{0};
    int engine_override_w_{0};

    void ensureDeviceInputBgrCapacity(size_t bytes);
    void ensureDeviceDepthCapacity(size_t bytes);
    void loadEngine(const std::string& path);
    void maybeApplyEngineInputOverride();
    std::vector<Detection> postprocess(int origW, int origH, float scale, int padLeft, int padTop);
    std::vector<Detection> postprocessYOLO26(int origW, int origH, float scale, int padLeft, int padTop);
    std::vector<Detection> postprocessYOLOv8(int origW, int origH, float scale, int padLeft, int padTop);
};
