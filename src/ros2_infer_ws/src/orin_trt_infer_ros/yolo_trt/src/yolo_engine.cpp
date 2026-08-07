#include "yolo_engine.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <cuda_runtime_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "yolo_gpu_preprocess.h"

void TensorRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << msg << std::endl;
}

YOLOEngine::YOLOEngine(const std::string& enginePath, float confThreshold, float nmsThreshold)
    : confThreshold_(confThreshold), nmsThreshold_(nmsThreshold) {
    loadEngine(enginePath);
}

YOLOEngine::YOLOEngine(const std::string& enginePath, float confThreshold, bool bboxInOriginalSpace,
                       int engineInH, int engineInW)
    : confThreshold_(confThreshold),
      nmsThreshold_(0.5f),
      bbox_in_original_space_(bboxInOriginalSpace),
      engine_override_h_(engineInH),
      engine_override_w_(engineInW) {
    loadEngine(enginePath);
}

YOLOEngine::~YOLOEngine() {
    cudaFree(d_input_bgr_u8_);
    cudaFree(d_depth_f32_);
    cudaFree(d_proposals_);
    cudaFree(d_proposals_count_);
    if (preprocessStream_) cudaStreamDestroy(preprocessStream_);
}

std::vector<Detection> YOLOEngine::infer(const cv::Mat& img, float conf) {
    if (img.empty() || img.cols <= 0 || img.rows <= 0) return {};
    if (img.type() != CV_8UC3) return {};
    confThreshold_ = conf;

    int origH = img.rows, origW = img.cols;
    int h = origH, w = origW;

    auto tPre0 = std::chrono::high_resolution_clock::now();

    cv::Mat packed;
    const uint8_t* h_bgr;
    size_t nbytes;
    if (!img.isContinuous()) {
        img.copyTo(packed);
        if (packed.type() != CV_8UC3 || !packed.isContinuous()) return {};
        h_bgr = packed.ptr<uint8_t>();
        nbytes = static_cast<size_t>(packed.total()) * 3u;
    } else {
        h_bgr = img.ptr<uint8_t>();
        nbytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
    }
    ensureDeviceInputBgrCapacity(nbytes);
    cudaMemcpyAsync(d_input_bgr_u8_, h_bgr, nbytes, cudaMemcpyHostToDevice, preprocessStream_);

    auto* d_input = static_cast<float*>(buffers_[inputIndex_]);
    float scale;
    int padLeft, padTop;

    if (inputW_ != inputH_) {
        scale = 1.f;
        padLeft = 0;
        padTop = 0;
        yolo_launch_resize_to_nchw(static_cast<const uint8_t*>(d_input_bgr_u8_), w, h, d_input, inputW_, inputH_,
                                   preprocessStream_);
    } else {
        scale = std::min(static_cast<float>(inputW_) / w, static_cast<float>(inputH_) / h);
        int newW = static_cast<int>(w * scale);
        int newH = static_cast<int>(h * scale);
        padLeft = (inputW_ - newW) / 2;
        padTop = (inputH_ - newH) / 2;
        yolo_launch_letterbox_to_nchw(static_cast<const uint8_t*>(d_input_bgr_u8_), w, h, d_input, inputW_, inputH_,
                                      newW, newH, padLeft, padTop, preprocessStream_);
    }

    cudaStreamSynchronize(preprocessStream_);

    auto tPre1 = std::chrono::high_resolution_clock::now();
    lastPreprocessMs_ = std::chrono::duration<float, std::milli>(tPre1 - tPre0).count();

    auto tInf0 = std::chrono::high_resolution_clock::now();
    if (!context_->enqueueV3(preprocessStream_)) {
        std::cerr << "TensorRT enqueueV3 failed" << std::endl;
        return {};
    }
    cudaStreamSynchronize(preprocessStream_);
    auto tInf1 = std::chrono::high_resolution_clock::now();
    lastInferMs_ = std::chrono::duration<float, std::milli>(tInf1 - tInf0).count();

    cudaMemcpy(outputHost_.data(), buffers_[outputIndex_], outputNumFloats_ * sizeof(float), cudaMemcpyDeviceToHost);

    auto tPost0 = std::chrono::high_resolution_clock::now();
    std::vector<Detection> dets = postprocess(origW, origH, scale, padLeft, padTop);
    auto tPost1 = std::chrono::high_resolution_clock::now();
    lastPostprocessMs_ = std::chrono::duration<float, std::milli>(tPost1 - tPost0).count();

    return dets;
}

std::vector<Detection> YOLOEngine::inferWithDepth(
    const cv::Mat& img, const cv::Mat& depth_m, float conf,
    float min_depth, float max_depth,
    float roi_y0_n, float roi_y1_n, float x_margin_n,
    float trim_close_ratio, float percentile) {
    
    if (img.empty() || depth_m.empty()) return {};
    confThreshold_ = conf;

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Copy Image and Depth to GPU
    size_t img_bytes = (size_t)img.total() * img.elemSize();
    ensureDeviceInputBgrCapacity(img_bytes);
    cudaMemcpyAsync(d_input_bgr_u8_, img.data, img_bytes, cudaMemcpyHostToDevice, preprocessStream_);

    size_t depth_bytes = (size_t)depth_m.total() * depth_m.elemSize();
    ensureDeviceDepthCapacity(depth_bytes);
    cudaMemcpyAsync(d_depth_f32_, depth_m.data, depth_bytes, cudaMemcpyHostToDevice, preprocessStream_);

    // 2. Preprocess Image
    int w = img.cols, h = img.rows;
    float scale;
    int padLeft, padTop;
    auto* d_input = static_cast<float*>(buffers_[inputIndex_]);

    if (inputW_ != inputH_) {
        scale = 1.f; padLeft = 0; padTop = 0;
        yolo_launch_resize_to_nchw(static_cast<const uint8_t*>(d_input_bgr_u8_), w, h, d_input, inputW_, inputH_, preprocessStream_);
    } else {
        scale = std::min((float)inputW_ / w, (float)inputH_ / h);
        int newW = (int)(w * scale), newH = (int)(h * scale);
        padLeft = (inputW_ - newW) / 2; padTop = (inputH_ - newH) / 2;
        yolo_launch_letterbox_to_nchw(static_cast<const uint8_t*>(d_input_bgr_u8_), w, h, d_input, inputW_, inputH_, newW, newH, padLeft, padTop, preprocessStream_);
    }

    // 3. Inference
    context_->enqueueV3(preprocessStream_);

    // 4. Postprocess on GPU
    if (!d_proposals_) {
        cudaMalloc(&d_proposals_, MAX_PROPOSALS * sizeof(YoloProposal));
        cudaMalloc(&d_proposals_count_, sizeof(int));
    }

    yolo_postprocess_gpu(
        static_cast<const float*>(buffers_[outputIndex_]), numAnchors_, numClasses_, confThreshold_,
        isYOLO26_, outputChannelsFirst_, w, h, scale, padLeft, padTop,
        d_proposals_, d_proposals_count_, MAX_PROPOSALS, preprocessStream_);

    // 5. 取回实际 proposal 数量后再做深度采样，避免用 MAX_PROPOSALS 启动多余线程块
    int h_count = 0;
    cudaMemcpyAsync(&h_count, d_proposals_count_, sizeof(int), cudaMemcpyDeviceToHost, preprocessStream_);
    cudaStreamSynchronize(preprocessStream_);
    h_count = std::min(h_count, MAX_PROPOSALS);

    // 深度图与彩色图分辨率可能不同，计算缩放系数将 bbox 坐标从彩色图空间映射到深度图空间
    const float depth_scale_x = (w > 0) ? static_cast<float>(depth_m.cols) / w : 1.0f;
    const float depth_scale_y = (h > 0) ? static_cast<float>(depth_m.rows) / h : 1.0f;

    yolo_depth_sampling_gpu(
        d_depth_f32_, depth_m.cols, depth_m.rows,
        d_proposals_, h_count,
        min_depth, max_depth, roi_y0_n, roi_y1_n, x_margin_n, trim_close_ratio, percentile,
        depth_scale_x, depth_scale_y, preprocessStream_);
    cudaStreamSynchronize(preprocessStream_);

    // 6. Copy back results

    std::vector<YoloProposal> h_proposals(h_count);
    if (h_count > 0) {
        cudaMemcpy(h_proposals.data(), d_proposals_, h_count * sizeof(YoloProposal), cudaMemcpyDeviceToHost);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    lastPreprocessMs_ = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // 7. CPU NMS and filtering
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<float> distances;

    for (const auto& p : h_proposals) {
        boxes.push_back(cv::Rect((int)p.x0, (int)p.y0, (int)(p.x1 - p.x0), (int)(p.y1 - p.y0)));
        scores.push_back(p.score);
        labels.push_back(p.cls);
        distances.push_back(p.depth_dist);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold_, nmsThreshold_, indices);

    std::vector<Detection> dets;
    for (int idx : indices) {
        const auto& r = boxes[idx];
        dets.push_back({labels[idx], scores[idx], r.x, r.y, r.width, r.height, distances[idx]});
    }

    return dets;
}

void YOLOEngine::ensureDeviceInputBgrCapacity(size_t bytes) {
    if (bytes <= d_input_bgr_cap_) return;
    cudaFree(d_input_bgr_u8_);
    if (cudaMalloc(&d_input_bgr_u8_, bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc d_input_bgr failed");
    d_input_bgr_cap_ = bytes;
}

void YOLOEngine::ensureDeviceDepthCapacity(size_t bytes) {
    if (bytes <= d_depth_cap_) return;
    cudaFree(d_depth_f32_);
    if (cudaMalloc(&d_depth_f32_, bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc d_depth failed");
    d_depth_cap_ = bytes;
}

void YOLOEngine::maybeApplyEngineInputOverride() {
    if (engine_override_h_ <= 0 || engine_override_w_ <= 0) return;
    if (engine_override_h_ == inputH_ && engine_override_w_ == inputW_) return;
    const size_t newInputFloats =
        1ull * 3ull * static_cast<size_t>(engine_override_h_) * static_cast<size_t>(engine_override_w_);
    cudaFree(buffers_[inputIndex_]);
    inputH_ = engine_override_h_;
    inputW_ = engine_override_w_;
    inputSize_ = newInputFloats;
    if (cudaMalloc(&buffers_[inputIndex_], inputSize_ * sizeof(float)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc input after engine_input override failed");
    
    // 重新绑定地址
    context_->setTensorAddress(engine_->getIOTensorName(inputIndex_), buffers_[inputIndex_]);
}

void YOLOEngine::loadEngine(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open engine: " + path);

    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    if (size == 0) throw std::runtime_error("Engine file is empty: " + path);
    f.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    f.read(data.data(), size);
    f.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
    context_.reset(engine_->createExecutionContext());

    int nb = engine_->getNbIOTensors();
    buffers_.resize(nb);
    for (int i = 0; i < nb; i++) {
        const char* name = engine_->getIOTensorName(i);
        auto dims = context_->getTensorShape(name);
        bool isInput = (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; j++) vol *= dims.d[j];

        if (isInput) {
            inputIndex_ = i;
            inputH_ = dims.d[2];
            inputW_ = dims.d[3];
            inputSize_ = vol;
        } else {
            outputIndex_ = i;
            int d1 = dims.d[1], d2 = dims.d[2];
            if (d1 == 6 || d2 == 6) {
                isYOLO26_ = true;
                numAnchors_ = (d1 == 6) ? d2 : d1;
                outputChannelsFirst_ = (d1 == 6);
            } else {
                if (d1 > d2) {
                    numClasses_ = d1 - 4;
                    numAnchors_ = d2;
                    outputChannelsFirst_ = true;
                } else {
                    numClasses_ = d2 - 4;
                    numAnchors_ = d1;
                    outputChannelsFirst_ = false;
                }
            }
            outputNumFloats_ = vol;
        }
        cudaMalloc(&buffers_[i], vol * sizeof(float));
        context_->setTensorAddress(name, buffers_[i]);
    }
    outputHost_.resize(outputNumFloats_);

    maybeApplyEngineInputOverride();
    if (cudaStreamCreate(&preprocessStream_) != cudaSuccess) throw std::runtime_error("cudaStreamCreate failed");
}

std::vector<Detection> YOLOEngine::postprocess(int origW, int origH, float scale, int padLeft, int padTop) {
    if (isYOLO26_) return postprocessYOLO26(origW, origH, scale, padLeft, padTop);
    return postprocessYOLOv8(origW, origH, scale, padLeft, padTop);
}

std::vector<Detection> YOLOEngine::postprocessYOLO26(int origW, int origH, float scale, int padLeft, int padTop) {
    std::vector<Detection> dets;
    const int stride = 6;
    float s = bbox_in_original_space_ ? 1.f : scale;
    int pl = bbox_in_original_space_ ? 0 : padLeft;
    int pt = bbox_in_original_space_ ? 0 : padTop;
    
    for (int i = 0; i < numAnchors_; i++) {
        float row[6];
        if (outputChannelsFirst_) {
            for (int j = 0; j < stride; j++) row[j] = outputHost_[i + j * numAnchors_];
        } else {
            for (int j = 0; j < stride; j++) row[j] = outputHost_[i * stride + j];
        }
        if (row[4] < confThreshold_) continue;

        float x1 = (row[0] - pl) / s, y1 = (row[1] - pt) / s;
        float x2 = (row[2] - pl) / s, y2 = (row[3] - pt) / s;
        int cls = (int)roundf(row[5]);
        
        x1 = std::max(0.f, std::min(x1, (float)origW));
        y1 = std::max(0.f, std::min(y1, (float)origH));
        x2 = std::max(0.f, std::min(x2, (float)origW));
        y2 = std::max(0.f, std::min(y2, (float)origH));
        
        dets.push_back({cls, row[4], (int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1)});
    }
    return dets;
}

std::vector<Detection> YOLOEngine::postprocessYOLOv8(int origW, int origH, float scale, int padLeft, int padTop) {
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    int stride = 4 + numClasses_;
    float s = bbox_in_original_space_ ? 1.f : scale;
    int pl = bbox_in_original_space_ ? 0 : padLeft;
    int pt = bbox_in_original_space_ ? 0 : padTop;

    for (int i = 0; i < numAnchors_; i++) {
        float score = 0;
        int cls = 0;
        if (outputChannelsFirst_) {
            for (int c = 0; c < numClasses_; c++) {
                float sc = outputHost_[i + (4+c)*numAnchors_];
                if (sc > score) { score = sc; cls = c; }
            }
        } else {
            for (int c = 0; c < numClasses_; c++) {
                float sc = outputHost_[i*stride + 4 + c];
                if (sc > score) { score = sc; cls = c; }
            }
        }
        if (score < confThreshold_) continue;

        float xc, yc, w, h;
        if (outputChannelsFirst_) {
            xc = outputHost_[i + 0*numAnchors_]; yc = outputHost_[i + 1*numAnchors_];
            w = outputHost_[i + 2*numAnchors_]; h = outputHost_[i + 3*numAnchors_];
        } else {
            xc = outputHost_[i*stride + 0]; yc = outputHost_[i*stride + 1];
            w = outputHost_[i*stride + 2]; h = outputHost_[i*stride + 3];
        }
        
        float x0 = (xc - w/2 - pl) / s, y0 = (yc - h/2 - pt) / s;
        float x1 = (xc + w/2 - pl) / s, y1 = (yc + h/2 - pt) / s;
        x0 = std::max(0.f, std::min(x0, (float)origW-1));
        y0 = std::max(0.f, std::min(y0, (float)origH-1));
        x1 = std::max(0.f, std::min(x1, (float)origW));
        y1 = std::max(0.f, std::min(y1, (float)origH));
        
        boxes.push_back(cv::Rect((int)x0, (int)y0, (int)(x1 - x0), (int)(y1 - y0)));
        scores.push_back(score);
        labels.push_back(cls);
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold_, nmsThreshold_, indices);
    std::vector<Detection> dets;
    for (int idx : indices) dets.push_back({labels[idx], scores[idx], boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height});
    return dets;
}
