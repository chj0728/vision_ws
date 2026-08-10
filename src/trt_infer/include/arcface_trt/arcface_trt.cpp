#include "arcface_trt.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// InsightFace ArcFace standard 5-point template (112×112 output)
//   Order: left_eye, right_eye, nose_tip, left_mouth, right_mouth
// ─────────────────────────────────────────────────────────────────────────────
static const cv::Point2f kArcFaceTemplate112[5] = {
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {56.0252f, 71.7366f},
    {41.5493f, 92.3655f},
    {70.7299f, 92.2041f},
};

// ─────────────────────────────────────────────────────────────────────────────
// Umeyama 2-D similarity transform (closed-form, N points)
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat umeyama2D(const cv::Point2f* src, const cv::Point2f* dst, int n) {
    cv::Point2f mu_s(0, 0), mu_d(0, 0);
    for (int i = 0; i < n; ++i) { mu_s += src[i]; mu_d += dst[i]; }
    mu_s *= (1.f / n);
    mu_d *= (1.f / n);

    float sigma_s = 0.f;
    for (int i = 0; i < n; ++i) {
        const auto d = src[i] - mu_s;
        sigma_s += d.x * d.x + d.y * d.y;
    }
    sigma_s /= n;
    if (sigma_s < 1e-8f) sigma_s = 1e-8f;

    cv::Mat H = cv::Mat::zeros(2, 2, CV_32F);
    for (int i = 0; i < n; ++i) {
        const auto ds = src[i] - mu_s;
        const auto dd = dst[i] - mu_d;
        H.at<float>(0, 0) += dd.x * ds.x;
        H.at<float>(0, 1) += dd.x * ds.y;
        H.at<float>(1, 0) += dd.y * ds.x;
        H.at<float>(1, 1) += dd.y * ds.y;
    }
    H *= (1.f / n);

    cv::Mat U, S_vec, Vt;
    cv::SVDecomp(H, S_vec, U, Vt);

    const float det_sign = cv::determinant(U) * cv::determinant(Vt);
    cv::Mat Dc = cv::Mat::eye(2, 2, CV_32F);
    if (det_sign < 0.f) Dc.at<float>(1, 1) = -1.f;

    const float c = (S_vec.at<float>(0) + Dc.at<float>(1, 1) * S_vec.at<float>(1)) / sigma_s;
    const cv::Mat R = U * Dc * Vt;

    cv::Mat mu_s_col = (cv::Mat_<float>(2, 1) << mu_s.x, mu_s.y);
    cv::Mat mu_d_col = (cv::Mat_<float>(2, 1) << mu_d.x, mu_d.y);
    const cv::Mat t  = mu_d_col - c * R * mu_s_col;

    cv::Mat M = cv::Mat::zeros(2, 3, CV_32F);
    const cv::Mat cR = c * R;
    M.at<float>(0, 0) = cR.at<float>(0, 0); M.at<float>(0, 1) = cR.at<float>(0, 1);
    M.at<float>(1, 0) = cR.at<float>(1, 0); M.at<float>(1, 1) = cR.at<float>(1, 1);
    M.at<float>(0, 2) = t.at<float>(0);
    M.at<float>(1, 2) = t.at<float>(1);
    return M;
}

// ─────────────────────────────────────────────────────────────────────────────
// ArcFaceTRT — construction
// ─────────────────────────────────────────────────────────────────────────────
ArcFaceTRT::ArcFaceTRT(const std::string& engine_path, float match_threshold)
    : match_threshold_(match_threshold) {
    initEngine(engine_path);
}

ArcFaceTRT::~ArcFaceTRT() {
    if (d_input_)  cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    delete context_;
    delete engine_;
    delete runtime_;
}

void ArcFaceTRT::initEngine(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) throw std::runtime_error("ArcFaceTRT: cannot open engine: " + path);
    f.seekg(0, std::ios::end);
    const size_t sz = static_cast<size_t>(f.tellg());
    if (sz == 0) throw std::runtime_error("ArcFaceTRT: engine file empty: " + path);
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    f.read(buf.data(), static_cast<std::streamsize>(sz));

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) throw std::runtime_error("ArcFaceTRT: createInferRuntime failed");

    engine_ = runtime_->deserializeCudaEngine(buf.data(), sz);
    if (!engine_)
        throw std::runtime_error("ArcFaceTRT: deserializeCudaEngine failed: " + path);

    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("ArcFaceTRT: createExecutionContext failed");

    const int nb = engine_->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = engine_->getIOTensorName(i);
        const auto mode  = engine_->getTensorIOMode(name);
        auto dims        = engine_->getTensorShape(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (dims.nbDims >= 1 && dims.d[0] == -1) {
                inputBatchDynamic_ = true;
                inputTensorName_   = name;
                dims.d[0] = 1;
                if (!context_->setInputShape(name, dims))
                    throw std::runtime_error(
                        std::string("ArcFaceTRT: setInputShape failed for ") + name);
                std::printf("[ArcFace_TRT] dynamic-batch engine, running batch=1\n");
            } else if (dims.nbDims >= 1 && dims.d[0] != 1) {
                throw std::runtime_error(
                    "ArcFaceTRT: engine has static batch=" +
                    std::to_string(dims.d[0]) + ". Need dynamic or batch=1.");
            }
            void* ptr = nullptr;
            if (cudaMalloc(&ptr, kInputSize * sizeof(float)) != cudaSuccess)
                throw std::runtime_error("ArcFaceTRT: cudaMalloc input failed");
            context_->setTensorAddress(name, ptr);
            d_input_ = ptr;
        } else {
            void* ptr = nullptr;
            if (cudaMalloc(&ptr, kOutputSize * sizeof(float)) != cudaSuccess)
                throw std::runtime_error("ArcFaceTRT: cudaMalloc output failed");
            context_->setTensorAddress(name, ptr);
            d_output_ = ptr;
        }
    }
    if (!d_input_ || !d_output_)
        throw std::runtime_error("ArcFaceTRT: failed to locate input/output GPU buffers");

    std::printf("[ArcFace_TRT] engine loaded: %s\n", path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// Face alignment: 5-point Umeyama → warpAffine → 112×112 BGR
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat ArcFaceTRT::alignFace(const cv::Mat& bgr, const cv::Point2f landmark[5]) {
    if (bgr.empty()) return {};
    for (int i = 0; i < 5; ++i)
        if (!std::isfinite(landmark[i].x) || !std::isfinite(landmark[i].y)) return {};

    const cv::Mat M = umeyama2D(landmark, kArcFaceTemplate112, 5);
    if (M.empty()) return {};

    cv::Mat aligned(112, 112, CV_8UC3);
    cv::warpAffine(bgr, aligned, M, cv::Size(112, 112),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return aligned;
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocess: BGR 112×112 → float32 NCHW, normalised by (x/127.5 - 1.0)
// ─────────────────────────────────────────────────────────────────────────────
static void preprocessArcFace(const cv::Mat& bgr_112, float* dst) {
    cv::Mat rgb;
    cv::cvtColor(bgr_112, rgb, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> ch(3);
    cv::split(rgb, ch);
    constexpr int ch_sz = 112 * 112;
    for (int c = 0; c < 3; ++c) {
        const uint8_t* src_row = ch[c].ptr<uint8_t>();
        float* dst_ch = dst + c * ch_sz;
        for (int p = 0; p < ch_sz; ++p)
            dst_ch[p] = static_cast<float>(src_row[p]) / 127.5f - 1.0f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding extraction
// ─────────────────────────────────────────────────────────────────────────────
bool ArcFaceTRT::extractEmbedding(const cv::Mat& aligned_112, FaceEmbedding& out_emb) const {
    if (!engine_ || aligned_112.empty()) return false;
    if (aligned_112.rows != 112 || aligned_112.cols != 112) return false;

    cv::Mat safe = aligned_112;
    if (safe.depth() != CV_8U) safe.convertTo(safe, CV_8U);
    if (safe.channels() == 1) cv::cvtColor(safe, safe, cv::COLOR_GRAY2BGR);
    else if (safe.channels() != 3) return false;
    if (!safe.isContinuous()) safe = safe.clone();

    preprocessArcFace(safe, h_input_);

    if (inputBatchDynamic_) {
        nvinfer1::Dims4 shape{1, 3, 112, 112};
        context_->setInputShape(inputTensorName_.c_str(), shape);
    }

    cudaStream_t stream = nullptr;
    if (cudaMemcpyAsync(d_input_, h_input_, kInputSize * sizeof(float),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
    if (!context_->enqueueV3(stream)) return false;
    if (cudaMemcpyAsync(h_output_, d_output_, kOutputSize * sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
    cudaStreamSynchronize(stream);

    std::memcpy(out_emb.v, h_output_, kOutputSize * sizeof(float));
    l2Normalize(out_emb);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// L2 normalisation
// ─────────────────────────────────────────────────────────────────────────────
void ArcFaceTRT::l2Normalize(FaceEmbedding& emb) {
    float norm = 0.f;
    for (int i = 0; i < FaceEmbedding::DIM; ++i) norm += emb.v[i] * emb.v[i];
    norm = std::sqrt(norm);
    if (norm < 1e-10f) return;
    const float inv = 1.f / norm;
    for (int i = 0; i < FaceEmbedding::DIM; ++i) emb.v[i] *= inv;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cosine similarity
// ─────────────────────────────────────────────────────────────────────────────
float ArcFaceTRT::cosineSim(const FaceEmbedding& a, const FaceEmbedding& b) {
    float s = 0.f;
    for (int i = 0; i < FaceEmbedding::DIM; ++i) s += a.v[i] * b.v[i];
    return std::clamp(s, -1.f, 1.f);
}
