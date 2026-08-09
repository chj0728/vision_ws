#include "sixdrepnet_trt.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

using namespace nvinfer1;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#define SDR_CHECK(status)                                                  \
    do {                                                                   \
        auto _ret = (status);                                              \
        if (_ret != 0) {                                                   \
            std::fprintf(stderr, "[SixDRepNet_TRT] CUDA error %d at %s:%d\n", \
                         static_cast<int>(_ret), __FILE__, __LINE__);      \
            std::abort();                                                  \
        }                                                                  \
    } while (0)

namespace {

/** Pack a resized 224×224 BGR cv::Mat into a float CHW array with
 *  ImageNet normalisation (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
 *  Input must be CV_8UC3 BGR; output must point to at least 3*224*224 floats. */
void preprocessFace(const cv::Mat& face_bgr, float* dst) {
    constexpr int kH = 224, kW = 224;

    cv::Mat resized;
    cv::resize(face_bgr, resized, cv::Size(kW, kH));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat flt;
    rgb.convertTo(flt, CV_32FC3, 1.0 / 255.0);

    // ImageNet mean/std (RGB order)
    cv::subtract(flt, cv::Scalar(0.485f, 0.456f, 0.406f), flt);
    cv::divide(flt, cv::Scalar(0.229f, 0.224f, 0.225f), flt);

    // HWC → CHW
    std::vector<cv::Mat> channels(3);
    cv::split(flt, channels);
    constexpr int ch_sz = kH * kW;
    for (int c = 0; c < 3; ++c) {
        std::memcpy(dst + c * ch_sz, channels[c].ptr<float>(), ch_sz * sizeof(float));
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// SixDRepNet_TRT implementation
// ---------------------------------------------------------------------------

SixDRepNet_TRT::SixDRepNet_TRT(const std::string& trt_path) {
    initEngine(trt_path);

    // Allocate GPU buffers; identify input vs output by tensor mode.
    const int n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        const TensorIOMode mode = engine_->getTensorIOMode(name);
        const Dims dims = engine_->getTensorShape(name);

        size_t sz = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            sz *= static_cast<size_t>(dims.d[d]);
        }

        void* ptr = nullptr;
        SDR_CHECK(cudaMalloc(&ptr, sz * sizeof(float)));
        context_->setTensorAddress(name, ptr);

        if (mode == TensorIOMode::kINPUT) {
            d_input_ = ptr;
            if (sz != static_cast<size_t>(INPUT_SIZE)) {
                std::fprintf(stderr,
                    "[SixDRepNet_TRT] WARNING: input tensor '%s' size=%zu, expected %d\n",
                    name, sz, INPUT_SIZE);
            }
        } else {
            d_output_ = ptr;
            if (sz != static_cast<size_t>(OUTPUT_SIZE)) {
                std::fprintf(stderr,
                    "[SixDRepNet_TRT] WARNING: output tensor '%s' size=%zu, expected %d\n",
                    name, sz, OUTPUT_SIZE);
            }
        }
    }

    if (!d_input_ || !d_output_) {
        throw std::runtime_error("SixDRepNet_TRT: failed to locate input/output GPU buffers");
    }
    std::printf("[SixDRepNet_TRT] engine loaded from %s\n", trt_path.c_str());
}

SixDRepNet_TRT::~SixDRepNet_TRT() {
    const int n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        void* ptr = const_cast<void*>(context_->getTensorAddress(engine_->getIOTensorName(i)));
        cudaFree(ptr);
    }
    delete context_;
    delete engine_;
    delete runtime_;
}

void SixDRepNet_TRT::initEngine(const std::string& trt_path) {
    std::ifstream file(trt_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("SixDRepNet_TRT: cannot open engine file: " + trt_path);
    }
    file.seekg(0, std::ios::end);
    const size_t sz = static_cast<size_t>(file.tellg());
    if (sz == 0) {
        throw std::runtime_error("SixDRepNet_TRT: engine file is empty: " + trt_path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    file.read(buf.data(), static_cast<std::streamsize>(sz));
    file.close();

    runtime_ = createInferRuntime(gLogger_);
    if (!runtime_) throw std::runtime_error("SixDRepNet_TRT: createInferRuntime failed");

    engine_ = runtime_->deserializeCudaEngine(buf.data(), sz);
    if (!engine_) {
        throw std::runtime_error(
            "SixDRepNet_TRT: deserializeCudaEngine failed – check GPU/TRT version match: " + trt_path);
    }

    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("SixDRepNet_TRT: createExecutionContext failed");
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

HeadPose SixDRepNet_TRT::predict(const cv::Mat& face_bgr) {
    if (face_bgr.empty()) return {};

    // CPU pre-process → host staging buffer
    cv::Mat safe = face_bgr;
    if (face_bgr.depth() != CV_8U) {
        face_bgr.convertTo(safe, CV_8U);
    }
    if (safe.channels() != 3) {
        if (safe.channels() == 1)       cv::cvtColor(safe, safe, cv::COLOR_GRAY2BGR);
        else if (safe.channels() == 4)  cv::cvtColor(safe, safe, cv::COLOR_BGRA2BGR);
        else return {};
    }
    preprocessFace(safe, h_input_);

    // H2D
    SDR_CHECK(cudaMemcpy(d_input_, h_input_, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Inference (synchronous via default CUDA stream 0)
    cudaStream_t stream = nullptr;
    if (!context_->enqueueV3(stream)) {
        std::fprintf(stderr, "[SixDRepNet_TRT] enqueueV3 failed\n");
        return {};
    }
    SDR_CHECK(cudaStreamSynchronize(stream));

    // D2H
    SDR_CHECK(cudaMemcpy(h_output_, d_output_, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    return sixdToEuler(h_output_);
}

// ---------------------------------------------------------------------------
// Rotation Matrix → Euler (degrees)
//
// Output tensor is 1×3×3 (row-major), flattened to 9 floats:
//   p[row*3 + col]  →  p[0..2]=row0, p[3..5]=row1, p[6..8]=row2
//
// Matches Python compute_euler_angles_from_rotation_matrices + draw_axis:
//   yaw   (about Y, head turning L/R) = atan2(-R[2,0], sy)     out[:,1]
//   pitch (about X, head nodding U/D) = atan2( R[2,1], R[2,2]) out[:,0]
//   roll  (about Z, head tilting)     = atan2( R[1,0], R[0,0]) out[:,2]
//
// draw_axis is called in Python as draw_axis(yaw, pitch, roll):
//   yaw is negated inside draw_axis — same negation applied in drawAxis() below.
// ---------------------------------------------------------------------------

HeadPose SixDRepNet_TRT::sixdToEuler(const float* p) {
    // R[row][col] = p[row*3 + col]
    // p[0]=R00, p[1]=R01, p[2]=R02
    // p[3]=R10, p[4]=R11, p[5]=R12
    // p[6]=R20, p[7]=R21, p[8]=R22

    constexpr float kR2D = 180.f / static_cast<float>(M_PI);

    // sy = sqrt(R00^2 + R10^2) — used for stable pitch extraction near gimbal lock
    const float sy = std::sqrt(p[0]*p[0] + p[3]*p[3]);

    // yaw   = atan2(-R[2,0], sy)      (Python out[:,1], parameter "y")
    // pitch = atan2( R[2,1], R[2,2])  (Python out[:,0], parameter "p")
    // roll  = atan2( R[1,0], R[0,0])  (Python out[:,2], parameter "r")
    const float yaw_rad   = std::atan2(-p[6], sy);
    const float pitch_rad = std::atan2( p[7], p[8]);
    const float roll_rad  = std::atan2( p[3], p[0]);

    return {yaw_rad * kR2D, pitch_rad * kR2D, roll_rad * kR2D};
}

// ---------------------------------------------------------------------------
// Visualisation: draw 3-axis arrows
// ---------------------------------------------------------------------------

void SixDRepNet_TRT::drawAxis(cv::Mat& viz, const cv::Rect& face_rect,
                               const HeadPose& hp, float skip_threshold_deg) const {
    if (viz.empty()) return;
    if (std::abs(hp.yaw) > skip_threshold_deg) return;

    // Arrow origin: centre of face bbox
    const float cx = static_cast<float>(face_rect.x) + face_rect.width  * 0.5f;
    const float cy = static_cast<float>(face_rect.y) + face_rect.height * 0.5f;

    // Arrow length proportional to the smaller face dimension
    const float size = static_cast<float>(std::max(face_rect.width, face_rect.height)) * 0.55f;

    // Convert angles to radians (yaw negated to match original draw_axis convention)
    const float pitch = hp.pitch * static_cast<float>(M_PI) / 180.f;
    const float yaw   = -hp.yaw  * static_cast<float>(M_PI) / 180.f;
    const float roll  = hp.roll  * static_cast<float>(M_PI) / 180.f;

    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float cy_ = std::cos(yaw),  sy = std::sin(yaw);
    const float cr = std::cos(roll),  sr = std::sin(roll);

    // X-axis endpoint (red)
    const float x1 = size * (cy_ * cr) + cx;
    const float y1 = size * (cp * sr + cr * sp * sy) + cy;

    // Y-axis endpoint (green)
    const float x2 = size * (-cy_ * sr) + cx;
    const float y2 = size * (cp * cr - sr * sp * sy) + cy;

    // Z-axis endpoint (blue, out of screen = towards viewer)
    const float x3 = size * sy + cx;
    const float y3 = size * (-cy_ * sp) + cy;

    const cv::Point origin(static_cast<int>(std::round(cx)), static_cast<int>(std::round(cy)));
    const cv::Point px(static_cast<int>(std::round(x1)), static_cast<int>(std::round(y1)));
    const cv::Point py(static_cast<int>(std::round(x2)), static_cast<int>(std::round(y2)));
    const cv::Point pz(static_cast<int>(std::round(x3)), static_cast<int>(std::round(y3)));

    const int thick = std::max(1, static_cast<int>(size / 40.f));

    // BGR order
    cv::arrowedLine(viz, origin, px, cv::Scalar(0,   0,   220), thick, cv::LINE_AA, 0, 0.25);  // X red
    cv::arrowedLine(viz, origin, py, cv::Scalar(0,   200,  0 ), thick, cv::LINE_AA, 0, 0.25);  // Y green
    cv::arrowedLine(viz, origin, pz, cv::Scalar(220,  80,   0 ), thick, cv::LINE_AA, 0, 0.25);  // Z blue-ish

    // Compact angle label just above the origin
    char label[48];
    std::snprintf(label, sizeof(label), "Y:%.0f P:%.0f R:%.0f",
                  static_cast<double>(hp.yaw),
                  static_cast<double>(hp.pitch),
                  static_cast<double>(hp.roll));
    const double kScale = std::clamp(size / 110.0, 0.32, 0.58);
    const int kThick = 1;
    int baseline = 0;
    const cv::Size tsz = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, kScale, kThick, &baseline);
    const int tx = std::clamp(static_cast<int>(cx) - tsz.width / 2, 0, std::max(0, viz.cols - tsz.width));
    const int ty = std::max(tsz.height + 2, static_cast<int>(cy) - static_cast<int>(size * 0.55f) - 4);
    // Dark outline for readability
    cv::putText(viz, label, {tx, ty}, cv::FONT_HERSHEY_DUPLEX, kScale,
                cv::Scalar(10, 10, 12), kThick + 2, cv::LINE_AA);
    cv::putText(viz, label, {tx, ty}, cv::FONT_HERSHEY_DUPLEX, kScale,
                cv::Scalar(240, 230, 60), kThick, cv::LINE_AA);
}
