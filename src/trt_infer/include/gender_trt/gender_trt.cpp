#include "gender_trt.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// GenderAgeTRT — construction / engine loading
// ─────────────────────────────────────────────────────────────────────────────

GenderAgeTRT::GenderAgeTRT(const std::string& engine_path) {
    initEngine(engine_path);
}

GenderAgeTRT::~GenderAgeTRT() {
    if (d_input_)  cudaFree(d_input_);
    for (auto& out : outputs_) {
        if (out.d_ptr) cudaFree(out.d_ptr);
    }
    delete context_;
    delete engine_;
    delete runtime_;
}

int GenderAgeTRT::tensorVolumeNoBatch(const nvinfer1::Dims& dims) {
    int v = 1;
    for (int i = 1; i < dims.nbDims; ++i) {
        if (dims.d[i] > 0) v *= dims.d[i];
    }
    return std::max(v, 1);
}

void GenderAgeTRT::initEngine(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) throw std::runtime_error("GenderAgeTRT: cannot open engine: " + path);
    f.seekg(0, std::ios::end);
    const size_t sz = static_cast<size_t>(f.tellg());
    if (sz == 0) throw std::runtime_error("GenderAgeTRT: engine file empty: " + path);
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    f.read(buf.data(), static_cast<std::streamsize>(sz));

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) throw std::runtime_error("GenderAgeTRT: createInferRuntime failed");

    engine_ = runtime_->deserializeCudaEngine(buf.data(), sz);
    if (!engine_) throw std::runtime_error("GenderAgeTRT: deserializeCudaEngine failed: " + path);

    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("GenderAgeTRT: createExecutionContext failed");

    const int nb = engine_->getNbIOTensors();
    std::printf("[GenderAge_TRT] IO tensors: %d\n", nb);
    for (int i = 0; i < nb; ++i) {
        const char* name = engine_->getIOTensorName(i);
        const auto mode  = engine_->getTensorIOMode(name);
        auto dims        = engine_->getTensorShape(name);
        const char* mode_s = (mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
        std::printf("[GenderAge_TRT]  - %s  %s  dims=[", mode_s, name);
        for (int d = 0; d < dims.nbDims; ++d) {
            std::printf("%s%ld", d ? "," : "", static_cast<long>(dims.d[d]));
        }
        std::printf("]\n");

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            // Detect dynamic batch
            if (dims.nbDims >= 1 && dims.d[0] == -1) {
                inputBatchDynamic_ = true;
                inputTensorName_   = name;
                dims.d[0] = 1;
                context_->setInputShape(name, dims);
                std::printf("[GenderAge_TRT] dynamic-batch engine, running batch=1\n");
            }
            // Detect input spatial size from dims (NCHW: [N,3,H,W])
            if (dims.nbDims >= 4) {
                input_h_ = static_cast<int>(dims.d[2] > 0 ? dims.d[2] : input_h_);
                input_w_ = static_cast<int>(dims.d[3] > 0 ? dims.d[3] : input_w_);
            }
            const int vol = 1 * 3 * input_h_ * input_w_;
            h_input_.resize(static_cast<size_t>(vol));
            void* ptr = nullptr;
            if (cudaMalloc(&ptr, static_cast<size_t>(vol) * sizeof(float)) != cudaSuccess)
                throw std::runtime_error("GenderAgeTRT: cudaMalloc input failed");
            context_->setTensorAddress(name, ptr);
            d_input_ = ptr;

        } else {
            // Output tensor — keep all outputs instead of overwriting one buffer
            OutputTensor out;
            out.name   = name;
            out.dims   = dims;
            out.volume = tensorVolumeNoBatch(dims);
            out.h_data.resize(static_cast<size_t>(out.volume));
            void* ptr = nullptr;
            if (cudaMalloc(&ptr, static_cast<size_t>(out.volume) * sizeof(float)) != cudaSuccess)
                throw std::runtime_error("GenderAgeTRT: cudaMalloc output failed");
            context_->setTensorAddress(name, ptr);
            out.d_ptr = ptr;
            outputs_.push_back(std::move(out));
        }
    }

    if (!d_input_ || outputs_.empty())
        throw std::runtime_error("GenderAgeTRT: failed to locate input/output GPU buffers");

    std::printf("[GenderAge_TRT] engine loaded: %s  input=%dx%d  outputs=%zu\n",
                path.c_str(), input_w_, input_h_, outputs_.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocess: BGR → RGB, resize to input_h×input_w.
// 归一化：原始 [0,255] float，不做减均值/除方差。
// 这是本模型训练时写死的约定（见 pvp/src/kernels/preprocess_genderage.cu）。
// Pack as NCHW float32.
// ─────────────────────────────────────────────────────────────────────────────
static void preprocessGenderAge(const cv::Mat& bgr, int in_h, int in_w, float* dst) {
    cv::Mat resized;
    if (bgr.cols == in_w && bgr.rows == in_h) {
        resized = bgr;
    } else {
        cv::resize(bgr, resized, cv::Size(in_w, in_h), 0, 0, cv::INTER_LINEAR);
    }

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> ch(3);
    cv::split(rgb, ch);
    const int ch_sz = in_h * in_w;
    for (int c = 0; c < 3; ++c) {
        const uint8_t* src = ch[c].ptr<uint8_t>();
        float* dst_ch = dst + c * ch_sz;
        for (int p = 0; p < ch_sz; ++p) {
            dst_ch[p] = static_cast<float>(src[p]);   // raw [0,255]，不归一化
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sigmoid helper
// ─────────────────────────────────────────────────────────────────────────────
static inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

static std::string lowerCopy(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// predict()
// ─────────────────────────────────────────────────────────────────────────────
GenderResult GenderAgeTRT::predict(const cv::Mat& aligned_bgr) const {
    GenderResult result;
    if (!engine_ || aligned_bgr.empty()) return result;

    cv::Mat safe = aligned_bgr;
    if (safe.depth() != CV_8U) safe.convertTo(safe, CV_8U);
    if (safe.channels() == 1) cv::cvtColor(safe, safe, cv::COLOR_GRAY2BGR);
    if (safe.channels() != 3) return result;
    if (!safe.isContinuous()) safe = safe.clone();

    preprocessGenderAge(safe, input_h_, input_w_, h_input_.data());

    if (inputBatchDynamic_) {
        nvinfer1::Dims4 shape{1, 3, input_h_, input_w_};
        context_->setInputShape(inputTensorName_.c_str(), shape);
    }

    cudaStream_t stream = nullptr;
    if (cudaMemcpyAsync(d_input_, h_input_.data(),
                        h_input_.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) return result;
    if (!context_->enqueueV3(stream)) return result;
    for (auto& out : outputs_) {
        if (cudaMemcpyAsync(out.h_data.data(), out.d_ptr,
                            out.h_data.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess) return result;
    }
    cudaStreamSynchronize(stream);

    // 首次推理打印所有输出值（诊断用）
    if (!debug_printed_) {
        debug_printed_ = true;
        for (const auto& out : outputs_) {
            std::printf("[GenderAge_TRT_RAW] %s size=%d", out.name.c_str(), out.volume);
            for (int i = 0; i < std::min(out.volume, 6); ++i) {
                std::printf("  [%d]=%.6f", i, out.h_data[i]);
            }
            std::printf("\n");
        }
        std::fflush(stdout);
    }

    // Pick gender/age output tensors by name first, then by volume.
    // This avoids systematic mis-parse when engine exports [gender, age] as size=2.
    const OutputTensor* out3 = nullptr;
    const OutputTensor* out2 = nullptr;
    const OutputTensor* out1_first = nullptr;
    const OutputTensor* out1_second = nullptr;
    const OutputTensor* out_gender = nullptr;
    const OutputTensor* out_age = nullptr;
    for (const auto& out : outputs_) {
        const std::string n = lowerCopy(out.name);
        if (n.find("gender") != std::string::npos && out_gender == nullptr) out_gender = &out;
        if (n.find("age")    != std::string::npos && out_age    == nullptr) out_age    = &out;
        if (out.volume >= 3 && out3 == nullptr) out3 = &out;
        if (out.volume == 2 && out2 == nullptr) out2 = &out;
        if (out.volume == 1) {
            if (out1_first == nullptr) out1_first = &out;
            else if (out1_second == nullptr) out1_second = &out;
        }
    }

    auto parse_gender_logits = [&](float f_logit, float m_logit) {
        if (m_logit >= f_logit) {
            result.gender = GenderResult::MALE;
        } else {
            result.gender = GenderResult::FEMALE;
        }
        const float mx = std::max(f_logit, m_logit);
        const float exp_m = std::exp(m_logit - mx);
        const float exp_f = std::exp(f_logit - mx);
        result.conf = std::max(exp_m, exp_f) / (exp_m + exp_f);
    };

    auto parse_age = [&](float v) {
        // Usually age_norm in [0,1], but keep robust for already-in-years export.
        if (v >= 0.f && v <= 1.5f) result.age = v * 100.f;
        else result.age = v;
    };

    // Case A: combined output [female, male, age]
    if (out3) {
        parse_gender_logits(out3->h_data[0], out3->h_data[1]);
        parse_age(out3->h_data[2]);
        result.conf = std::clamp(result.conf, 0.f, 1.f);
        return result;
    }

    // Case B: split outputs, common is [2] gender + [1] age
    if (out2) {
        const std::string out2_name = lowerCopy(out2->name);
        const float v0 = out2->h_data[0];
        const float v1 = out2->h_data[1];

        // Explicit named outputs: trust semantic names first.
        if (out_gender && out_gender->volume == 2) {
            parse_gender_logits(out_gender->h_data[0], out_gender->h_data[1]);
        } else if (out_gender && out_gender->volume == 1) {
            const float g = out_gender->h_data[0];
            if (g >= 0.f && g <= 1.f) {
                result.gender = (g >= 0.5f) ? GenderResult::MALE : GenderResult::FEMALE;
                result.conf   = std::max(g, 1.f - g);
            } else {
                result.gender = (g > 0.f) ? GenderResult::MALE : GenderResult::FEMALE;
                result.conf   = sigmoid(std::abs(g));
            }
        } else if (out2_name.find("age") != std::string::npos) {
            // Some exports pack [gender_scalar, age] in one tensor.
            const float g = v0;
            if (g >= 0.f && g <= 1.f) {
                result.gender = (g >= 0.5f) ? GenderResult::MALE : GenderResult::FEMALE;
                result.conf   = std::max(g, 1.f - g);
            } else {
                result.gender = (g > 0.f) ? GenderResult::MALE : GenderResult::FEMALE;
                result.conf   = sigmoid(std::abs(g));
            }
            parse_age(v1);
        } else if (!out1_first && v1 >= 0.f && v1 <= 1.5f &&
                   (v0 < 0.f || v0 > 1.f || std::fabs(v0 - 0.5f) > 0.35f)) {
            // Heuristic for anonymous 2-value output: likely [gender_scalar, age].
            const float g = v0;
            if (g >= 0.f && g <= 1.f) {
                result.gender = (g >= 0.5f) ? GenderResult::MALE : GenderResult::FEMALE;
                result.conf   = std::max(g, 1.f - g);
            } else {
                result.gender = (g > 0.f) ? GenderResult::MALE : GenderResult::FEMALE;
                result.conf   = sigmoid(std::abs(g));
            }
            parse_age(v1);
        } else {
            parse_gender_logits(v0, v1);
        }

        if (out_age && out_age->volume >= 1) parse_age(out_age->h_data[0]);
        else if (out1_first) parse_age(out1_first->h_data[0]);
        result.conf = std::clamp(result.conf, 0.f, 1.f);
        return result;
    }

    // Case C: only one scalar output (gender logit or prob)
    if (out1_first) {
        const float g = out1_first->h_data[0];
        if (out1_second) parse_age(out1_second->h_data[0]);
        if (g >= 0.f && g <= 1.f) {
            // probability-like scalar: assume male prob.
            result.gender = (g >= 0.5f) ? GenderResult::MALE : GenderResult::FEMALE;
            result.conf   = std::max(g, 1.f - g);
        } else {
            result.gender = (g > 0.f) ? GenderResult::MALE : GenderResult::FEMALE;
            result.conf   = sigmoid(std::abs(g));
        }
        result.conf = std::clamp(result.conf, 0.f, 1.f);
        return result;
    }

    // Fallback: no parseable outputs.
    return result;
}
