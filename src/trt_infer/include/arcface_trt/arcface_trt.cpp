#include "arcface_trt.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// InsightFace ArcFace standard 5-point template (112×112 output)
//   Order: left_eye, right_eye, nose_tip, left_mouth, right_mouth
// ─────────────────────────────────────────────────────────────────────────────
static const cv::Point2f kArcFaceTemplate112[5] = {
    {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
    {41.5493f, 92.3655f}, {70.7299f, 92.2041f},
};

static const char *dataTypeName(nvinfer1::DataType type) {
  switch (type) {
  case nvinfer1::DataType::kFLOAT:
    return "float32";
  case nvinfer1::DataType::kHALF:
    return "float16";
  case nvinfer1::DataType::kINT8:
    return "int8";
  case nvinfer1::DataType::kINT32:
    return "int32";
  case nvinfer1::DataType::kBOOL:
    return "bool";
  default:
    return "unsupported";
  }
}

static size_t dataTypeSize(nvinfer1::DataType type) {
  switch (type) {
  case nvinfer1::DataType::kFLOAT:
    return sizeof(float);
  case nvinfer1::DataType::kHALF:
    return sizeof(__half);
  default:
    return 0;
  }
}

static int64_t dimsVolume(const nvinfer1::Dims &dims) {
  int64_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0)
      return -1;
    volume *= dims.d[i];
  }
  return volume;
}

static std::string dimsString(const nvinfer1::Dims &dims) {
  std::ostringstream stream;
  stream << '[';
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i > 0)
      stream << ',';
    stream << dims.d[i];
  }
  stream << ']';
  return stream.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// Umeyama 2-D similarity transform (closed-form, N points)
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat umeyama2D(const cv::Point2f *src, const cv::Point2f *dst,
                         int n) {
  cv::Point2f mu_s(0, 0), mu_d(0, 0);
  for (int i = 0; i < n; ++i) {
    mu_s += src[i];
    mu_d += dst[i];
  }
  mu_s *= (1.f / n);
  mu_d *= (1.f / n);

  float sigma_s = 0.f;
  for (int i = 0; i < n; ++i) {
    const auto d = src[i] - mu_s;
    sigma_s += d.x * d.x + d.y * d.y;
  }
  sigma_s /= n;
  if (sigma_s < 1e-8f)
    sigma_s = 1e-8f;

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
  if (det_sign < 0.f)
    Dc.at<float>(1, 1) = -1.f;

  const float c =
      (S_vec.at<float>(0) + Dc.at<float>(1, 1) * S_vec.at<float>(1)) / sigma_s;
  const cv::Mat R = U * Dc * Vt;

  cv::Mat mu_s_col = (cv::Mat_<float>(2, 1) << mu_s.x, mu_s.y);
  cv::Mat mu_d_col = (cv::Mat_<float>(2, 1) << mu_d.x, mu_d.y);
  const cv::Mat t = mu_d_col - c * R * mu_s_col;

  cv::Mat M = cv::Mat::zeros(2, 3, CV_32F);
  const cv::Mat cR = c * R;
  M.at<float>(0, 0) = cR.at<float>(0, 0);
  M.at<float>(0, 1) = cR.at<float>(0, 1);
  M.at<float>(1, 0) = cR.at<float>(1, 0);
  M.at<float>(1, 1) = cR.at<float>(1, 1);
  M.at<float>(0, 2) = t.at<float>(0);
  M.at<float>(1, 2) = t.at<float>(1);
  return M;
}

// ─────────────────────────────────────────────────────────────────────────────
// ArcFaceTRT — construction
// ─────────────────────────────────────────────────────────────────────────────
ArcFaceTRT::ArcFaceTRT(const std::string &engine_path, float match_threshold)
    : match_threshold_(match_threshold) {
  initEngine(engine_path);
}

ArcFaceTRT::~ArcFaceTRT() {
  if (d_input_)
    cudaFree(d_input_);
  if (d_output_)
    cudaFree(d_output_);
  delete context_;
  delete engine_;
  delete runtime_;
}

void ArcFaceTRT::initEngine(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good())
    throw std::runtime_error("ArcFaceTRT: cannot open engine: " + path);
  f.seekg(0, std::ios::end);
  const size_t sz = static_cast<size_t>(f.tellg());
  if (sz == 0)
    throw std::runtime_error("ArcFaceTRT: engine file empty: " + path);
  f.seekg(0, std::ios::beg);
  std::vector<char> buf(sz);
  f.read(buf.data(), static_cast<std::streamsize>(sz));

  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_)
    throw std::runtime_error("ArcFaceTRT: createInferRuntime failed");

  engine_ = runtime_->deserializeCudaEngine(buf.data(), sz);
  if (!engine_)
    throw std::runtime_error("ArcFaceTRT: deserializeCudaEngine failed: " +
                             path);

  context_ = engine_->createExecutionContext();
  if (!context_)
    throw std::runtime_error("ArcFaceTRT: createExecutionContext failed");

  int input_count = 0;
  int output_count = 0;
  nvinfer1::Dims input_dims{};

  const int nb = engine_->getNbIOTensors();
  for (int i = 0; i < nb; ++i) {
    const char *name = engine_->getIOTensorName(i);
    const auto mode = engine_->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      ++input_count;
      inputTensorName_ = name;
      inputDataType_ = engine_->getTensorDataType(name);
      input_dims = engine_->getTensorShape(name);
    } else {
      ++output_count;
      outputTensorName_ = name;
      outputDataType_ = engine_->getTensorDataType(name);
    }
  }

  if (input_count != 1 || output_count != 1) {
    throw std::runtime_error(
        "ArcFaceTRT: expected exactly 1 input and 1 output, got " +
        std::to_string(input_count) + " input(s), " +
        std::to_string(output_count) + " output(s)");
  }
  if (dataTypeSize(inputDataType_) == 0 || dataTypeSize(outputDataType_) == 0) {
    throw std::runtime_error(
        std::string(
            "ArcFaceTRT: only float32/float16 I/O is supported, got input=") +
        dataTypeName(inputDataType_) +
        " output=" + dataTypeName(outputDataType_));
  }
  if (input_dims.nbDims != 4) {
    throw std::runtime_error(
        "ArcFaceTRT: expected input shape [1,3,112,112], got " +
        dimsString(input_dims));
  }

  const int expected_input_dims[4] = {1, 3, 112, 112};
  for (int i = 0; i < 4; ++i) {
    if (input_dims.d[i] == -1) {
      inputShapeDynamic_ = true;
    } else if (input_dims.d[i] != expected_input_dims[i]) {
      throw std::runtime_error(
          "ArcFaceTRT: expected input shape [1,3,112,112], got " +
          dimsString(input_dims));
    }
  }
  if (inputShapeDynamic_ &&
      !context_->setInputShape(inputTensorName_.c_str(),
                               nvinfer1::Dims4{1, 3, 112, 112})) {
    throw std::runtime_error("ArcFaceTRT: setInputShape failed for " +
                             inputTensorName_);
  }

  const nvinfer1::Dims resolved_input =
      context_->getTensorShape(inputTensorName_.c_str());
  const nvinfer1::Dims resolved_output =
      context_->getTensorShape(outputTensorName_.c_str());
  if (dimsVolume(resolved_input) != kInputSize) {
    throw std::runtime_error("ArcFaceTRT: resolved input shape is " +
                             dimsString(resolved_input) +
                             ", expected [1,3,112,112]");
  }
  if (dimsVolume(resolved_output) != kOutputSize) {
    throw std::runtime_error("ArcFaceTRT: output shape is " +
                             dimsString(resolved_output) +
                             ", expected 512 values");
  }

  inputBytes_ = kInputSize * dataTypeSize(inputDataType_);
  outputBytes_ = kOutputSize * dataTypeSize(outputDataType_);
  if (inputDataType_ == nvinfer1::DataType::kHALF)
    h_input_half_.resize(kInputSize);
  if (outputDataType_ == nvinfer1::DataType::kHALF)
    h_output_half_.resize(kOutputSize);

  if (cudaMalloc(&d_input_, inputBytes_) != cudaSuccess)
    throw std::runtime_error("ArcFaceTRT: cudaMalloc input failed");
  if (cudaMalloc(&d_output_, outputBytes_) != cudaSuccess)
    throw std::runtime_error("ArcFaceTRT: cudaMalloc output failed");
  if (!context_->setTensorAddress(inputTensorName_.c_str(), d_input_) ||
      !context_->setTensorAddress(outputTensorName_.c_str(), d_output_)) {
    throw std::runtime_error("ArcFaceTRT: setTensorAddress failed");
  }

  std::printf("[ArcFace_TRT] engine loaded: %s  input=%s %s  output=%s %s\n",
              path.c_str(), inputTensorName_.c_str(),
              dataTypeName(inputDataType_), outputTensorName_.c_str(),
              dataTypeName(outputDataType_));
}

// ─────────────────────────────────────────────────────────────────────────────
// Face alignment: 5-point Umeyama → warpAffine → 112×112 BGR
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat ArcFaceTRT::alignFace(const cv::Mat &bgr,
                              const cv::Point2f landmark[5]) {
  if (bgr.empty())
    return {};
  for (int i = 0; i < 5; ++i)
    if (!std::isfinite(landmark[i].x) || !std::isfinite(landmark[i].y))
      return {};

  const cv::Mat M = umeyama2D(landmark, kArcFaceTemplate112, 5);
  if (M.empty())
    return {};

  cv::Mat aligned(112, 112, CV_8UC3);
  cv::warpAffine(bgr, aligned, M, cv::Size(112, 112), cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  return aligned;
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocess: BGR 112×112 → float32 NCHW, normalised by (x/127.5 - 1.0)
// ─────────────────────────────────────────────────────────────────────────────
static void preprocessArcFace(const cv::Mat &bgr_112, float *dst) {
  cv::Mat rgb;
  cv::cvtColor(bgr_112, rgb, cv::COLOR_BGR2RGB);
  std::vector<cv::Mat> ch(3);
  cv::split(rgb, ch);
  constexpr int ch_sz = 112 * 112;
  for (int c = 0; c < 3; ++c) {
    const uint8_t *src_row = ch[c].ptr<uint8_t>();
    float *dst_ch = dst + c * ch_sz;
    for (int p = 0; p < ch_sz; ++p)
      dst_ch[p] = static_cast<float>(src_row[p]) / 127.5f - 1.0f;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding extraction
// ─────────────────────────────────────────────────────────────────────────────
bool ArcFaceTRT::extractEmbedding(const cv::Mat &aligned_112,
                                  FaceEmbedding &out_emb) const {
  std::fill(std::begin(out_emb.v), std::end(out_emb.v), 0.f);
  if (!engine_ || !context_ || aligned_112.empty())
    return false;
  if (aligned_112.rows != 112 || aligned_112.cols != 112)
    return false;

  cv::Mat safe = aligned_112;
  if (safe.depth() != CV_8U)
    safe.convertTo(safe, CV_8U);
  if (safe.channels() == 1)
    cv::cvtColor(safe, safe, cv::COLOR_GRAY2BGR);
  else if (safe.channels() != 3)
    return false;
  if (!safe.isContinuous())
    safe = safe.clone();

  preprocessArcFace(safe, h_input_);

  if (inputShapeDynamic_ &&
      !context_->setInputShape(inputTensorName_.c_str(),
                               nvinfer1::Dims4{1, 3, 112, 112})) {
    std::fprintf(stderr,
                 "[ArcFace_TRT] setInputShape failed during inference\n");
    return false;
  }

  cudaStream_t stream = nullptr;
  const void *host_input = h_input_;
  if (inputDataType_ == nvinfer1::DataType::kHALF) {
    for (int i = 0; i < kInputSize; ++i)
      h_input_half_[static_cast<size_t>(i)] = __float2half(h_input_[i]);
    host_input = h_input_half_.data();
  }

  cudaError_t status = cudaMemcpyAsync(d_input_, host_input, inputBytes_,
                                       cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "[ArcFace_TRT] input copy failed: %s\n",
                 cudaGetErrorString(status));
    return false;
  }
  if (!context_->enqueueV3(stream)) {
    std::fprintf(stderr, "[ArcFace_TRT] enqueueV3 failed\n");
    return false;
  }

  void *host_output = outputDataType_ == nvinfer1::DataType::kHALF
                          ? static_cast<void *>(h_output_half_.data())
                          : static_cast<void *>(h_output_);
  status = cudaMemcpyAsync(host_output, d_output_, outputBytes_,
                           cudaMemcpyDeviceToHost, stream);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "[ArcFace_TRT] output copy failed: %s\n",
                 cudaGetErrorString(status));
    return false;
  }
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "[ArcFace_TRT] inference synchronization failed: %s\n",
                 cudaGetErrorString(status));
    return false;
  }

  if (outputDataType_ == nvinfer1::DataType::kHALF) {
    for (int i = 0; i < kOutputSize; ++i)
      out_emb.v[i] = __half2float(h_output_half_[static_cast<size_t>(i)]);
  } else {
    std::memcpy(out_emb.v, h_output_, outputBytes_);
  }
  if (!l2Normalize(out_emb)) {
    std::fprintf(stderr,
                 "[ArcFace_TRT] invalid embedding: non-finite or zero norm\n");
    return false;
  }
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// L2 normalisation
// ─────────────────────────────────────────────────────────────────────────────
bool ArcFaceTRT::isFiniteEmbedding(const FaceEmbedding &emb) {
  for (int i = 0; i < FaceEmbedding::DIM; ++i)
    if (!std::isfinite(emb.v[i]))
      return false;
  return true;
}

bool ArcFaceTRT::l2Normalize(FaceEmbedding &emb) {
  if (!isFiniteEmbedding(emb)) {
    std::fill(std::begin(emb.v), std::end(emb.v), 0.f);
    return false;
  }

  double norm_squared = 0.0;
  for (int i = 0; i < FaceEmbedding::DIM; ++i)
    norm_squared += static_cast<double>(emb.v[i]) * emb.v[i];
  const double norm = std::sqrt(norm_squared);
  if (!std::isfinite(norm) || norm < 1e-10) {
    std::fill(std::begin(emb.v), std::end(emb.v), 0.f);
    return false;
  }

  const float inv = static_cast<float>(1.0 / norm);
  for (int i = 0; i < FaceEmbedding::DIM; ++i)
    emb.v[i] *= inv;
  return isFiniteEmbedding(emb);
}

// ─────────────────────────────────────────────────────────────────────────────
// Cosine similarity
// ─────────────────────────────────────────────────────────────────────────────
float ArcFaceTRT::cosineSim(const FaceEmbedding &a, const FaceEmbedding &b) {
  if (!isFiniteEmbedding(a) || !isFiniteEmbedding(b))
    return -1.f;
  float s = 0.f;
  for (int i = 0; i < FaceEmbedding::DIM; ++i)
    s += a.v[i] * b.v[i];
  return std::clamp(s, -1.f, 1.f);
}
