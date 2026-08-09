#ifndef SIXDREPNET_TRT_H
#define SIXDREPNET_TRT_H

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

/** Yaw/pitch/roll in degrees. Convention matches the original 6DRepNet paper:
 *    yaw   : rotation about Y-axis (left/right, positive = turn right)
 *    pitch : rotation about X-axis (up/down,    positive = look down)
 *    roll  : rotation about Z-axis (tilt,        positive = tilt right)
 */
struct HeadPose {
    float yaw{0.f};
    float pitch{0.f};
    float roll{0.f};
};

/**
 * TensorRT wrapper for 6DRepNet head-pose estimation.
 *
 * Expected engine I/O (produced by export_sixdrepnet_onnx.py + trtexec):
 *   Input  "input"  : 1 × 3 × 224 × 224  float32
 *   Output "output" : 1 × 6               float32   (6D rotation representation)
 *
 * The class detects which binding is input/output by tensor mode so it is
 * robust to custom ONNX export naming.
 */
class SixDRepNet_TRT {
public:
    explicit SixDRepNet_TRT(const std::string& trt_path);
    ~SixDRepNet_TRT();

    /** Run inference on a face crop (BGR, any size).
     *  Internally: resize → BGR2RGB → float [0,1] → ImageNet-normalize → CHW.
     *  Returns HeadPose in degrees. */
    HeadPose predict(const cv::Mat& face_bgr);

    /** Draw three-axis arrows centred on face_rect into viz (BGR image).
     *  Skips drawing if |yaw| > skip_threshold_deg (extreme side-face).
     *  Axis colours: X = red, Y = green, Z = blue. */
    void drawAxis(cv::Mat& viz, const cv::Rect& face_rect, const HeadPose& hp,
                  float skip_threshold_deg = 85.f) const;

private:
    /** Gram-Schmidt orthonormalisation: 6D vector → rotation matrix → Euler angles (degrees). */
    static HeadPose sixdToEuler(const float* pred6d);

    void initEngine(const std::string& trt_path);

    Logger gLogger_;

    nvinfer1::IRuntime*          runtime_{nullptr};
    nvinfer1::ICudaEngine*       engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};

    /** GPU buffer for the input tensor  (1×3×224×224 float32). */
    void* d_input_{nullptr};
    /** GPU buffer for the output tensor (1×6 float32). */
    void* d_output_{nullptr};

    /** Host staging buffer for pre-processed input. */
    float h_input_[3 * 224 * 224]{};
    /** Host buffer for output rotation matrix (1×3×3, row-major → 9 floats). */
    float h_output_[9]{};

    static constexpr int INPUT_H    = 224;
    static constexpr int INPUT_W    = 224;
    static constexpr int INPUT_SIZE = 3 * INPUT_H * INPUT_W;   // 150528
    static constexpr int OUTPUT_SIZE = 9;  // 3×3 rotation matrix (Gram-Schmidt inside model)
};

#endif  // SIXDREPNET_TRT_H
