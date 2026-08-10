#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void yolo_launch_letterbox_to_nchw(const uint8_t* d_bgr, int sw, int sh, float* d_nchw, int dw, int dh,
                                     int new_w, int new_h, int pad_left, int pad_top, cudaStream_t stream);

void yolo_launch_resize_to_nchw(const uint8_t* d_bgr, int sw, int sh, float* d_nchw, int dw, int dh,
                                cudaStream_t stream);

struct YoloProposal {
    float x0, y0, x1, y1;
    float score;
    int cls;
    float depth_dist;
};

void yolo_postprocess_gpu(
    const float* d_output, int num_anchors, int num_classes, float conf_threshold,
    bool is_yolo26, bool channels_first,
    int orig_w, int orig_h, float scale, int pad_left, int pad_top,
    YoloProposal* d_proposals, int* d_count, int max_proposals,
    cudaStream_t stream);

/** GPU 深度采样：计算 ROI 内的鲁棒分位数距离。
 *  depth_scale_x = depth_w / color_w，depth_scale_y = depth_h / color_h，
 *  用于将 bbox（彩色图坐标系）映射到深度图坐标系，分辨率相同时传 1.0f 即可。*/
void yolo_depth_sampling_gpu(
    const float* d_depth, int depth_w, int depth_h,
    YoloProposal* d_proposals, int count,
    float min_depth, float max_depth,
    float roi_y0_n, float roi_y1_n, float x_margin_n,
    float trim_close_ratio, float percentile,
    float depth_scale_x, float depth_scale_y,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
