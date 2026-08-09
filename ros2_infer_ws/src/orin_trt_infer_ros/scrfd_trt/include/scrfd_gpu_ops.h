#ifndef SCRFD_GPU_OPS_H
#define SCRFD_GPU_OPS_H

#include <cuda_runtime.h>
#include <stdint.h>

enum class GPUPreprocess { InsightFacePython, NamdvtUpstream };

void scrfd_preprocess_gpu(
    const uint8_t* src, int src_w, int src_h, int src_step,
    float* dst, int dst_w, int dst_h,
    GPUPreprocess p_type,
    float map_scale, float map_wpad, float map_hpad, bool use_wpad_map,
    cudaStream_t stream);

struct GPUFaceProposal {
    float x0, y0, x1, y1;
    float prob;
    float kps[10];
};

void scrfd_postprocess_gpu(
    const float* score_8, const float* bbox_8, const float* kps_8,
    const float* score_16, const float* bbox_16, const float* kps_16,
    const float* score_32, const float* bbox_32, const float* kps_32,
    int output_size_8, int output_size_16, int output_size_32,
    int feat_hw_8, int feat_hw_16, int feat_hw_32,
    const float* base_anchors_8, int num_anchors_8,
    const float* base_anchors_16, int num_anchors_16,
    const float* base_anchors_32, int num_anchors_32,
    float prob_threshold, bool use_sigmoid,
    GPUFaceProposal* out_proposals, int* out_count, int max_proposals,
    cudaStream_t stream);

#endif // SCRFD_GPU_OPS_H
