#include "scrfd_gpu_ops.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

namespace {

__device__ float sigmoid_f(float x) {
    x = fmaxf(-50.0f, fminf(50.0f, x));
    return 1.0f / (1.0f + expf(-x));
}

// SCRFD Preprocess: Bilinear resize + Norm + NCHW
// InsightFace: padding at bottom-right, RGB
// Namdvt: padding at top-left, BGR
__global__ void scrfd_preprocess_kernel(
    const uint8_t* src, int sw, int sh, int s_step,
    float* dst, int dw, int dh,
    GPUPreprocess p_type,
    float map_scale, float map_wpad, float map_hpad, bool use_wpad_map) {
    
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dw || dy >= dh) return;

    float sx, sy;
    if (p_type == GPUPreprocess::InsightFacePython) {
        // InsightFace padding is usually at the bottom/right. 
        // We just resize the original image to [new_w, new_height] and place it at (0,0)
        sx = dx / map_scale;
        sy = dy / map_scale;
    } else {
        // Namdvt: padding at top-left (wpad, hpad)
        sx = (dx - map_wpad) / map_scale;
        sy = (dy - map_hpad) / map_scale;
    }

    float r = 0, g = 0, b = 0;
    if (sx >= 0 && sx < sw - 1 && sy >= 0 && sy < sh - 1) {
        int x0 = (int)sx;
        int y0 = (int)sy;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float ux = sx - x0;
        float uy = sy - y0;

        auto get_pix = [&](int x, int y) {
            return src + y * s_step + x * 3;
        };

        const uint8_t* p00 = get_pix(x0, y0);
        const uint8_t* p10 = get_pix(x1, y0);
        const uint8_t* p01 = get_pix(x0, y1);
        const uint8_t* p11 = get_pix(x1, y1);

        auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
        
        b = lerp(lerp(p00[0], p10[0], ux), lerp(p01[0], p11[0], ux), uy);
        g = lerp(lerp(p00[1], p10[1], ux), lerp(p01[1], p11[1], ux), uy);
        r = lerp(lerp(p00[2], p10[2], ux), lerp(p01[2], p11[2], ux), uy);
    }

    // Norm: (x - 127.5) / 128
    float v0, v1, v2;
    if (p_type == GPUPreprocess::InsightFacePython) {
        // RGB
        v0 = (r - 127.5f) / 128.0f;
        v1 = (g - 127.5f) / 128.0f;
        v2 = (b - 127.5f) / 128.0f;
    } else {
        // BGR
        v0 = (b - 127.5f) / 128.0f;
        v1 = (g - 127.5f) / 128.0f;
        v2 = (r - 127.5f) / 128.0f;
    }

    int plane = dw * dh;
    dst[dy * dw + dx] = v0;
    dst[plane + dy * dw + dx] = v1;
    dst[2 * plane + dy * dw + dx] = v2;
}

__global__ void scrfd_postprocess_stride_kernel(
    const float* score_ptr, const float* bbox_ptr, const float* kps_ptr,
    int n, int spatial_side, int feat_stride, const float* base_anchors, int num_anchors,
    float prob_threshold, bool use_sigmoid,
    GPUFaceProposal* out_proposals, int* out_count, int max_proposals) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || num_anchors <= 0 || spatial_side <= 0) return;

    /* 显式 H=W=spatial_side（640 输入下为 80/40/20），避免设备上 sqrt 舍入导致整层无检出 */
    const int w = spatial_side;
    const int plane = w * w;

    const int q = idx / plane;
    const int spatial = idx % plane;
    const int i = spatial / w;
    const int j = spatial % w;

    float raw_score = score_ptr[idx];
    float prob = use_sigmoid ? sigmoid_f(raw_score) : raw_score;

    if (prob < prob_threshold) return;

    int count = atomicAdd(out_count, 1);
    if (count >= max_proposals) return;

    const float* anchor = base_anchors + q * 4;
    float anchor_x = anchor[0] + j * feat_stride;
    float anchor_y = anchor[1] + i * feat_stride;
    float anchor_w = anchor[2] - anchor[0];
    float anchor_h = anchor[3] - anchor[1];

    float cx = anchor_x + anchor_w * 0.5f;
    float cy = anchor_y + anchor_h * 0.5f;

    /* bbox / kps：NCHW [1,8,H,W]、[1,20,H,W]，与 CPU generate_proposals 中 (4*q+d)*plane+spatial 一致 */
    float dx = bbox_ptr[(4 * q + 0) * plane + spatial] * feat_stride;
    float dy = bbox_ptr[(4 * q + 1) * plane + spatial] * feat_stride;
    float dw = bbox_ptr[(4 * q + 2) * plane + spatial] * feat_stride;
    float dh = bbox_ptr[(4 * q + 3) * plane + spatial] * feat_stride;

    out_proposals[count].x0 = cx - dx;
    out_proposals[count].y0 = cy - dy;
    out_proposals[count].x1 = cx + dw;
    out_proposals[count].y1 = cy + dh;
    out_proposals[count].prob = prob;

    for (int k = 0; k < 5; ++k) {
        const int kc = 10 * q + 2 * k;
        out_proposals[count].kps[k * 2] = cx + kps_ptr[kc * plane + spatial] * feat_stride;
        out_proposals[count].kps[k * 2 + 1] = cy + kps_ptr[(kc + 1) * plane + spatial] * feat_stride;
    }
}

} // namespace

void scrfd_preprocess_gpu(
    const uint8_t* src, int src_w, int src_h, int src_step,
    float* dst, int dst_w, int dst_h,
    GPUPreprocess p_type,
    float map_scale, float map_wpad, float map_hpad, bool use_wpad_map,
    cudaStream_t stream) {
    
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    
    scrfd_preprocess_kernel<<<grid, block, 0, stream>>>(
        src, src_w, src_h, src_step, dst, dst_w, dst_h,
        p_type, map_scale, map_wpad, map_hpad, use_wpad_map);
}

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
    cudaStream_t stream) {
    
    cudaMemsetAsync(out_count, 0, sizeof(int), stream);

    auto launch = [&](const float* s, const float* b, const float* k, int n, int spatial_side, int stride,
                      const float* anchors, int n_anchors) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scrfd_postprocess_stride_kernel<<<blocks, threads, 0, stream>>>(
            s, b, k, n, spatial_side, stride, anchors, n_anchors, prob_threshold, use_sigmoid,
            out_proposals, out_count, max_proposals);
    };

    launch(score_8, bbox_8, kps_8, output_size_8, feat_hw_8, 8, base_anchors_8, num_anchors_8);
    launch(score_16, bbox_16, kps_16, output_size_16, feat_hw_16, 16, base_anchors_16, num_anchors_16);
    launch(score_32, bbox_32, kps_32, output_size_32, feat_hw_32, 32, base_anchors_32, num_anchors_32);
}
