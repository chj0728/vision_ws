#include <cuda_runtime.h>
#include <cstdint>
#include "yolo_gpu_preprocess.h"

namespace {

__device__ void sample_bgr_norm(const uint8_t* src, int w, int h, float sx, float sy, float& out_r, float& out_g,
                                float& out_b) {
    sx = fminf(fmaxf(sx, 0.f), static_cast<float>(w - 1));
    sy = fminf(fmaxf(sy, 0.f), static_cast<float>(h - 1));
    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    int x1 = x0 + 1;
    if (x1 >= w) x1 = w - 1;
    int y1 = y0 + 1;
    if (y1 >= h) y1 = h - 1;
    float dx = sx - static_cast<float>(x0);
    float dy = sy - static_cast<float>(y0);
    auto pix = [&](int x, int y) {
        const uint8_t* p = src + (y * w + x) * 3;
        return p;
    };
    const uint8_t* p00 = pix(x0, y0);
    const uint8_t* p10 = pix(x1, y0);
    const uint8_t* p01 = pix(x0, y1);
    const uint8_t* p11 = pix(x1, y1);
    auto lerp2 = [](float a, float b, float t) { return a * (1.f - t) + b * t; };
    float b0 = lerp2(static_cast<float>(p00[0]), static_cast<float>(p10[0]), dx);
    float b1 = lerp2(static_cast<float>(p01[0]), static_cast<float>(p11[0]), dx);
    float B = lerp2(b0, b1, dy) * (1.f / 255.f);
    float g0 = lerp2(static_cast<float>(p00[1]), static_cast<float>(p10[1]), dx);
    float g1 = lerp2(static_cast<float>(p01[1]), static_cast<float>(p11[1]), dx);
    float G = lerp2(g0, g1, dy) * (1.f / 255.f);
    float r0 = lerp2(static_cast<float>(p00[2]), static_cast<float>(p10[2]), dx);
    float r1 = lerp2(static_cast<float>(p01[2]), static_cast<float>(p11[2]), dx);
    float R = lerp2(r0, r1, dy) * (1.f / 255.f);
    out_r = R;
    out_g = G;
    out_b = B;
}

__global__ void letterbox_to_nchw_kernel(const uint8_t* __restrict__ src, int sw, int sh,
                                           float* __restrict__ dst, int dw, int dh, int new_w, int new_h,
                                           int pad_left, int pad_top) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh) return;

    const int plane = dh * dw;
    const int idx = oy * dw + ox;
    constexpr float kPad = 114.f / 255.f;

    if (ox < pad_left || ox >= pad_left + new_w || oy < pad_top || oy >= pad_top + new_h) {
        dst[idx] = kPad;
        dst[plane + idx] = kPad;
        dst[2 * plane + idx] = kPad;
        return;
    }

    const float rx = static_cast<float>(ox - pad_left);
    const float ry = static_cast<float>(oy - pad_top);
    const float sx = (rx + 0.5f) * (static_cast<float>(sw) / static_cast<float>(new_w)) - 0.5f;
    const float sy = (ry + 0.5f) * (static_cast<float>(sh) / static_cast<float>(new_h)) - 0.5f;

    float R, G, B;
    sample_bgr_norm(src, sw, sh, sx, sy, R, G, B);
    dst[idx] = R;
    dst[plane + idx] = G;
    dst[2 * plane + idx] = B;
}

__global__ void resize_to_nchw_kernel(const uint8_t* __restrict__ src, int sw, int sh, float* __restrict__ dst,
                                      int dw, int dh) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh) return;

    const int plane = dh * dw;
    const int idx = oy * dw + ox;
    const float sx = (static_cast<float>(ox) + 0.5f) * (static_cast<float>(sw) / static_cast<float>(dw)) - 0.5f;
    const float sy = (static_cast<float>(oy) + 0.5f) * (static_cast<float>(sh) / static_cast<float>(dh)) - 0.5f;

    float R, G, B;
    sample_bgr_norm(src, sw, sh, sx, sy, R, G, B);
    dst[idx] = R;
    dst[plane + idx] = G;
    dst[2 * plane + idx] = B;
}

__global__ void yolo_postprocess_kernel(
    const float* d_output, int num_anchors, int num_classes, float conf_threshold,
    bool is_yolo26, bool channels_first,
    int orig_w, int orig_h, float scale, int pad_left, int pad_top,
    YoloProposal* d_proposals, int* d_count, int max_proposals) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_anchors) return;

    float score = 0;
    int cls = 0;
    float x0, y0, x1, y1;

    if (is_yolo26) {
        const int stride = 6;
        float row[6];
        if (channels_first) {
            for (int j = 0; j < stride; j++) row[j] = d_output[idx + j * num_anchors];
        } else {
            for (int j = 0; j < stride; j++) row[j] = d_output[idx * stride + j];
        }
        score = row[4];
        if (score < conf_threshold) return;
        x0 = row[0]; y0 = row[1]; x1 = row[2]; y1 = row[3];
        cls = (int)roundf(row[5]);
    } else {
        // YOLOv8
        int stride = 4 + num_classes;
        if (channels_first) {
            score = 0;
            for (int c = 0; c < num_classes; c++) {
                float s = d_output[idx + (4 + c) * num_anchors];
                if (s > score) { score = s; cls = c; }
            }
            if (score < conf_threshold) return;
            float xc = d_output[idx + 0 * num_anchors];
            float yc = d_output[idx + 1 * num_anchors];
            float w  = d_output[idx + 2 * num_anchors];
            float h  = d_output[idx + 3 * num_anchors];
            x0 = xc - w/2; y0 = yc - h/2; x1 = xc + w/2; y1 = yc + h/2;
        } else {
            const float* row = d_output + idx * stride;
            score = 0;
            for (int c = 0; c < num_classes; c++) {
                if (row[4+c] > score) { score = row[4+c]; cls = c; }
            }
            if (score < conf_threshold) return;
            float xc = row[0], yc = row[1], w = row[2], h = row[3];
            x0 = xc - w/2; y0 = yc - h/2; x1 = xc + w/2; y1 = yc + h/2;
        }
    }

    int count = atomicAdd(d_count, 1);
    if (count >= max_proposals) return;

    d_proposals[count].x0 = (x0 - pad_left) / scale;
    d_proposals[count].y0 = (y0 - pad_top) / scale;
    d_proposals[count].x1 = (x1 - pad_left) / scale;
    d_proposals[count].y1 = (y1 - pad_top) / scale;
    d_proposals[count].score = score;
    d_proposals[count].cls = cls;
    d_proposals[count].depth_dist = -1.0f;
}

__global__ void yolo_depth_sampling_kernel(
    const float* d_depth, int dw, int dh,
    YoloProposal* d_proposals, int count,
    float min_depth, float max_depth,
    float roi_y0_n, float roi_y1_n, float x_margin_n,
    float trim_close_ratio, float percentile,
    float depth_scale_x, float depth_scale_y) {
    
    int p_idx = blockIdx.x;
    if (p_idx >= count) return;

    YoloProposal& prop = d_proposals[p_idx];
    // 将彩色图坐标系下的 bbox 缩放到深度图坐标系
    float bx = prop.x0 * depth_scale_x;
    float by = prop.y0 * depth_scale_y;
    float bw = (prop.x1 - prop.x0) * depth_scale_x;
    float bh = (prop.y1 - prop.y0) * depth_scale_y;

    if (bw <= 1 || bh <= 1) return;

    int x0 = (int)(bx + bw * x_margin_n);
    int x1 = (int)(bx + bw * (1.0f - x_margin_n));
    int y0 = (int)(by + bh * roi_y0_n);
    int y1 = (int)(by + bh * roi_y1_n);

    // Clamp
    x0 = max(0, min(dw-1, x0)); x1 = max(0, min(dw-1, x1));
    y0 = max(0, min(dh-1, y0)); y1 = max(0, min(dh-1, y1));

    if (x1 <= x0 || y1 <= y0) return;

    // GPU sampling: use a small fixed number of samples to avoid complex sorting
    // For Orin, we can use a small array in local memory and bitonic sort or just a simple selection
    const int MAX_SAMPLES = 64;
    float samples[MAX_SAMPLES];
    int n_samples = 0;

    int step_x = max(1, (x1 - x0) / 8);
    int step_y = max(1, (y1 - y0) / 8);

    for (int vy = y0; vy < y1 && n_samples < MAX_SAMPLES; vy += step_y) {
        for (int vx = x0; vx < x1 && n_samples < MAX_SAMPLES; vx += step_x) {
            float z = d_depth[vy * dw + vx];
            if (isfinite(z) && z >= min_depth && z <= max_depth) {
                samples[n_samples++] = z;
            }
        }
    }

    if (n_samples < 3) return;

    // Simple bubble sort for small N
    for (int m = 0; m < n_samples; m++) {
        for (int n = m + 1; n < n_samples; n++) {
            if (samples[m] > samples[n]) {
                float tmp = samples[m];
                samples[m] = samples[n];
                samples[n] = tmp;
            }
        }
    }

    int k = (int)(n_samples * trim_close_ratio);
    int m = n_samples - k;
    if (m <= 0) return;
    int j = (int)(m * percentile);
    if (j >= m) j = m - 1;
    
    prop.depth_dist = samples[k + j];
}

} // namespace

extern "C" void yolo_launch_letterbox_to_nchw(const uint8_t* d_bgr, int sw, int sh, float* d_nchw, int dw, int dh,
                                              int new_w, int new_h, int pad_left, int pad_top, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dw + block.x - 1) / block.x, (dh + block.y - 1) / block.y);
    letterbox_to_nchw_kernel<<<grid, block, 0, stream>>>(d_bgr, sw, sh, d_nchw, dw, dh, new_w, new_h, pad_left,
                                                         pad_top);
}

extern "C" void yolo_launch_resize_to_nchw(const uint8_t* d_bgr, int sw, int sh, float* d_nchw, int dw, int dh,
                                           cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dw + block.x - 1) / block.x, (dh + block.y - 1) / block.y);
    resize_to_nchw_kernel<<<grid, block, 0, stream>>>(d_bgr, sw, sh, d_nchw, dw, dh);
}

extern "C" void yolo_postprocess_gpu(
    const float* d_output, int num_anchors, int num_classes, float conf_threshold,
    bool is_yolo26, bool channels_first,
    int orig_w, int orig_h, float scale, int pad_left, int pad_top,
    YoloProposal* d_proposals, int* d_count, int max_proposals,
    cudaStream_t stream) {
    
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    int threads = 256;
    int blocks = (num_anchors + threads - 1) / threads;
    yolo_postprocess_kernel<<<blocks, threads, 0, stream>>>(
        d_output, num_anchors, num_classes, conf_threshold, is_yolo26, channels_first,
        orig_w, orig_h, scale, pad_left, pad_top, d_proposals, d_count, max_proposals);
}

extern "C" void yolo_depth_sampling_gpu(
    const float* d_depth, int depth_w, int depth_h,
    YoloProposal* d_proposals, int count,
    float min_depth, float max_depth,
    float roi_y0_n, float roi_y1_n, float x_margin_n,
    float trim_close_ratio, float percentile,
    float depth_scale_x, float depth_scale_y,
    cudaStream_t stream) {
    
    if (count <= 0) return;
    yolo_depth_sampling_kernel<<<count, 1, 0, stream>>>(
        d_depth, depth_w, depth_h, d_proposals, count, min_depth, max_depth,
        roi_y0_n, roi_y1_n, x_margin_n, trim_close_ratio, percentile,
        depth_scale_x, depth_scale_y);
}
