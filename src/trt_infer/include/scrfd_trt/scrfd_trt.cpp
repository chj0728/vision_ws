#include "scrfd_trt.h"
#include "scrfd_gpu_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <opencv2/imgcodecs.hpp>

using namespace nvinfer1;

namespace {

inline float sigmoid_f(float x) {
    x = std::min(50.f, std::max(-50.f, x));
    return 1.f / (1.f + std::exp(-x));
}

}  // namespace

void qsort_descent_inplace(std::vector<FaceObject> &face_objects, int left, int right) {
    int i = left;
    int j = right;
    float p = face_objects[(left + right) / 2].prob;

    while (i <= j) {
        while (face_objects[i].prob > p) {
            i++;
        }

        while (face_objects[j].prob < p) {
            j--;
        }

        if (i <= j) {
            std::swap(face_objects[i], face_objects[j]);
            i++;
            j--;
        }
    }

    if (left < j) {
        qsort_descent_inplace(face_objects, left, j);
    }

    if (i < right) {
        qsort_descent_inplace(face_objects, i, right);
    }
}

void qsort_descent_inplace(std::vector<FaceObject> &face_objects) {
    if (face_objects.empty()) {
        return;
    }

    qsort_descent_inplace(face_objects, 0, face_objects.size() - 1);
}

float intersection_area(const FaceObject& a, const FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void nms_sorted_bboxes(const std::vector<FaceObject> &face_objects, std::vector<int> &picked,
                              float nms_threshold) {
    picked.clear();

    const int n = face_objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = face_objects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const FaceObject &a = face_objects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const FaceObject &b = face_objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

int SCRFD_TRT::draw(cv::Mat& rgb, const std::vector<FaceObject>& faceobjects)
{
    for (size_t i = 0; i < faceobjects.size(); i++)
    {
        const FaceObject& obj = faceobjects[i];
        cv::rectangle(rgb, obj.rect, cv::Scalar(0, 255, 0));

        cv::circle(rgb, obj.landmark[0], 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(rgb, obj.landmark[1], 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(rgb, obj.landmark[2], 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(rgb, obj.landmark[3], 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(rgb, obj.landmark[4], 2, cv::Scalar(255, 255, 0), -1);

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    return 0;
}

SCRFD_TRT::SCRFD_TRT(const std::string trt_path) {
    initEngine(trt_path);
    initAnchors();

    // Init GPU buffers
    max_src_img_size = 1920 * 1080 * 3;
    CHECK(cudaMalloc(&d_src_img, max_src_img_size));
    
    // Allocate all engine buffers on GPU once
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        auto dims = engine->getTensorShape(name);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; j++) size *= dims.d[j];
        
        void* ptr = nullptr;
        CHECK(cudaMalloc(&ptr, size * sizeof(float)));
        context->setTensorAddress(name, ptr);
    }

    CHECK(cudaMalloc(&d_base_anchors_8, base_anchors_8.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_base_anchors_16, base_anchors_16.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_base_anchors_32, base_anchors_32.size() * sizeof(float)));
    
    CHECK(cudaMemcpy(d_base_anchors_8, base_anchors_8.data(), base_anchors_8.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_base_anchors_16, base_anchors_16.data(), base_anchors_16.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_base_anchors_32, base_anchors_32.data(), base_anchors_32.size() * sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&d_out_proposals, MAX_PROPOSALS * sizeof(GPUFaceProposal)));
    CHECK(cudaMalloc(&d_out_count, sizeof(int)));
}

void SCRFD_TRT::setPreprocess(Preprocess p) {
    preprocess_ = p;
    score_policy_done_ = false;
    score_use_sigmoid_ = false;
}

SCRFD_TRT::~SCRFD_TRT() {
    cudaFree(d_src_img);
    cudaFree(d_base_anchors_8);
    cudaFree(d_base_anchors_16);
    cudaFree(d_base_anchors_32);
    cudaFree(d_out_proposals);
    cudaFree(d_out_count);

    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        void* ptr = const_cast<void*>(context->getTensorAddress(engine->getIOTensorName(i)));
        cudaFree(ptr);
    }

    delete context;
    delete engine;
    delete runtime;
}

void SCRFD_TRT::initEngine(const std::string trt_path) {
    std::cout << "init scrfd_trt from " << trt_path << "\n";
    std::ifstream file(trt_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("SCRFD: Cannot open engine file at " + trt_path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("SCRFD: Engine file is empty or size invalid: " + trt_path);
    }
    file.seekg(0, std::ios::beg);
    
    std::vector<char> trtModelStream(size);
    file.read(trtModelStream.data(), size);
    file.close();

    runtime = createInferRuntime(gLogger);
    if (!runtime) throw std::runtime_error("SCRFD: createInferRuntime failed");
    
    engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
    if (!engine) {
        throw std::runtime_error("SCRFD: deserializeCudaEngine failed. Check if the engine was built for this GPU/TRT version.");
    }
    
    context = engine->createExecutionContext();
    if (!context) throw std::runtime_error("SCRFD: createExecutionContext failed");
}

void SCRFD_TRT::initAnchors() {
    auto gen = [&](int base_size, float ratios[], float scales[], std::vector<float>& out) {
        float** anchors = generate_anchors(base_size, ratios, scales);
        int n = 1 * 2; // ratios[1] * scales[2]
        out.clear();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 4; j++) out.push_back(anchors[i][j]);
            delete[] anchors[i];
        }
        delete[] anchors;
    };

    float ratios[] = {1.0};
    float scales[] = {1.0, 2.0};
    gen(16, ratios, scales, base_anchors_8);
    num_anchors_stride_8 = 2;
    gen(64, ratios, scales, base_anchors_16);
    num_anchors_stride_16 = 2;
    gen(256, ratios, scales, base_anchors_32);
    num_anchors_stride_32 = 2;
}

// Dummy as we now do inference directly in detect
void SCRFD_TRT::doInference(IExecutionContext&, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, int) {}

void SCRFD_TRT::updateScorePolicy(const float* s8, size_t n8, const float* s16, size_t n16, const float* s32, size_t n32) {
    if (score_policy_done_) return;
    float mn = s8[0], mx = s8[0];
    auto eat = [&](const float* p, size_t n) {
        for (size_t k = 0; k < n; ++k) {
            mn = std::min(mn, p[k]);
            mx = std::max(mx, p[k]);
        }
    };
    eat(s8, n8); eat(s16, n16); eat(s32, n32);
    score_use_sigmoid_ = (mn < -0.01f || mx > 1.01f);
    score_policy_done_ = true;
}

int SCRFD_TRT::detect(cv::Mat image, std::vector<FaceObject>& faceobjects, float prob_threshold, float nms_threshold) {
    if (image.empty()) return 0;

    /** ROI 子图常为父图上的非连续视图；用 total()*elemSize() 从 data 一次 memcpy 会错位，SCRFD 输入全废 → 脸恒为 0 */
    cv::Mat src_storage;
    const cv::Mat* psrc = &image;
    if (!image.isContinuous()) {
        src_storage = image.clone();
        psrc = &src_storage;
    }
    const cv::Mat& src = *psrc;
    const int src_w = src.cols;
    const int src_h = src.rows;
    const int packed_step = src_w * static_cast<int>(src.elemSize());
    const size_t packed_bytes = static_cast<size_t>(packed_step) * static_cast<size_t>(src_h);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 1. 将连续 packed BGR 拷到 GPU（与 scrfd_preprocess 内步长一致）
    if (packed_bytes > (size_t)max_src_img_size) {
        cudaFree(d_src_img);
        max_src_img_size = static_cast<int>(packed_bytes);
        cudaMalloc(&d_src_img, max_src_img_size);
    }
    CHECK(cudaMemcpyAsync(d_src_img, src.data, packed_bytes, cudaMemcpyHostToDevice, stream));

    // 2. Preprocess on GPU
    float map_scale = 1.f;
    float map_wpad = 0.f;
    float map_hpad = 0.f;
    bool use_wpad_map = false;
    GPUPreprocess p_type = (preprocess_ == Preprocess::InsightFacePython) ? GPUPreprocess::InsightFacePython : GPUPreprocess::NamdvtUpstream;

    if (preprocess_ == Preprocess::InsightFacePython) {
        const float im_ratio = static_cast<float>(src_h) / static_cast<float>(src_w);
        const float model_ratio = static_cast<float>(INPUT_H) / static_cast<float>(INPUT_W);
        if (im_ratio > model_ratio) map_scale = static_cast<float>(INPUT_H) / static_cast<float>(src_h);
        else map_scale = static_cast<float>(INPUT_W) / static_cast<float>(src_w);
    } else {
        map_scale = static_cast<float>(INPUT_H) / static_cast<float>(std::max(src_w, src_h));
        map_wpad = INPUT_W - static_cast<float>(src_w) * map_scale;
        map_hpad = INPUT_H - static_cast<float>(src_h) * map_scale;
        use_wpad_map = true;
    }

    float* d_input = (float*)const_cast<void*>(context->getTensorAddress(INPUT_BLOB_NAME));
    scrfd_preprocess_gpu((uint8_t*)d_src_img, src_w, src_h, packed_step, d_input, INPUT_W, INPUT_H, p_type, map_scale, map_wpad,
                         map_hpad, use_wpad_map, stream);

    // 3. Inference
    if (!context->enqueueV3(stream)) {
        std::fprintf(stderr, "[SCRFD_TRT] enqueueV3 failed (TensorRT)\n");
        cudaStreamDestroy(stream);
        return 0;
    }
    CHECK(cudaStreamSynchronize(stream));

    // 4. Postprocess on GPU
    float* d_s8 = (float*)const_cast<void*>(context->getTensorAddress(OUTPUT_SCORE_8_BLOB_NAME));
    float* d_b8 = (float*)const_cast<void*>(context->getTensorAddress(OUTPUT_BBOX_8_BLOB_NAME));
    float* d_k8 = (float*)const_cast<void*>(context->getTensorAddress(OUTPUT_KPS_8_BLOB_NAME));
    float* d_s16 = (float*)const_cast<void*>(context->getTensorAddress(OUTPUT_SCORE_16_BLOB_NAME));
    float* d_b16 = (float*)const_cast<void*>(context->getTensorAddress(OUTPUT_BBOX_16_BLOB_NAME));
    float* d_k16 = (float*)const_cast<void*>(context->getTensorAddress(OUTPUT_KPS_16_BLOB_NAME));
    float* d_s32 = (float*)const_cast<void*>(context->getTensorAddress(OUTPUT_SCORE_32_BLOB_NAME));
    float* d_b32 = (float*)const_cast<void*>(context->getTensorAddress(OUTPUT_BBOX_32_BLOB_NAME));
    float* d_k32 = (float*)const_cast<void*>(context->getTensorAddress(OUTPUT_KPS_32_BLOB_NAME));

    // 与 insightface SCRFD 一致：用整幅 score 张量判断 logits 还是已 sigmoid（勿只看 score[0]）
    if (!score_policy_done_) {
        std::vector<float> h_s8(OUTPUT_SCORE_8_SIZE);
        std::vector<float> h_s16(OUTPUT_SCORE_16_SIZE);
        std::vector<float> h_s32(OUTPUT_SCORE_32_SIZE);
        cudaMemcpyAsync(h_s8.data(), d_s8, OUTPUT_SCORE_8_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_s16.data(), d_s16, OUTPUT_SCORE_16_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_s32.data(), d_s32, OUTPUT_SCORE_32_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        updateScorePolicy(h_s8.data(), h_s8.size(), h_s16.data(), h_s16.size(), h_s32.data(), h_s32.size());
    }

    const int kFeat8 = INPUT_W / 8;
    const int kFeat16 = INPUT_W / 16;
    const int kFeat32 = INPUT_W / 32;
    scrfd_postprocess_gpu(d_s8, d_b8, d_k8, d_s16, d_b16, d_k16, d_s32, d_b32, d_k32,
        OUTPUT_SCORE_8_SIZE, OUTPUT_SCORE_16_SIZE, OUTPUT_SCORE_32_SIZE, kFeat8, kFeat16, kFeat32,
        d_base_anchors_8, num_anchors_stride_8, d_base_anchors_16, num_anchors_stride_16, d_base_anchors_32, num_anchors_stride_32,
        prob_threshold, score_use_sigmoid_, (GPUFaceProposal*)d_out_proposals, d_out_count, MAX_PROPOSALS, stream);

    int h_count = 0;
    CHECK(cudaMemcpyAsync(&h_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
    
    std::vector<GPUFaceProposal> h_proposals(std::min(h_count, MAX_PROPOSALS));
    if (h_count > 0) {
        CHECK(cudaMemcpyAsync(h_proposals.data(), d_out_proposals, h_proposals.size() * sizeof(GPUFaceProposal), cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // 5. Final NMS on CPU
    std::vector<FaceObject> proposals;
    for (auto& hp : h_proposals) {
        FaceObject obj;
        obj.prob = hp.prob;
        float x0, y0, x1, y1;
        if (use_wpad_map) {
            x0 = (hp.x0 - map_wpad) / map_scale;
            y0 = (hp.y0 - map_hpad) / map_scale;
            x1 = (hp.x1 - map_wpad) / map_scale;
            y1 = (hp.y1 - map_hpad) / map_scale;
        } else {
            x0 = hp.x0 / map_scale;
            y0 = hp.y0 / map_scale;
            x1 = hp.x1 / map_scale;
            y1 = hp.y1 / map_scale;
        }
        obj.rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
        for (int k = 0; k < 5; k++) {
            if (use_wpad_map) {
                obj.landmark[k].x = (hp.kps[k*2] - map_wpad) / map_scale;
                obj.landmark[k].y = (hp.kps[k*2+1] - map_hpad) / map_scale;
            } else {
                obj.landmark[k].x = hp.kps[k*2] / map_scale;
                obj.landmark[k].y = hp.kps[k*2+1] / map_scale;
            }
        }
        proposals.push_back(obj);
    }

    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    faceobjects.clear();
    for (int idx : picked) faceobjects.push_back(proposals[idx]);

    return 0;
}

float** SCRFD_TRT::generate_anchors(int base_size, const float ratios[], const float scales[]) {
    int num_ratio = 1;
    int num_scale = 2;
    float** anchors = new float*[num_ratio * num_scale];
    for (int i = 0; i < num_ratio; i++) {
        for (int j = 0; j < num_scale; j++) {
            float rs_w = base_size * scales[j];
            float rs_h = base_size * scales[j];
            anchors[i * num_scale + j] = new float[4];
            anchors[i * num_scale + j][0] = -rs_w * 0.5f;
            anchors[i * num_scale + j][1] = -rs_h * 0.5f;
            anchors[i * num_scale + j][2] = rs_w * 0.5f;
            anchors[i * num_scale + j][3] = rs_h * 0.5f;
        }
    }
    return anchors;
}

void SCRFD_TRT::generate_proposals(float**, int, int, float*, float*, float*, float, std::vector<FaceObject>&, int) {}
