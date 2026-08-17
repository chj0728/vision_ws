/**
 * @file perception_pipeline.hpp
 * @brief Perception pipeline header file.
 * @author Cao Haojie
 * @date 2024-06-20
 */

#ifndef PERCEPTION_PIPELINE_HPP
#define PERCEPTION_PIPELINE_HPP

#include <memory>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "pipeline/yolo_pipeline.hpp"

#include "trt_infer_msgs/msg/perception_result.hpp"
using trt_infer_msgs::msg::PerceptionResult;

class PerceptionPipeline {

private:
    YAML::Node config_;
    std::unique_ptr<YOLOPipeline> yolo_pipeline_ptr_;

public:
    PerceptionPipeline(YAML::Node & config);
    ~PerceptionPipeline();

    void initialize();
    // void run();

    /**
     * @brief 处理RGB和深度图像，生成感知结果
     * 
     * @param rgb 
     * @param depth 
     * @param perception_result 
     */
    void process(
        const cv::Mat& rgb,
        const cv::Mat& depth,
        PerceptionResult& perception_result);
};

#endif // PERCEPTION_PIPELINE_HPP