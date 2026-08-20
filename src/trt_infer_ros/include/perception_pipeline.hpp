/**
 * @file perception_pipeline.hpp
 * @brief 统一编排人体检测、追踪、人脸检测和人脸识别模块。
 */

#ifndef PERCEPTION_PIPELINE_HPP
#define PERCEPTION_PIPELINE_HPP

#include <memory>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "pipeline/arcface_pipeline.hpp"
#include "pipeline/iou_tracker.hpp"
#include "pipeline/scrfd_pipeline.hpp"
#include "pipeline/yolo_pipeline.hpp"

#include "trt_infer_msgs/msg/perception_result.hpp"
using trt_infer_msgs::msg::PerceptionResult;

/** @brief 感知算法总调度器，不负责 ROS 订阅和发布。 */
class PerceptionPipeline {

private:
  YAML::Node config_;                               // 完整的 Pipeline YAML 配置
  std::unique_ptr<YOLOPipeline> yolo_pipeline_ptr_; // 人体检测与距离估计模块
  std::unique_ptr<IouTracker> iou_tracker_ptr_;     // 人体框时序追踪模块
  std::unique_ptr<SCRFDPipeline> scrfd_pipeline_ptr_;     // 人脸检测模块
  std::unique_ptr<ArcFacePipeline> arcface_pipeline_ptr_; // 人脸识别模块
  int frame_number_{0}; // Pipeline 已处理的帧编号

public:
  /**
   * @brief 创建总 Pipeline 并初始化所有子模块。
   *
   * @param config 完整的 Pipeline YAML 配置节点。
   */
  PerceptionPipeline(YAML::Node &config);
  ~PerceptionPipeline();

  /**
   * @brief 按数据依赖顺序初始化所有感知子模块。
   */
  void initialize();

  /**
   * @brief 处理一组 RGB-D 图像并生成统一感知结果。
   *
   * 调用顺序为 YOLO、IoU Tracker、SCRFD、ArcFace。
   *
   * @param rgb 当前帧 BGR 彩色图像。
   * @param depth 当前帧以米为单位的浮点深度图像。
   * @param perception_result 输出人体、轨迹、人脸和身份识别结果。
   */
  void process(const cv::Mat &rgb, const cv::Mat &depth,
               PerceptionResult &perception_result);
};

#endif // PERCEPTION_PIPELINE_HPP