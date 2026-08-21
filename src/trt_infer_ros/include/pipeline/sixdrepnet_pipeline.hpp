#pragma once

#include "pipeline/perception_frame_context.hpp"
#include "sixdrepnet_trt/sixdrepnet_trt.h"

#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "trt_infer_msgs/msg/head_pose.hpp"
#include "trt_infer_msgs/msg/perception_result.hpp"

/**
 * @brief 基于 SCRFD 人脸框执行 SixDRepNet 头部姿态估计。
 *
 * 该模块从帧上下文读取全图坐标下的人脸框，扩展并裁剪人脸区域，
 * 将估计得到的偏航角、俯仰角和翻滚角写入 HeadPose 消息。
 */
class SixDRepNetPipeline {
public:
  /**
   * @brief 创建 SixDRepNet Pipeline，并完成参数加载和模型初始化。
   *
   * @param config 完整的 Pipeline YAML 配置节点。
   */
  explicit SixDRepNetPipeline(const YAML::Node &config);
  ~SixDRepNetPipeline();

  /**
   * @brief 从 sixdrepnet_pipeline 配置节点读取模型和人脸区域参数。
   *
   * @param config 完整的 Pipeline YAML 配置节点。
   */
  void loadParameters(const YAML::Node &config);

  /** @brief 根据配置加载 SixDRepNet TensorRT 引擎。 */
  void initialize();

  /**
   * @brief 对当前帧有效人脸执行头部姿态估计。
   *
   * @param rgb 当前帧 BGR 彩色图像。
   * @param frame_context 当前帧 SCRFD 输出的全图人脸框。
   * @param perception_result 写入每个人的 HeadPose 消息。
   */
  void process(const cv::Mat &rgb, const PerceptionFrameContext &frame_context,
               trt_infer_msgs::msg::PerceptionResult &perception_result);

  /** @brief 返回当前模块是否启用。 */
  bool isEnabled() const { return enabled_; }

  /** @brief 返回最终解析得到的 SixDRepNet 引擎路径。 */
  std::string getEnginePath() const { return sixdrepnet_engine_path_; }

private:
  std::unique_ptr<SixDRepNet_TRT>
      sixdrepnet_engine_ptr_; // SixDRepNet TensorRT 推理实例

  bool enabled_{false};                // 是否启用头部姿态估计
  std::string sixdrepnet_engine_name_; // 默认引擎文件名
  std::string sixdrepnet_engine_path_; // 解析后的引擎绝对路径
  int min_face_px_{36};                // 执行推理所需的最小人脸边长
  float face_expand_ratio_{0.12f};     // 人脸框四周扩展比例
};
