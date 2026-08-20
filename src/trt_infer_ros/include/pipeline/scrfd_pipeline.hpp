#pragma once

#include "pipeline/perception_frame_context.hpp"
#include "scrfd_trt/scrfd_trt.h"

#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "trt_infer_msgs/msg/perception_result.hpp"

/**
 * @brief 基于人体头肩区域执行 SCRFD 人脸检测。
 *
 * 该模块读取 YOLO 输出的人体框，将人脸框写入 ROS 消息，并通过帧上下文
 * 保留 ArcFace 所需的五点关键点。
 */
class SCRFDPipeline {
public:
  /**
   * @brief 创建 SCRFD Pipeline，并完成参数加载和模型初始化。
   *
   * @param config 完整的 Pipeline YAML 配置节点。
   */
  explicit SCRFDPipeline(const YAML::Node &config);
  ~SCRFDPipeline();

  /**
   * @brief 从 scrfd_pipeline 配置节点读取模型和检测参数。
   *
   * @param config 完整的 Pipeline YAML 配置节点。
   */
  void loadParameters(const YAML::Node &config);

  /**
   * @brief 根据配置加载 SCRFD TensorRT 引擎。
   */
  void initialize();

  /**
   * @brief 在每个人体头肩区域内检测人脸。
   *
   * @param rgb 当前帧 BGR 彩色图像。
   * @param perception_result 输入人体检测结果，并写入人脸框和置信度。
   * @param frame_context 写入全图坐标下的人脸框和五点关键点。
   */
  void process(const cv::Mat &rgb,
               trt_infer_msgs::msg::PerceptionResult &perception_result,
               PerceptionFrameContext &frame_context);

  /** @brief 返回当前模块是否启用。 */
  bool isEnabled() const { return enabled_; }

  /** @brief 返回最终解析得到的 SCRFD 引擎路径。 */
  std::string getEnginePath() const { return scrfd_engine_path_; }

private:
  std::unique_ptr<SCRFD_TRT> scrfd_engine_ptr_; // SCRFD TensorRT 推理实例

  bool enabled_{false};                       // 是否启用人脸检测
  std::string scrfd_engine_name_;             // 默认引擎文件名
  std::string scrfd_engine_path_;             // 解析后的引擎绝对路径
  std::string preprocess_;                    // SCRFD 图像预处理模式
  float prob_threshold_{0.38f};               // 人脸候选置信度阈值
  float nms_threshold_{0.45f};                // 人脸框 NMS IoU 阈值
  double person_head_height_ratio_{0.58};     // 头肩区域占人体框高度的比例
  double person_head_width_pad_ratio_{0.24};  // 头肩区域左右扩展比例
  double person_head_top_expand_ratio_{0.14}; // 人体框顶部向上扩展比例
  int face_roi_min_side_{48}; // 执行检测所需的最小 ROI 边长，单位为像素
  int max_person_rois_{8};    // 单帧最多处理的人体 ROI 数量
};
