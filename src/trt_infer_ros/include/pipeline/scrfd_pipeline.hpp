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
   * @param bgr 当前帧 BGR 彩色图像。
   * @param perception_result 输入人体检测结果，并写入人脸框和置信度。
   * @param frame_context 写入全图坐标下的人脸框和五点关键点。
   */
  void process(const cv::Mat &bgr,
               trt_infer_msgs::msg::PerceptionResult &perception_result,
               PerceptionFrameContext &frame_context);

  /** @brief 返回当前模块是否启用。 */
  bool isEnabled() const { return enabled_; }

  /** @brief 返回最终解析得到的 SCRFD 引擎路径。 */
  std::string getEnginePath() const { return engine_path_; }

private:
  /** @brief 根据裁后人体框构建头肩区域，并裁到彩色图边界内。 */
  cv::Rect buildHeadShoulderRoi(const trt_infer_msgs::msg::BoundingBox &body,
                                const cv::Size &image_size) const;

  /**
   * @brief 检测 ROI 内最高置信度人脸，无候选时返回 false。
   * @param roi_face 输出 ROI 局部坐标下的人脸框、五点关键点和置信度。
   */
  bool detectBestFaceInRoi(const cv::Mat &bgr, const cv::Rect &head_roi,
                           FaceObject &roi_face);

  /** @brief 转为全图坐标；裁后框有效时写回消息和上下文，否则保持原值。 */
  static void writeFaceResult(const FaceObject &roi_face,
                              const cv::Rect &head_roi,
                              const cv::Size &image_size,
                              trt_infer_msgs::msg::PersonMeta &person,
                              PersonFrameContext &person_context);

  std::unique_ptr<SCRFD_TRT> face_detector_; // SCRFD TensorRT 人脸检测器

  bool enabled_{false};                    // 是否启用人脸检测
  std::string engine_filename_;            // 未指定路径时使用的引擎文件名
  std::string engine_path_;                // 根据工作空间根目录解析后的引擎路径
  std::string preprocess_mode_;            // 转为小写的图像预处理模式
  float face_confidence_threshold_{0.38f}; // 人脸候选置信度阈值
  float face_nms_iou_threshold_{0.45f};    // 人脸候选 NMS 的 IoU 阈值
  double head_roi_height_ratio_{0.58};     // 基础 ROI 高度占裁后人体高度的比例
  double head_roi_side_padding_ratio_{0.24};  // 左右各扩展的裁后人体宽度比例
  double head_roi_top_expansion_ratio_{0.14}; // 向上扩展的裁后人体高度比例
  int min_head_roi_side_px_{48};   // 头肩检测区域最小边长，不是检测出的人脸尺寸
  int max_person_roi_attempts_{8}; // 单帧尝试人数上限，无效或过小 ROI 也计数
};
