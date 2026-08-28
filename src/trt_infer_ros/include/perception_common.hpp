#ifndef PERCEPTION_COMMON_HPP
#define PERCEPTION_COMMON_HPP

#include <algorithm>
#include <cctype>
#include <string>

#include <opencv2/opencv.hpp>

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

struct CompressedDepthHeader {
  int32_t format;
  float depth_quant_a;
  float depth_quant_b;
};

static_assert(sizeof(CompressedDepthHeader) == 12);

/**
 * @brief 将字符串转换为小写
 *
 * @param value
 * @return std::string
 */
inline std::string toLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

/**
 * @brief 检查图像消息的编码是否为JPEG格式
 *
 * @param encoding
 * @return true
 * @return false
 */
inline bool isJpegInImageMsg(const std::string &encoding) {
  const std::string normalized = toLower(encoding);
  return normalized.find("jpeg") != std::string::npos ||
         normalized.find("jpg") != std::string::npos ||
         normalized.find("mjpeg") != std::string::npos ||
         normalized.find("mjpg") != std::string::npos;
}

/**
 * @brief 确保图像为BGR8格式,
 *        Converts accepted camera image layouts into the BGR8 format required
 * by YOLO.
 *
 * @param image
 */
inline void ensureBgrU8C3(cv::Mat &image) {
  if (image.empty() || image.cols <= 0 || image.rows <= 0) {
    image.release();
    return;
  }
  try {
    if (image.depth() != CV_8U) {
      cv::Mat u8;
      image.convertTo(u8, CV_8U);
      image = std::move(u8);
    }
    if (image.channels() == 3)
      return;

    cv::Mat bgr;
    if (image.channels() == 1) {
      cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
      cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    } else {
      image.release();
      return;
    }
    image = std::move(bgr);
  } catch (const cv::Exception &) {
    image.release();
  }
}

/**
 * @brief 在图像上绘制头部姿态框
 *
 * @param image 图像
 * @param face_bbox 人脸边界框
 * @param yaw 偏航角
 * @param pitch 俯仰角
 * @param roll 翻滚角
 */
inline void drawHeadPoseBox(cv::Mat &image, const cv::Rect2f &face_bbox,
                            float yaw, float pitch, float roll) {
  constexpr float kDegreesToRadians = CV_PI / 180.0F;
  const float yaw_rad = yaw * kDegreesToRadians;
  const float pitch_rad = pitch * kDegreesToRadians;
  const float roll_rad = roll * kDegreesToRadians;
  const float side = std::min(face_bbox.width, face_bbox.height) * 0.7F;
  const float focal_length = side * 3.0F;
  const cv::Point2f center(face_bbox.x + face_bbox.width * 0.5F,
                           face_bbox.y + face_bbox.height * 0.5F);

  const float cos_yaw = std::cos(yaw_rad);
  const float sin_yaw = std::sin(yaw_rad);
  const float cos_pitch = std::cos(pitch_rad);
  const float sin_pitch = std::sin(pitch_rad);
  const float cos_roll = std::cos(roll_rad);
  const float sin_roll = std::sin(roll_rad);
  const cv::Matx33f rotation(
      cos_roll * cos_yaw, cos_roll * sin_yaw * sin_pitch - sin_roll * cos_pitch,
      cos_roll * sin_yaw * cos_pitch + sin_roll * sin_pitch, sin_roll * cos_yaw,
      sin_roll * sin_yaw * sin_pitch + cos_roll * cos_pitch,
      sin_roll * sin_yaw * cos_pitch - cos_roll * sin_pitch, -sin_yaw,
      cos_yaw * sin_pitch, cos_yaw * cos_pitch);

  const float half_side = side * 0.5F;
  const auto project = [&](const cv::Vec3f &point) {
    const cv::Vec3f rotated = rotation * point;
    const float scale = focal_length / (focal_length + rotated[2]);
    return cv::Point(cvRound(center.x + rotated[0] * scale),
                     cvRound(center.y + rotated[1] * scale));
  };
  const std::array<cv::Vec3f, 8> vertices = {
      cv::Vec3f{-half_side, -half_side, -half_side},
      cv::Vec3f{half_side, -half_side, -half_side},
      cv::Vec3f{half_side, half_side, -half_side},
      cv::Vec3f{-half_side, half_side, -half_side},
      cv::Vec3f{-half_side, -half_side, half_side},
      cv::Vec3f{half_side, -half_side, half_side},
      cv::Vec3f{half_side, half_side, half_side},
      cv::Vec3f{-half_side, half_side, half_side}};
  std::array<cv::Point, 8> projected_vertices;
  for (std::size_t index = 0; index < vertices.size(); ++index) {
    projected_vertices[index] = project(vertices[index]);
  }

  constexpr std::array<std::array<int, 2>, 12> edges = {{{{0, 1}},
                                                         {{1, 2}},
                                                         {{2, 3}},
                                                         {{3, 0}},
                                                         {{4, 5}},
                                                         {{5, 6}},
                                                         {{6, 7}},
                                                         {{7, 4}},
                                                         {{0, 4}},
                                                         {{1, 5}},
                                                         {{2, 6}},
                                                         {{3, 7}}}};
  for (const auto &edge : edges) {
    cv::line(image, projected_vertices[edge[0]], projected_vertices[edge[1]],
             cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  }

  const cv::Point projected_origin = project(cv::Vec3f(0.0F, 0.0F, 0.0F));
  const float axis_length = side * 0.9F;
  const std::array<cv::Vec3f, 3> axes = {cv::Vec3f(axis_length, 0.0F, 0.0F),
                                         cv::Vec3f(0.0F, axis_length, 0.0F),
                                         cv::Vec3f(0.0F, 0.0F, axis_length)};
  // OpenCV uses BGR: X is red, Y is green, and Z is blue as in RViz TF.
  const std::array<cv::Scalar, 3> axis_colors = {
      cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)};
  for (std::size_t index = 0; index < axes.size(); ++index) {
    cv::arrowedLine(image, projected_origin, project(axes[index]),
                    axis_colors[index], 2, cv::LINE_AA, 0, 0.2);
  }
}

/**
 * @brief 解码压缩的彩色图像消息
 *
 * @param message 压缩的彩色图像消息
 * @param color_image 解码后的彩色图像
 * @return true 解码成功
 * @return false 解码失败
 */
inline bool
decodeCompressedColor(const sensor_msgs::msg::CompressedImage &message,
                      cv::Mat &color_image) {
  if (message.data.empty() ||
      message.data.size() >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  const cv::Mat encoded(1, static_cast<int>(message.data.size()), CV_8UC1,
                        const_cast<uint8_t *>(message.data.data()));
  color_image = cv::imdecode(encoded, cv::IMREAD_COLOR);
  return !color_image.empty();
}

/**
 * @brief 解码压缩的深度图像消息
 *
 * @param message 压缩的深度图像消息
 * @param depth_image 解码后的深度图像
 * @param encoding 解码后的深度图像编码格式
 * @return true 解码成功
 * @return false 解码失败
 */
inline bool
decodeCompressedDepth(const sensor_msgs::msg::CompressedImage &message,
                      cv::Mat &depth_image, std::string &encoding) {
  constexpr std::array<uint8_t, 8> kPngSignature = {0x89, 0x50, 0x4e, 0x47,
                                                    0x0d, 0x0a, 0x1a, 0x0a};
  const auto png_begin =
      std::search(message.data.begin(), message.data.end(),
                  kPngSignature.begin(), kPngSignature.end());
  if (png_begin == message.data.end()) {
    return false;
  }

  const std::size_t png_offset =
      static_cast<std::size_t>(std::distance(message.data.begin(), png_begin));
  const std::size_t png_size = message.data.size() - png_offset;
  if (png_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return false;
  }

  const cv::Mat encoded(
      1, static_cast<int>(png_size), CV_8UC1,
      const_cast<uint8_t *>(message.data.data() + png_offset));
  const cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_UNCHANGED);
  if (decoded.empty()) {
    return false;
  }

  std::string normalized_format = message.format;
  std::transform(normalized_format.begin(), normalized_format.end(),
                 normalized_format.begin(), [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  if (normalized_format.find("32fc1") == std::string::npos) {
    if (decoded.type() != CV_16UC1) {
      return false;
    }
    depth_image = decoded;
    encoding = sensor_msgs::image_encodings::TYPE_16UC1;
    return true;
  }

  if (png_offset < sizeof(CompressedDepthHeader) ||
      decoded.type() != CV_16UC1) {
    return false;
  }
  CompressedDepthHeader header{};
  std::memcpy(&header, message.data.data(), sizeof(header));
  if (header.format != 0 || !std::isfinite(header.depth_quant_a) ||
      !std::isfinite(header.depth_quant_b) || header.depth_quant_a <= 0.0F) {
    return false;
  }

  depth_image.create(decoded.rows, decoded.cols, CV_32FC1);
  const float invalid_depth = std::numeric_limits<float>::quiet_NaN();
  for (int row = 0; row < decoded.rows; ++row) {
    const auto *source = decoded.ptr<uint16_t>(row);
    auto *destination = depth_image.ptr<float>(row);
    for (int col = 0; col < decoded.cols; ++col) {
      const float denominator =
          static_cast<float>(source[col]) - header.depth_quant_b;
      destination[col] = source[col] != 0 && denominator > 0.0F
                             ? header.depth_quant_a / denominator
                             : invalid_depth;
    }
  }
  encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  return true;
}

/**
 * @brief 解码深度图像消息为浮点米单位的深度图像
 *
 * @param depth_msg
 * @param depth_meters
 * @return true
 * @return false
 */
inline bool
decodeToFloatMeters(const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
                    cv::Mat &depth_meters) {
  if (!depth_msg)
    return false;

  if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    cv::Mat depth_raw(depth_msg->height, depth_msg->width, CV_16UC1,
                      const_cast<uint8_t *>(depth_msg->data.data()),
                      static_cast<size_t>(depth_msg->step));
    depth_raw.convertTo(depth_meters, CV_32F, 0.001);
    return true;
  }

  if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    cv::Mat depth_raw(depth_msg->height, depth_msg->width, CV_32FC1,
                      const_cast<uint8_t *>(depth_msg->data.data()),
                      static_cast<size_t>(depth_msg->step));
    depth_meters = depth_raw;
    return true;
  }

  return false;
}
#endif // PERCEPTION_COMMON_HPP