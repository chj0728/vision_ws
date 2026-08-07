#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmw/qos_profiles.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "remote_infer_bridge_cpp/msg/person_perception.hpp"
#include "remote_infer_bridge_cpp/msg/scene_perception_result.hpp"
#include "yolo_engine.hpp"

namespace {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool isJpegInImageMsg(const std::string& enc) {
    const std::string e = toLower(enc);
    return e.find("jpeg") != std::string::npos || e.find("jpg") != std::string::npos ||
           e.find("mjpeg") != std::string::npos || e.find("mjpg") != std::string::npos;
}

void ensureBgrU8C3(cv::Mat& img) {
    if (img.empty() || img.cols <= 0 || img.rows <= 0) {
        img.release();
        return;
    }
    try {
        if (img.depth() != CV_8U) {
            cv::Mat u8;
            img.convertTo(u8, CV_8U);
            img = std::move(u8);
        }
        if (img.channels() == 3) return;
        cv::Mat c3;
        if (img.channels() == 1) {
            cv::cvtColor(img, c3, cv::COLOR_GRAY2BGR);
        } else if (img.channels() == 4) {
            cv::cvtColor(img, c3, cv::COLOR_BGRA2BGR);
        } else {
            img.release();
            return;
        }
        img = std::move(c3);
    } catch (const cv::Exception&) {
        img.release();
    }
}

}  // namespace

class TrtInferNode : public rclcpp::Node {
public:
    TrtInferNode() : Node("trt_infer_ros_node") {
        model_path_ = declare_parameter<std::string>("model_path", "yolo26m_fp16.engine");
        conf_ = static_cast<float>(declare_parameter<double>("conf_threshold", 0.45));
        image_topic_ = declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
        depth_topic_ = declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
        out_topic_ = declare_parameter<std::string>("detections_topic", "/remote_infer/detections");

        max_distance_m_ = static_cast<float>(declare_parameter<double>("max_distance_m", 5.0));
        only_human_class_ = declare_parameter<bool>("only_human_class", true);
        human_class_id_ = declare_parameter<int>("human_class_id", 0);

        depth_sync_exact_ = declare_parameter<bool>("depth_sync_exact", false);
        sync_queue_size_ = static_cast<uint32_t>(declare_parameter<int>("depth_sync_queue_size", 40));
        depth_scale_to_meters_ = static_cast<float>(declare_parameter<double>("depth_scale_to_meters", 0.001));

        depth_roi_y0_ = declare_parameter<double>("depth_roi_y0", 0.52);
        depth_roi_y1_ = declare_parameter<double>("depth_roi_y1", 0.98);
        depth_roi_x_margin_ = declare_parameter<double>("depth_roi_x_margin", 0.2);
        min_depth_m_ = static_cast<float>(declare_parameter<double>("min_depth_m", 0.08));
        max_depth_read_m_ = static_cast<float>(declare_parameter<double>("max_depth_read_m", 25.0));

        depth_percentile_ = declare_parameter<double>("depth_percentile", 0.5);
        depth_trim_close_ratio_ = declare_parameter<double>("depth_trim_close_ratio", 0.0);

        distance_ema_enable_ = declare_parameter<bool>("distance_ema_enable", false);
        distance_ema_alpha_ = static_cast<float>(declare_parameter<double>("distance_ema_alpha", 0.35));

        log_frame_timing_ = declare_parameter<bool>("log_frame_timing", false);
        log_frame_timing_period_ms_ = declare_parameter<int>("log_frame_timing_period_ms", 1000);

        std::string bbox_space = declare_parameter<std::string>("bbox_coord_space", "letterbox");
        const bool bbox_orig = (bbox_space == "original" || bbox_space == "orig");
        const int engine_in_h = declare_parameter<int>("engine_input_h", 0);
        const int engine_in_w = declare_parameter<int>("engine_input_w", 0);
        engine_ = std::make_unique<YOLOEngine>(model_path_, conf_, bbox_orig, engine_in_h, engine_in_w);

        pub_ = create_publisher<remote_infer_bridge_cpp::msg::ScenePerceptionResult>(
            out_topic_, rclcpp::QoS(2).reliable());

        rclcpp::QoS image_qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default), rmw_qos_profile_default);
        image_qos.keep_last(6);
        const rmw_qos_profile_t qprof = image_qos.get_rmw_qos_profile();

        color_sub_.subscribe(this, image_topic_, qprof);
        depth_sub_.subscribe(this, depth_topic_, qprof);
        if (depth_sync_exact_) {
            using Pol = message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
            sync_exact_.reset(new message_filters::Synchronizer<Pol>(Pol(sync_queue_size_), color_sub_, depth_sub_));
            sync_exact_->registerCallback(std::bind(&TrtInferNode::onSyncedColorDepth, this, std::placeholders::_1,
                                                    std::placeholders::_2));
        } else {
            using Pol =
                message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
            sync_approx_.reset(new message_filters::Synchronizer<Pol>(Pol(sync_queue_size_), color_sub_, depth_sub_));
            sync_approx_->registerCallback(std::bind(&TrtInferNode::onSyncedColorDepth, this, std::placeholders::_1,
                                                     std::placeholders::_2));
        }
    }

private:
    bool decodeToFloatMeters(const sensor_msgs::msg::Image::ConstSharedPtr& d, cv::Mat& out_m) {
        if (!d) return false;
        const std::string enc = toLower(d->encoding);
        if (enc == sensor_msgs::image_encodings::TYPE_16UC1 || enc == "16uc1") {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(d, sensor_msgs::image_encodings::TYPE_16UC1);
            if (depth_f32_buf_.rows != (int)d->height || depth_f32_buf_.cols != (int)d->width)
                depth_f32_buf_.create(d->height, d->width, CV_32F);
            cv_ptr->image.convertTo(depth_f32_buf_, CV_32F, (double)depth_scale_to_meters_);
            out_m = depth_f32_buf_;
            return true;
        }
        if (enc == sensor_msgs::image_encodings::TYPE_32FC1 || enc == "32fc1") {
            out_m = cv_bridge::toCvShare(d, sensor_msgs::image_encodings::TYPE_32FC1)->image;
            return true;
        }
        return false;
    }

    void onSyncedColorDepth(const sensor_msgs::msg::Image::ConstSharedPtr& color_msg,
                            const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
        const auto t_recv = std::chrono::steady_clock::now();
        
        cv::Mat img;
        const bool jpeg = isJpegInImageMsg(color_msg->encoding) || 
            (color_msg->data.size() >= 2 && (uint8_t)color_msg->data[0] == 0xff && (uint8_t)color_msg->data[1] == 0xd8);
        
        if (jpeg) {
            img = cv::imdecode(cv::Mat(1, color_msg->data.size(), CV_8UC1, const_cast<uint8_t*>(color_msg->data.data())), cv::IMREAD_COLOR);
        } else {
            img = cv_bridge::toCvShare(color_msg, sensor_msgs::image_encodings::BGR8)->image;
        }
        ensureBgrU8C3(img);
        
        cv::Mat depth_m;
        if (!decodeToFloatMeters(depth_msg, depth_m)) return;

        // ALL GPU PIPELINE
        auto dets = engine_->inferWithDepth(img, depth_m, conf_, min_depth_m_, max_depth_read_m_, 
                                            (float)depth_roi_y0_, (float)depth_roi_y1_, (float)depth_roi_x_margin_, 
                                            (float)depth_trim_close_ratio_, (float)depth_percentile_);

        remote_infer_bridge_cpp::msg::ScenePerceptionResult out;
        out.image_width = img.cols; out.image_height = img.rows;

        if (dets.size() != distance_ema_state_.size()) {
            distance_ema_state_.resize(dets.size());
            distance_ema_valid_.assign(dets.size(), 0);
        }

        for (size_t i = 0; i < dets.size(); ++i) {
            auto& d = dets[i];
            if (only_human_class_ && d.class_id != human_class_id_) continue;

            float y = d.distance;
            bool depth_ok = (y > 0.f && y <= max_distance_m_);
            
            if (depth_ok && distance_ema_enable_) {
                if (!distance_ema_valid_[i]) { distance_ema_state_[i] = y; distance_ema_valid_[i] = 1; }
                else distance_ema_state_[i] = distance_ema_alpha_ * y + (1.f - distance_ema_alpha_) * distance_ema_state_[i];
                y = distance_ema_state_[i];
            }

            float rounded = std::round(y * 1000.f) / 1000.f;

            remote_infer_bridge_cpp::msg::PersonPerception pp;
            pp.body_x = d.x; pp.body_y = d.y; pp.body_w = d.w; pp.body_h = d.h;
            pp.body_conf = d.conf;
            pp.distance = depth_ok ? rounded : -1.f;
            pp.has_face = false;
            out.persons.push_back(pp);
        }

        out.header = color_msg->header;
        out.body_pipeline_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t_recv).count();
        pub_->publish(out);
    }

    std::string model_path_;
    float conf_;
    std::string image_topic_, depth_topic_, out_topic_;
    float max_distance_m_, depth_scale_to_meters_, min_depth_m_, max_depth_read_m_;
    bool only_human_class_, depth_sync_exact_, distance_ema_enable_, log_frame_timing_;
    int human_class_id_, log_frame_timing_period_ms_;
    uint32_t sync_queue_size_;
    double depth_roi_y0_, depth_roi_y1_, depth_roi_x_margin_, depth_percentile_, depth_trim_close_ratio_;
    float distance_ema_alpha_;
    std::vector<float> distance_ema_state_;
    std::vector<uint8_t> distance_ema_valid_;
    cv::Mat depth_f32_buf_;
    std::unique_ptr<YOLOEngine> engine_;
    rclcpp::Publisher<remote_infer_bridge_cpp::msg::ScenePerceptionResult>::SharedPtr pub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> color_sub_, depth_sub_;
    std::unique_ptr<message_filters::Synchronizer<message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> sync_exact_;
    std::unique_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> sync_approx_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrtInferNode>());
    rclcpp::shutdown();
    return 0;
}
