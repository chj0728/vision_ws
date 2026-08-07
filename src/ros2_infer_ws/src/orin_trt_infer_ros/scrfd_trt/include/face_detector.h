#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

struct FaceObject {
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};

#endif
