// detector.h - YOLOv8n (COCO) fruit detector: keeps only apple / banana / orange.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "json.hpp"

struct Detection {
    cv::Rect box;      // in original image pixels
    std::string name;  // "apple", "banana" or "orange"
    float score = 0;   // YOLO class confidence
};

class Detector {
public:
    Detector(const nlohmann::json& cfg, const std::string& model_path, int threads);
    std::vector<Detection> detect(const cv::Mat& bgr) const;

private:
    std::unique_ptr<Ort::Session> session_;
    std::string in_name_, out_name_;
    int size_ = 640;
    float conf_ = 0.4f, iou_ = 0.45f, padding_ = 0.08f;
    int max_fruits_ = 10;
    std::map<int, std::string> classes_;  // COCO id -> fruit name
};
