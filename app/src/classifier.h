// classifier.h - ResNet-18 fresh/rotten classifier (6 classes, ONNX).
#pragma once
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "json.hpp"

struct ClassResult {
    std::vector<float> probs;  // softmax over labels
    float p_rotten = 0;        // sum of probabilities of "rotten*" labels
    std::string top_label;     // e.g. "rottenbanana"
    std::string top_fruit;     // "apple" / "banana" / "orange" (from top label)
    float top_prob = 0;
};

class Classifier {
public:
    Classifier(const nlohmann::json& cfg, const std::string& model_path, int threads);
    // Classifies several crops in one batch.
    std::vector<ClassResult> classify(const std::vector<cv::Mat>& crops_bgr) const;
    float fallback_min_confidence() const { return fallback_min_conf_; }

private:
    std::unique_ptr<Ort::Session> session_;
    std::string in_name_, out_name_;
    int size_ = 224;
    float mean_[3] = {0.485f, 0.456f, 0.406f};
    float std_[3] = {0.229f, 0.224f, 0.225f};
    std::vector<std::string> labels_;
    float fallback_min_conf_ = 0.6f;
};
