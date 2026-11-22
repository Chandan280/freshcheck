// pipeline.h - the full per-image pipeline: Detector -> crops -> Classifier -> Grader.
#pragma once
#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "classifier.h"
#include "detector.h"
#include "grader.h"
#include "json.hpp"

class Pipeline {
public:
    // cfg: full config; base_dir: folder of config.json (model paths are relative to it).
    Pipeline(const nlohmann::json& cfg, const std::string& base_dir);

    // Returns {fruits:[...], fallback, image_size, latency_ms}.
    nlohmann::json run(const cv::Mat& bgr) const;
    const Grader& grader() const { return *grader_; }

private:
    std::unique_ptr<Detector> detector_;
    std::unique_ptr<Classifier> classifier_;
    std::unique_ptr<Grader> grader_;
};
