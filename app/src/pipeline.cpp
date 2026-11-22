// pipeline.cpp
#include "pipeline.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "logger.h"

namespace fs = std::filesystem;

static std::string resolve(const std::string& base, const std::string& p) {
    fs::path path(p);
    return path.is_absolute() ? p : (fs::path(base) / path).lexically_normal().string();
}

static void require_file(const std::string& path, const std::string& what) {
    if (!fs::exists(path))
        throw std::runtime_error(what + " model not found at " + path +
                                 " (see README: export it from training/train_classifier.ipynb)");
}

Pipeline::Pipeline(const nlohmann::json& cfg, const std::string& base_dir) {
    const int threads = cfg.value("threads", 0);
    std::string det_path = resolve(base_dir, cfg.at("detector").at("model"));
    std::string cls_path = resolve(base_dir, cfg.at("classifier").at("model"));
    require_file(det_path, "Detector");
    require_file(cls_path, "Classifier");
    logx::info("Loading detector: " + det_path);
    detector_ = std::make_unique<Detector>(cfg.at("detector"), det_path, threads);
    logx::info("Loading classifier: " + cls_path);
    classifier_ = std::make_unique<Classifier>(cfg.at("classifier"), cls_path, threads);
    grader_ = std::make_unique<Grader>(cfg);
}

static double round2(double v) { return std::round(v * 100.0) / 100.0; }

nlohmann::json Pipeline::run(const cv::Mat& bgr) const {
    auto t0 = std::chrono::steady_clock::now();

    // 1) Detect fruits. 2) Crop each box.
    std::vector<Detection> dets = detector_->detect(bgr);
    std::vector<cv::Mat> crops;
    for (auto& d : dets) crops.push_back(bgr(d.box));

    // Fallback: no fruit found -> classify the whole image.
    bool fallback = dets.empty();
    if (fallback) crops.push_back(bgr);

    // 3) Classify all crops in one batch.
    std::vector<ClassResult> cls = classifier_->classify(crops);

    // 4) Grade each fruit.
    nlohmann::json fruits = nlohmann::json::array();
    for (size_t i = 0; i < cls.size(); ++i) {
        const auto& c = cls[i];
        // In fallback mode, drop low-confidence whole-image guesses (probably not a fruit).
        if (fallback && c.top_prob < classifier_->fallback_min_confidence()) continue;
        std::string name = fallback ? c.top_fruit : dets[i].name;
        cv::Rect box = fallback ? cv::Rect(0, 0, bgr.cols, bgr.rows) : dets[i].box;
        Grade g = grader_->grade(name, c.p_rotten);
        fruits.push_back({
            {"name", name},
            {"box", {box.x, box.y, box.width, box.height}},
            {"grade", g.grade},
            {"confidence", round2(g.confidence)},
            {"p_rotten", round2(c.p_rotten)},
            {"detector_score", fallback ? 0.0 : round2(dets[i].score)},
            {"classifier_label", c.top_label},
            {"action", g.action},
            {"action_color", g.action_color},
            {"use_within_days", g.use_within_days},
            {"message_en", g.message_en},
            {"message_hi", g.message_hi},
        });
    }

    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return {{"fruits", fruits},
            {"fallback", fallback},
            {"image_size", {bgr.cols, bgr.rows}},
            {"latency_ms", std::round(ms * 10) / 10}};
}
