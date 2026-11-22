// classifier.cpp - resize 224x224, ImageNet normalisation, softmax.
#include "classifier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "ort_env.h"

Classifier::Classifier(const nlohmann::json& cfg, const std::string& model_path, int threads) {
    session_ = make_session(model_path, threads);
    io_names(*session_, in_name_, out_name_);
    size_ = cfg.value("input_size", 224);
    for (int c = 0; c < 3; ++c) {
        mean_[c] = cfg.at("mean")[c];
        std_[c] = cfg.at("std")[c];
    }
    labels_ = cfg.at("labels").get<std::vector<std::string>>();
    fallback_min_conf_ = cfg.value("fallback_min_confidence", 0.6f);
}

// "rottenbanana" -> "banana", "freshapples" -> "apple".
static std::string fruit_of(const std::string& label) {
    for (const char* f : {"apple", "banana", "orange"})
        if (label.find(f) != std::string::npos) return f;
    return "fruit";
}

std::vector<ClassResult> Classifier::classify(const std::vector<cv::Mat>& crops) const {
    if (crops.empty()) return {};
    const int B = (int)crops.size(), plane = size_ * size_;
    std::vector<float> input((size_t)B * 3 * plane);

    // Same preprocessing as training (val transform): Resize(224,224) + Normalize.
    for (int b = 0; b < B; ++b) {
        cv::Mat r;
        cv::resize(crops[b], r, cv::Size(size_, size_), 0, 0, cv::INTER_LINEAR);
        float* dst = input.data() + (size_t)b * 3 * plane;
        for (int y = 0; y < size_; ++y) {
            const cv::Vec3b* row = r.ptr<cv::Vec3b>(y);
            for (int x = 0; x < size_; ++x) {
                for (int c = 0; c < 3; ++c) {  // c: 0=R,1=G,2=B ; OpenCV is BGR
                    float v = row[x][2 - c] / 255.f;
                    dst[c * plane + y * size_ + x] = (v - mean_[c]) / std_[c];
                }
            }
        }
    }

    std::array<int64_t, 4> shape{B, 3, size_, size_};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                    shape.data(), shape.size());
    const char* in_names[] = {in_name_.c_str()};
    const char* out_names[] = {out_name_.c_str()};
    auto out = session_->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    const float* logits = out[0].GetTensorData<float>();
    const int K = (int)labels_.size();
    if (out[0].GetTensorTypeAndShapeInfo().GetShape().back() != K)
        throw std::runtime_error("classifier output size does not match config labels");

    std::vector<ClassResult> results(B);
    for (int b = 0; b < B; ++b) {
        const float* l = logits + (size_t)b * K;
        float mx = *std::max_element(l, l + K), sum = 0;
        auto& res = results[b];
        res.probs.resize(K);
        for (int k = 0; k < K; ++k) sum += (res.probs[k] = std::exp(l[k] - mx));
        int top = 0;
        for (int k = 0; k < K; ++k) {
            res.probs[k] /= sum;
            if (labels_[k].rfind("rotten", 0) == 0) res.p_rotten += res.probs[k];
            if (res.probs[k] > res.probs[top]) top = k;
        }
        res.top_label = labels_[top];
        res.top_fruit = fruit_of(labels_[top]);
        res.top_prob = res.probs[top];
    }
    return results;
}
