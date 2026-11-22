// detector.cpp - letterbox -> YOLOv8 -> filter fruit classes -> NMS.
#include "detector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>

#include "ort_env.h"

Detector::Detector(const nlohmann::json& cfg, const std::string& model_path, int threads) {
    session_ = make_session(model_path, threads);
    io_names(*session_, in_name_, out_name_);
    size_ = cfg.value("input_size", 640);
    conf_ = cfg.value("conf_threshold", 0.4f);
    iou_ = cfg.value("nms_iou", 0.45f);
    padding_ = cfg.value("box_padding", 0.08f);
    max_fruits_ = cfg.value("max_fruits", 10);
    for (auto& [id, name] : cfg.at("coco_classes").items()) classes_[std::stoi(id)] = name;
}

// Intersection-over-union of two boxes.
static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.f;
}

std::vector<Detection> Detector::detect(const cv::Mat& bgr) const {
    // 1) Letterbox: resize keeping aspect ratio, pad to size x size with grey (114).
    const int W = bgr.cols, H = bgr.rows;
    const float scale = std::min(size_ / (float)W, size_ / (float)H);
    const int nw = (int)std::round(W * scale), nh = (int)std::round(H * scale);
    const int padx = (size_ - nw) / 2, pady = (size_ - nh) / 2;
    cv::Mat canvas(size_, size_, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(nw, nh));
    resized.copyTo(canvas(cv::Rect(padx, pady, nw, nh)));

    // 2) HWC BGR uint8 -> NCHW RGB float [0,1].
    std::vector<float> input(3 * size_ * size_);
    const int plane = size_ * size_;
    for (int y = 0; y < size_; ++y) {
        const cv::Vec3b* row = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < size_; ++x) {
            const int i = y * size_ + x;
            input[0 * plane + i] = row[x][2] / 255.f;  // R
            input[1 * plane + i] = row[x][1] / 255.f;  // G
            input[2 * plane + i] = row[x][0] / 255.f;  // B
        }
    }

    // 3) Run the model.
    std::array<int64_t, 4> shape{1, 3, size_, size_};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                    shape.data(), shape.size());
    const char* in_names[] = {in_name_.c_str()};
    const char* out_names[] = {out_name_.c_str()};
    auto out = session_->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);

    // YOLOv8 output is [1, 84, N]: 4 box values (cx,cy,w,h) + 80 class scores.
    auto dims = out[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* data = out[0].GetTensorData<float>();
    const bool transposed = dims[1] > dims[2];  // some exports give [1, N, 84]
    const int64_t C = transposed ? dims[2] : dims[1];
    const int64_t N = transposed ? dims[1] : dims[2];
    auto at = [&](int64_t c, int64_t i) { return transposed ? data[i * C + c] : data[c * N + i]; };

    // 4) Keep candidates whose best fruit-class score passes the threshold.
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<std::string> names;
    for (int64_t i = 0; i < N; ++i) {
        float best = 0;
        std::string best_name;
        for (auto& [cls, name] : classes_) {
            float s = at(4 + cls, i);
            if (s > best) { best = s; best_name = name; }
        }
        if (best < conf_) continue;
        float cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
        // Undo letterbox to get original-image coordinates.
        float x0 = (cx - w / 2 - padx) / scale, y0 = (cy - h / 2 - pady) / scale;
        boxes.emplace_back(x0, y0, w / scale, h / scale);
        scores.push_back(best);
        names.push_back(best_name);
    }

    // 5) Class-agnostic NMS (a fruit should get one box, even if apple/orange both fire).
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<int> keep;
    for (int i : order) {
        bool ok = true;
        for (int k : keep) if (iou(boxes[i], boxes[k]) > iou_) { ok = false; break; }
        if (ok) keep.push_back(i);
        if ((int)keep.size() >= max_fruits_) break;
    }

    // 6) Pad boxes slightly (more context for the classifier) and clip to the image.
    std::vector<Detection> dets;
    const cv::Rect img(0, 0, W, H);
    for (int k : keep) {
        const auto& b = boxes[k];
        float px = b.width * padding_, py = b.height * padding_;
        cv::Rect r((int)(b.x - px), (int)(b.y - py), (int)(b.width + 2 * px), (int)(b.height + 2 * py));
        r &= img;
        if (r.width < 8 || r.height < 8) continue;
        dets.push_back({r, names[k], scores[k]});
    }
    return dets;
}
