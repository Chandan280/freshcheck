// server.cpp
#include "server.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "httplib.h"
#include "logger.h"

using nlohmann::json;

// Minimal base64 decoder (accepts an optional "data:image/...;base64," prefix).
static std::string base64_decode(const std::string& in) {
    std::string s = in;
    auto comma = s.find(',');
    if (s.rfind("data:", 0) == 0 && comma != std::string::npos) s = s.substr(comma + 1);
    static const std::string chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int val = 0, bits = -8;
    for (unsigned char c : s) {
        auto pos = chars.find(c);
        if (pos == std::string::npos) continue;  // skips '=', newlines, spaces
        val = (val << 6) + (int)pos;
        bits += 6;
        if (bits >= 0) {
            out.push_back(char((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

// Decodes image bytes; downsizes very large photos so latency stays low.
static cv::Mat decode_image(const std::string& bytes) {
    std::vector<uchar> buf(bytes.begin(), bytes.end());
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) return img;
    const int max_side = 1280;
    int m = std::max(img.cols, img.rows);
    if (m > max_side) {
        double s = (double)max_side / m;
        cv::resize(img, img, cv::Size(), s, s, cv::INTER_AREA);
    }
    return img;
}

static bool truthy(const std::string& v) { return v == "1" || v == "true" || v == "yes"; }

void Server::record(const json& result) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& f : result["fruits"]) {
        std::string name = f["name"], grade = f["grade"];
        ++logged_;
        ++by_grade_[grade];
        ++by_fruit_[name];
        kg_saved_ += pipe_.grader().saved_kg(name, grade);
        inr_saved_ += pipe_.grader().saved_inr(name, grade);
    }
}

json Server::stats() {
    std::lock_guard<std::mutex> lock(mu_);
    json g = {{"Fresh", by_grade_["Fresh"]}, {"Medium", by_grade_["Medium"]}, {"Rotten", by_grade_["Rotten"]}};
    return {{"scans", frames_},
            {"fruits_logged", logged_},
            {"by_grade", g},
            {"by_fruit", by_fruit_},
            {"kg_saved", std::round(kg_saved_ * 100) / 100},
            {"inr_saved", std::round(inr_saved_)},
            {"avg_latency_ms", frames_ ? std::round(latency_sum_ / frames_ * 10) / 10 : 0},
            {"note", "Estimates from config.json values (avg weight, price per kg)."}};
}

void Server::listen(const std::string& host, int port) {
    httplib::Server svr;
    svr.set_payload_max_length(15 * 1024 * 1024);

    // CORS + request log.
    svr.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                             {"Access-Control-Allow-Headers", "Content-Type"}});
    svr.Options(R"(/api/.*)", [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        if (req.path != "/api/grade") logx::info(req.method + " " + req.path + " -> " + std::to_string(res.status));
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(json{{"status", "ok"}, {"engine", "cpp"}}.dump(), "application/json");
    });

    svr.Get("/api/stats", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(stats().dump(), "application/json");
    });

    svr.Post("/api/reset", [this](const httplib::Request&, httplib::Response& res) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            frames_ = logged_ = 0;
            kg_saved_ = inr_saved_ = latency_sum_ = 0;
            by_grade_.clear();
            by_fruit_.clear();
        }
        res.set_content(stats().dump(), "application/json");
    });

    // POST /api/grade : multipart "image" file, JSON {"image": base64, "record": bool},
    // or a raw image/* body. Add ?record=1 (or "record") to count the fruits in the session totals.
    svr.Post("/api/grade", [this](const httplib::Request& req, httplib::Response& res) {
        std::string bytes;
        bool rec = req.has_param("record") && truthy(req.get_param_value("record"));
        try {
            if (req.is_multipart_form_data()) {
                if (req.has_file("image")) bytes = req.get_file_value("image").content;
                if (req.has_file("record")) rec = truthy(req.get_file_value("record").content);
            } else if (req.get_header_value("Content-Type").find("json") != std::string::npos) {
                json body = json::parse(req.body);
                bytes = base64_decode(body.value("image", ""));
                if (body.contains("record")) rec = body["record"].is_boolean() ? body["record"].get<bool>() : false;
            } else {
                bytes = req.body;  // raw image bytes
            }
            cv::Mat img = decode_image(bytes);
            if (img.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "could not decode image"}}.dump(), "application/json");
                return;
            }
            json out = pipe_.run(img);
            out["recorded"] = rec;
            {
                std::lock_guard<std::mutex> lock(mu_);
                ++frames_;
                latency_sum_ += out["latency_ms"].get<double>();
            }
            if (rec) record(out);
            logx::info("grade " + std::to_string(img.cols) + "x" + std::to_string(img.rows) + " fruits=" +
                       std::to_string(out["fruits"].size()) + " fallback=" + (out["fallback"].get<bool>() ? "1" : "0") +
                       " " + std::to_string((int)std::lround(out["latency_ms"].get<double>())) + "ms" + (rec ? " [logged]" : ""));
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& e) {
            logx::error(std::string("grade failed: ") + e.what());
            res.status = 500;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // Web UI: "/" -> web/index.html, other files from web/.
    svr.Get("/", [this](const httplib::Request&, httplib::Response& res) {
        std::ifstream f(web_dir_ + "/index.html");
        std::stringstream ss;
        ss << f.rdbuf();
        res.set_content(ss.str(), "text/html; charset=utf-8");
    });
    svr.set_mount_point("/static", web_dir_);

    logx::info("FreshCheck listening on http://" + host + ":" + std::to_string(port));
    if (!svr.listen(host, port)) logx::error("Could not bind port " + std::to_string(port));
}
