// main.cpp - FreshCheck entry point.
//   freshcheck                      -> start web server (port from config, default 7860)
//   freshcheck --image apple.jpg    -> print JSON result and exit
//   options: --config path/to/config.json  --port 8080
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "json.hpp"
#include "logger.h"
#include "pipeline.h"
#include "server.h"

namespace fs = std::filesystem;

// Looks for config.json next to the working dir, then in app/, then next to the binary.
static std::string find_config(const char* argv0) {
    for (fs::path p : {fs::path("config.json"), fs::path("app/config.json"),
                       fs::absolute(argv0).parent_path() / "../config.json"})
        if (fs::exists(p)) return p.string();
    return "config.json";
}

int main(int argc, char** argv) {
    std::string config_path, image_path;
    int port = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (a == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: freshcheck [--config config.json] [--image photo.jpg] [--port 7860]\n";
            return 0;
        }
    }
    if (config_path.empty()) config_path = find_config(argv[0]);

    try {
        std::ifstream f(config_path);
        if (!f) throw std::runtime_error("config not found: " + config_path);
        nlohmann::json cfg = nlohmann::json::parse(f, nullptr, true, /*ignore_comments=*/true);
        std::string base = fs::absolute(config_path).parent_path().string();
        logx::info("Config: " + fs::absolute(config_path).lexically_normal().string());

        Pipeline pipeline(cfg, base);

        if (!image_path.empty()) {  // CLI mode
            cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
            if (img.empty()) throw std::runtime_error("could not read image: " + image_path);
            std::cout << pipeline.run(img).dump(2) << std::endl;
            return 0;
        }

        if (port < 0) port = cfg.value("port", 7860);
        if (const char* env = std::getenv("PORT")) port = std::stoi(env);
        std::string web_dir = (fs::path(base) / cfg.value("web_dir", "web")).string();
        Server server(pipeline, web_dir);
        server.listen("0.0.0.0", port);
    } catch (const std::exception& e) {
        logx::error(e.what());
        return 1;
    }
    return 0;
}
