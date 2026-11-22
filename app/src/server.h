// server.h - HTTP API + static web UI (cpp-httplib).
#pragma once
#include <map>
#include <mutex>
#include <string>

#include "json.hpp"
#include "pipeline.h"

class Server {
public:
    Server(const Pipeline& pipeline, const std::string& web_dir) : pipe_(pipeline), web_dir_(web_dir) {}
    void listen(const std::string& host, int port);

private:
    // Adds logged fruits to the session totals.
    void record(const nlohmann::json& result);
    nlohmann::json stats();

    const Pipeline& pipe_;
    std::string web_dir_;

    // Session totals (since server start or last reset). Shared by all clients.
    std::mutex mu_;
    long frames_ = 0, logged_ = 0;
    double kg_saved_ = 0, inr_saved_ = 0, latency_sum_ = 0;
    std::map<std::string, long> by_grade_, by_fruit_;
};
