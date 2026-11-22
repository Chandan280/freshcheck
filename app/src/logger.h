// logger.h - tiny timestamped logger (header only).
#pragma once
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace logx {

inline std::mutex& log_mutex() {
    static std::mutex m;
    return m;
}

// Writes "2026-11-22 19:20:01.123 [INFO] message" to stderr.
inline void write(const char* level, const std::string& msg) {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::lock_guard<std::mutex> lock(log_mutex());
    std::cerr << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << '.' << std::setw(3)
              << std::setfill('0') << ms.count() << " [" << level << "] " << msg << std::endl;
}

inline void info(const std::string& m) { write("INFO", m); }
inline void warn(const std::string& m) { write("WARN", m); }
inline void error(const std::string& m) { write("ERROR", m); }

}  // namespace logx
