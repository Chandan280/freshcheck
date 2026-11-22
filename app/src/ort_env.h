// ort_env.h - one shared ONNX Runtime environment + session helper.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

inline Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "freshcheck");
    return env;
}

// Creates a CPU session. threads = 0 lets ONNX Runtime pick (all cores).
inline std::unique_ptr<Ort::Session> make_session(const std::string& path, int threads) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return std::make_unique<Ort::Session>(ort_env(), path.c_str(), opts);
}

// Returns the input and output tensor names of a session.
inline void io_names(Ort::Session& s, std::string& in, std::string& out) {
    Ort::AllocatorWithDefaultOptions alloc;
    in = s.GetInputNameAllocated(0, alloc).get();
    out = s.GetOutputNameAllocated(0, alloc).get();
}
