# FreshCheck - C++ server image for Hugging Face Spaces (Docker, free CPU tier) or any Docker host.
#   docker build -t freshcheck .   &&   docker run -p 7860:7860 freshcheck
# Needs models/yolov8n.onnx and models/resnet18_fresh.onnx in the build folder.

# ---------- stage 1: build ----------
FROM ubuntu:24.04 AS build
ARG ORT_VERSION=1.20.1
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake curl ca-certificates \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*
# Prebuilt ONNX Runtime (CPU) from the official GitHub release.
RUN mkdir -p /opt/onnxruntime && curl -fsSL \
      https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz \
    | tar -xz -C /opt/onnxruntime --strip-components=1
WORKDIR /src
COPY app/CMakeLists.txt app/CMakeLists.txt
COPY app/src app/src
COPY app/third_party app/third_party
RUN cmake -S app -B app/build -DONNXRUNTIME_DIR=/opt/onnxruntime && cmake --build app/build -j"$(nproc)"

# ---------- stage 2: slim runtime ----------
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libopencv-core406t64 libopencv-imgproc406t64 libopencv-imgcodecs406t64 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/onnxruntime/lib /opt/onnxruntime/lib
COPY --from=build /src/app/build/freshcheck /app/app/freshcheck
COPY app/config.json /app/app/config.json
COPY app/web /app/app/web
COPY models /app/models
# Hugging Face runs containers as user 1000.
RUN useradd -m -u 1000 user || true
USER 1000
WORKDIR /app
ENV PORT=7860
EXPOSE 7860
CMD ["/app/app/freshcheck", "--config", "/app/app/config.json"]
