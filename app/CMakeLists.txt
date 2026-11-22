# FreshCheck C++17 server.
#   cmake -S app -B app/build -DONNXRUNTIME_DIR=/path/to/onnxruntime && cmake --build app/build -j
cmake_minimum_required(VERSION 3.16)
project(freshcheck CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV: only core, imgproc, imgcodecs are needed.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Threads REQUIRED)

# ONNX Runtime (prebuilt release tarball from GitHub). Point ONNXRUNTIME_DIR at its folder.
set(ONNXRUNTIME_DIR "/opt/onnxruntime" CACHE PATH "ONNX Runtime install folder (has include/ and lib/)")
find_path(ORT_INCLUDE onnxruntime_cxx_api.h
  PATHS ${ONNXRUNTIME_DIR}/include ${ONNXRUNTIME_DIR}/include/onnxruntime
        ${ONNXRUNTIME_DIR}/include/onnxruntime/core/session
  NO_DEFAULT_PATH)
find_library(ORT_LIB onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib NO_DEFAULT_PATH)
if(NOT ORT_INCLUDE OR NOT ORT_LIB)
  message(FATAL_ERROR "ONNX Runtime not found. Set -DONNXRUNTIME_DIR=... (see README)")
endif()
message(STATUS "ONNX Runtime: ${ORT_LIB}")

add_executable(freshcheck
  src/main.cpp src/detector.cpp src/classifier.cpp src/grader.cpp src/pipeline.cpp src/server.cpp)
target_include_directories(freshcheck PRIVATE src third_party ${ORT_INCLUDE})
target_link_libraries(freshcheck PRIVATE ${OpenCV_LIBS} ${ORT_LIB} Threads::Threads)
# Find libonnxruntime.so at runtime without LD_LIBRARY_PATH.
set_target_properties(freshcheck PROPERTIES BUILD_RPATH "${ONNXRUNTIME_DIR}/lib" INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
