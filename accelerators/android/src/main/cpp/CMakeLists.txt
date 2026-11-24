cmake_minimum_required(VERSION 3.18)
project(flutter_onnx_nnapi LANGUAGES CXX)
add_library(flutter_onnx_nnapi SHARED nnapi.cpp)
target_link_libraries(flutter_onnx_nnapi PRIVATE dl)
