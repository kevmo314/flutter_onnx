cmake_minimum_required(VERSION 3.18)
project(flutter_onnx_nnapi LANGUAGES CXX)
add_library(flutter_onnx_nnapi SHARED nnapi.cpp)
# Permit declarations from API 29 in decltype; every NNAPI call uses dlsym.
target_compile_definitions(flutter_onnx_nnapi PRIVATE __ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__)
target_link_libraries(flutter_onnx_nnapi PRIVATE dl)
