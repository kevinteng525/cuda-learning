cmake_minimum_required(VERSION 3.17.0)
project(testdebug VERSION 0.1.0 LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)

# 添加sample目录
add_subdirectory(sample)

set(CMAKE_CUDA_STANDARD 11)

# 为main.cu创建可执行文件
add_executable(testdebug main.cu)
target_link_libraries(testdebug PRIVATE CUDA::cudart)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(testdebug PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()