# sample目录的CMakeLists.txt
# 为vector_add.cu创建可执行文件
add_executable(vector_add vector_add.cu)
target_link_libraries(vector_add PRIVATE CUDA::cudart)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(vector_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()