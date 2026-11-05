cmake_minimum_required(VERSION 3.12)

# 定义项目并指定语言为 C++ 和 CUDA
project(MixTestProject LANGUAGES C CXX CUDA)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA 包
find_package(CUDA REQUIRED)

# 设置 CUDA 编译器
# set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(HDF5 REQUIRED)

# set(GDCM_DIR "/usr/lib/x86_64-linux-gnu/gdcm-3.0")
# 查找 ITK 包
find_package(ITK 5.2 REQUIRED)
include(${ITK_USE_FILE})

# 查找 Boost
find_package(Boost REQUIRED COMPONENTS filesystem)

# 查找 DCMTK
find_package(DCMTK REQUIRED)

# 查找头文件
# file(GLOB_RECURSE INCLUDE_DIRS LIST_DIRECTORIES true ${CMAKE_SOURCE_DIR}/include/*)
# set (INCLUDE_DIRS_LIST ${CMAKE_SOURCE_DIR}/include)
# # 遍历找到的所有目录，添加到 include 路径
# foreach(dir ${INCLUDE_DIRS})
# if(IS_DIRECTORY ${dir})
# list(APPEND INCLUDE_DIRS_LIST ${dir})
# endif()
# endforeach()
# message(STATUS "Include directories: ${INCLUDE_DIRS_LIST}")

# 查找源文件
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/*.cu)

# 添加可执行文件
add_executable(nmrecon ${SRC_FILES})

# 链接 ITK 和 CUDA 库
target_link_libraries(nmrecon
    PUBLIC
    Boost::filesystem
    ${ITK_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${DCMTK_LIBRARIES}
    ${CUDA_LIBRARIES} cudart cufft)

# 设置目标的包含目录
target_include_directories(nmrecon PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${ITK_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${DCMTK_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/third_party/nlohmann_json/include
    # The historical vendored directory is absent on this workstation; the
    # existing conda installation supplies the same header-only dependency.
    /home/wengfenghua/anaconda3/include
)

# 添加动态链接共享库 (用于 PyTorch 显存级直通调用)
set(LIB_SRC_FILES ${SRC_FILES})
list(FILTER LIB_SRC_FILES EXCLUDE REGEX ".*src/nmrecon\\.cpp$")
list(FILTER LIB_SRC_FILES EXCLUDE REGEX ".*src/cpp/reconstructor\\.cpp$")
list(FILTER LIB_SRC_FILES EXCLUDE REGEX ".*src/cpp/io/dicom\\.cpp$")
list(FILTER LIB_SRC_FILES EXCLUDE REGEX ".*src/cuda/io/io\\.cu$")
add_library(nmrecon_shared SHARED ${LIB_SRC_FILES})
set_target_properties(nmrecon_shared PROPERTIES OUTPUT_NAME "nmrecon")
target_compile_definitions(nmrecon_shared PRIVATE NO_DCMTK)
target_link_libraries(nmrecon_shared
    PUBLIC
    Boost::filesystem
    ${ITK_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${CUDA_LIBRARIES} cudart cufft)
target_include_directories(nmrecon_shared PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${ITK_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${DCMTK_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/third_party/nlohmann_json/include
    /home/wengfenghua/anaconda3/include
)
set_target_properties(nmrecon_shared PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type: Debug or Release")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/debug/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/debug/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/debug/bin)
    target_compile_options(nmrecon PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g> )
    target_compile_options(nmrecon PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G --keep --use_fast_math --extended-lambda>)
    target_compile_options(nmrecon_shared PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g> )
    target_compile_options(nmrecon_shared PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G --keep --use_fast_math --extended-lambda>)
else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/release/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/release/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/release/bin)
    target_compile_options(nmrecon PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(nmrecon PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math --extended-lambda>)
    target_compile_options(nmrecon_shared PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    target_compile_options(nmrecon_shared PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math --extended-lambda>)
endif()

# 确保 CUDA 编译器设置正确
set_target_properties(nmrecon PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS nmrecon
        RUNTIME DESTINATION bin)        
# 测试模块配置（可选开关）
# option(BUILD_TESTS "Build test modules" ON)
# if(BUILD_TESTS AND DCMTK_FOUND)
#     add_subdirectory(tests)
# endif()
# enable_testing()
