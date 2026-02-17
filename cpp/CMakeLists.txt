cmake_minimum_required(VERSION 3.16)
project(hailo_inference_pipeline CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(HailoRT REQUIRED)

# nlohmann/json - prefer system package, fallback to FetchContent
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/inference_pipeline.cpp
    src/inference_utils.cpp
    src/postprocess/classification.cpp
    src/postprocess/nms_on_host.cpp
    src/postprocess/palm_detection.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    HailoRT::libhailort
    nlohmann_json::nlohmann_json
    pthread
)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -O2)
