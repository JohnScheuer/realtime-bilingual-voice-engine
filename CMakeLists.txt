cmake_minimum_required(VERSION 3.22)
project(realtime_bilingual_voice_engine CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W3)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-O3 -mavx2 -mfma -Wall -Wextra)
endif()

set(CMAKE_CUDA_ARCHITECTURES "75;80;89;120")

# whisper.cpp Integration via FetchContent
include(FetchContent)
set(GGML_CUDA ON CACHE BOOL "Enable CUDA in GGML" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
    GIT_TAG        v1.7.1
)
FetchContent_MakeAvailable(whisper)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${whisper_SOURCE_DIR}/include
    ${whisper_SOURCE_DIR}/ggml/include
)

# Core Library
add_library(voice_engine_core STATIC
    src/cuda_stream_transcriber.cpp
)
target_link_libraries(voice_engine_core PUBLIC whisper)

# Executables
add_executable(realtime_voice_engine src/main.cpp)
target_link_libraries(realtime_voice_engine PRIVATE voice_engine_core)

add_executable(voice_engine_test tests/voice_engine_test.cpp)
target_link_libraries(voice_engine_test PRIVATE voice_engine_core)
