cmake_minimum_required(VERSION 3.22)
project(realtime_bilingual_voice_engine CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimization Flags for CPU & GPU Target Architecture
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
    $<$<COMPILE_LANGUAGE:CXX>:-mfma>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)

set(CMAKE_CUDA_ARCHITECTURES "75;80;89;120")

# Check for PulseAudio Libraries (Optional fallback)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PULSE libpulse libpulse-simple)
endif()

if(PULSE_FOUND)
    add_definitions(-DHAVE_PULSEAUDIO)
    message(STATUS "[Audio] PulseAudio found. Live microphone capture enabled.")
else()
    message(WARNING "[Audio] PulseAudio not found. Falling back to high-throughput synthetic stream generator.")
endif()

# whisper.cpp Integration via FetchContent with GGML CUDA enabled
include(FetchContent)
set(WHISPER_CUDA ON CACHE BOOL "Enable CUDA in whisper.cpp" FORCE)
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

if(PULSE_FOUND)
    include_directories(${PULSE_INCLUDE_DIRS})
endif()

# Core Transcriber Library
add_library(voice_engine_core STATIC
    src/cuda_stream_transcriber.cpp
)
target_link_libraries(voice_engine_core PUBLIC whisper)
if(PULSE_FOUND)
    target_link_libraries(voice_engine_core PUBLIC ${PULSE_LIBRARIES})
endif()

# Main Application Executable
add_executable(realtime_voice_engine
    src/main.cpp
)
target_link_libraries(realtime_voice_engine PRIVATE voice_engine_core)

# Test Runner Executable
add_executable(voice_engine_test
    tests/voice_engine_test.cpp
)
target_link_libraries(voice_engine_test PRIVATE voice_engine_core)
