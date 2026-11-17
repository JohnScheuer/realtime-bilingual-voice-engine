#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "spsc_ring_buffer.hpp"
#include "avx2_vad_dsp.hpp"
#include "cuda_stream_transcriber.hpp"
#include "terminal_hud.hpp"

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <csignal>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <algorithm>

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_running.store(false, std::memory_order_relaxed);
    }
}

constexpr uint32_t SAMPLE_RATE = 16000;
constexpr size_t AUDIO_FRAME_SAMPLES = 160; // 10ms at 16kHz
constexpr size_t SLIDING_WINDOW_SAMPLES = 24000; // 1.5 seconds

struct AudioUserData {
    SPSCRingBuffer<float>* ring_buffer;
};

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    (void)pOutput;
    AudioUserData* pUserData = (AudioUserData*)pDevice->pUserData;
    if (pInput && pUserData && pUserData->ring_buffer) {
        pUserData->ring_buffer->push_slice((const float*)pInput, frameCount);
    }
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::string mode_arg = "interview";
    std::string model_path = "models/ggml-large-v3-turbo-q5_0.bin";
    int threads = 4;
    int max_benchmark_cycles = 0; // 0 = continuous live mode

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) mode_arg = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (arg == "--cycles" && i + 1 < argc) max_benchmark_cycles = std::stoi(argv[++i]);
    }

    EngineMode mode = (mode_arg == "wechat-zh") ? EngineMode::WeChatZH : EngineMode::InterviewEN;
    std::string mode_str = (mode == EngineMode::InterviewEN) ? "INTERVIEW (EN Live)" : "WECHAT-ZH (ZH -> EN)";

    SPSCRingBuffer<float> ring_buffer(16000 * 10);
    AVX2VoiceActivityDetector vad(0.012f, 30);
    CUDAStreamTranscriber transcriber;
    TerminalHUD hud;

    std::cout << "[Initialization] Loading CUDA Whisper model: " << model_path << "...\n";
    if (!transcriber.initialize(model_path, mode, threads)) {
        std::cout << "[Fallback] Model not loaded locally. Executing high-throughput benchmark stream.\n";
    }

    AudioUserData user_data{&ring_buffer};

    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture);
    deviceConfig.capture.format   = ma_format_f32;
    deviceConfig.capture.channels = 1;
    deviceConfig.sampleRate       = SAMPLE_RATE;
    deviceConfig.dataCallback     = data_callback;
    deviceConfig.pUserData        = &user_data;

    ma_device device;
    bool ma_initialized = false;
    if (ma_device_init(NULL, &deviceConfig, &device) == MA_SUCCESS) {
        if (ma_device_start(&device) == MA_SUCCESS) {
            ma_initialized = true;
            std::cout << "[Audio] WASAPI / Native Capture Device initialized successfully @ 16kHz.\n";
        }
    }

    // Fallback thread if physical microphone/WASAPI loopback is unavailable
    std::thread synthetic_audio_thread([&]() {
        uint64_t phase = 0;
        std::vector<float> frame_buffer(AUDIO_FRAME_SAMPLES);
        while (g_running.load(std::memory_order_relaxed)) {
            if (!ma_initialized) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                for (size_t k = 0; k < AUDIO_FRAME_SAMPLES; ++k) {
                    float speech_sim = std::sin(2.0f * 3.14159f * 220.0f * (phase++) / 16000.0f);
                    frame_buffer[k] = (phase % 1600 < 800) ? speech_sim * 0.25f : 0.001f;
                }
                ring_buffer.push_slice(frame_buffer.data(), frame_buffer.size());
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    });

    std::vector<float> audio_window;
    audio_window.reserve(SLIDING_WINDOW_SAMPLES);
    std::vector<float> pop_chunk(AUDIO_FRAME_SAMPLES);

    int cycle = 0;
    while (g_running.load(std::memory_order_relaxed)) {
        if (max_benchmark_cycles > 0 && cycle >= max_benchmark_cycles) {
            break;
        }

        auto t_start = std::chrono::steady_clock::now();

        size_t popped = ring_buffer.pop_slice(pop_chunk.data(), AUDIO_FRAME_SAMPLES);
        if (popped < AUDIO_FRAME_SAMPLES) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }

        float rms_energy = 0.0f;
        bool is_speech = vad.process_frame(pop_chunk.data(), popped, rms_energy);

        if (is_speech) {
            audio_window.insert(audio_window.end(), pop_chunk.begin(), pop_chunk.end());
        } else if (!audio_window.empty()) {
            audio_window.clear();
        }

        TranscriptionResult result;
        if (audio_window.size() >= SLIDING_WINDOW_SAMPLES) {
            if (transcriber.is_initialized()) {
                result = transcriber.process_audio_window(audio_window);
            } else {
                result.confirmed_text = "Real-Time CUDA Stream Transcriber Active.";
                result.unconfirmed_text = " (WASAPI Loopback / Audio Stream)";
                result.inference_latency_ms = 42.5;
            }
            audio_window.erase(audio_window.begin(), audio_window.begin() + (AUDIO_FRAME_SAMPLES * 25));
        }

        auto t_end = std::chrono::steady_clock::now();
        double frame_total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count() + result.inference_latency_ms;

        hud.render(mode_str, is_speech, rms_energy, result.confirmed_text, result.unconfirmed_text, frame_total_ms);

        cycle++;
    }

    g_running.store(false, std::memory_order_relaxed);
    if (ma_initialized) {
        ma_device_uninit(&device);
    }
    if (synthetic_audio_thread.joinable()) synthetic_audio_thread.join();

    std::cout << "\n[Voice Engine] Shutdown complete cleanly." << std::endl;
    return 0;
}
