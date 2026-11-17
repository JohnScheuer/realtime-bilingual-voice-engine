#include "cuda_stream_transcriber.hpp"
#include "whisper.h"
#include <iostream>
#include <algorithm>
#include <sstream>

CUDAStreamTranscriber::CUDAStreamTranscriber()
    : context_(nullptr), mode_(EngineMode::InterviewEN), n_threads_(4) {}

CUDAStreamTranscriber::~CUDAStreamTranscriber() {
    if (context_) {
        whisper_free(context_);
        context_ = nullptr;
    }
}

bool CUDAStreamTranscriber::initialize(const std::string& model_path, EngineMode mode, int n_threads, const std::string& target_lang) {
    mode_ = mode;
    n_threads_ = n_threads;
    language_ = target_lang;

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;

    context_ = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!context_) {
        std::cerr << "[CUDAStreamTranscriber] Error: Failed to load GGML Whisper model from " << model_path << std::endl;
        return false;
    }

    std::cout << "[CUDAStreamTranscriber] Initialized Whisper model on GPU SM 12.0 successfully." << std::endl;
    return true;
}

void CUDAStreamTranscriber::set_mode(EngineMode mode, const std::string& target_lang) {
    mode_ = mode;
    language_ = target_lang;
}

std::string CUDAStreamTranscriber::compute_local_agreement(const std::string& current_transcript) {
    if (last_hypothesis_.empty()) {
        last_hypothesis_ = current_transcript;
        return "";
    }

    std::stringstream ss1(last_hypothesis_);
    std::stringstream ss2(current_transcript);

    std::string w1, w2;
    std::string agreed_prefix = "";

    while (ss1 >> w1 && ss2 >> w2) {
        if (w1 == w2) {
            if (!agreed_prefix.empty()) agreed_prefix += " ";
            agreed_prefix += w1;
        } else {
            break;
        }
    }

    last_hypothesis_ = current_transcript;
    return agreed_prefix;
}

TranscriptionResult CUDAStreamTranscriber::process_audio_window(const std::vector<float>& pcm_window) {
    TranscriptionResult result;
    result.inference_latency_ms = 0.0;
    result.is_final = false;

    if (!context_ || pcm_window.empty()) return result;

    auto t_start = std::chrono::high_resolution_clock::now();

    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_special = false;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.n_threads = n_threads_;
    wparams.no_context = true;
    wparams.single_segment = true;

    if (mode_ == EngineMode::InterviewEN) {
        wparams.translate = false;
        wparams.language = "en";
    } else { // WeChatZH
        wparams.translate = true;
        wparams.language = "zh";
    }

    if (whisper_full(context_, wparams, pcm_window.data(), static_cast<int>(pcm_window.size())) != 0) {
        std::cerr << "[CUDAStreamTranscriber] Failed to execute whisper_full CUDA inference." << std::endl;
        return result;
    }

    std::string current_transcript = "";
    int n_segments = whisper_full_n_segments(context_);
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(context_, i);
        if (text) {
            current_transcript += text;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.inference_latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::string agreed = compute_local_agreement(current_transcript);
    if (!agreed.empty()) {
        if (!confirmed_prefix_.empty()) confirmed_prefix_ += " ";
        confirmed_prefix_ += agreed;
    }

    result.confirmed_text = confirmed_prefix_;
    result.unconfirmed_text = current_transcript;

    return result;
}
