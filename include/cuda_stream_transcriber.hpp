#ifndef CUDA_STREAM_TRANSCRIBER_HPP
#define CUDA_STREAM_TRANSCRIBER_HPP

#include <string>
#include <vector>
#include <memory>
#include <chrono>

struct whisper_context;

enum class EngineMode {
    InterviewEN,   // Live English Subtitles (Minimal Latency, translate = false, lang = "en")
    WeChatZH       // Mandarin -> English/Portuguese Translation (translate = true, lang = "zh")
};

struct TranscriptionResult {
    std::string confirmed_text;
    std::string unconfirmed_text;
    double inference_latency_ms;
    bool is_final;
};

class CUDAStreamTranscriber {
public:
    CUDAStreamTranscriber();
    ~CUDAStreamTranscriber();

    bool initialize(const std::string& model_path, EngineMode mode, int n_threads = 4, const std::string& target_lang = "en");

    TranscriptionResult process_audio_window(const std::vector<float>& pcm_window);

    void set_mode(EngineMode mode, const std::string& target_lang = "en");
    [[nodiscard]] bool is_initialized() const noexcept { return context_ != nullptr; }

private:
    whisper_context* context_;
    EngineMode mode_;
    std::string language_;
    int n_threads_;

    std::string last_hypothesis_;
    std::string confirmed_prefix_;

    std::string compute_local_agreement(const std::string& current_transcript);
};

#endif // CUDA_STREAM_TRANSCRIBER_HPP
