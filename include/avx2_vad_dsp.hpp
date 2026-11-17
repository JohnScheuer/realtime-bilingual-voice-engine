#ifndef AVX2_VAD_DSP_HPP
#define AVX2_VAD_DSP_HPP

#include <immintrin.h>
#include <cstddef>
#include <cmath>
#include <algorithm>

class AVX2VoiceActivityDetector {
public:
    explicit AVX2VoiceActivityDetector(float energy_threshold = 0.012f, size_t hangover_frames = 30)
        : energy_threshold_(energy_threshold),
          hangover_limit_(hangover_frames),
          hangover_counter_(0),
          adaptive_noise_floor_(0.002f) {}

    ~AVX2VoiceActivityDetector() = default;

    inline float compute_rms_energy_avx2(const float* pcm_data, size_t num_samples) {
        if (!pcm_data || num_samples == 0) return 0.0f;

        size_t i = 0;
        __m256 sum_vec = _mm256_setzero_ps();

        for (; i + 8 <= num_samples; i += 8) {
            __m256 samples = _mm256_loadu_ps(&pcm_data[i]);
            sum_vec = _mm256_fmadd_ps(samples, samples, sum_vec);
        }

        alignas(32) float accumulated[8];
        _mm256_storeu_ps(accumulated, sum_vec);

        float total_sq_sum = accumulated[0] + accumulated[1] + accumulated[2] + accumulated[3] +
                             accumulated[4] + accumulated[5] + accumulated[6] + accumulated[7];

        for (; i < num_samples; ++i) {
            total_sq_sum += pcm_data[i] * pcm_data[i];
        }

        return std::sqrt(total_sq_sum / static_cast<float>(num_samples));
    }

    inline float compute_zero_crossing_rate(const float* pcm_data, size_t num_samples) {
        if (!pcm_data || num_samples < 2) return 0.0f;

        size_t crossings = 0;
        for (size_t i = 1; i < num_samples; ++i) {
            if ((pcm_data[i] >= 0.0f && pcm_data[i - 1] < 0.0f) ||
                (pcm_data[i] < 0.0f && pcm_data[i - 1] >= 0.0f)) {
                crossings++;
            }
        }
        return static_cast<float>(crossings) / static_cast<float>(num_samples - 1);
    }

    inline bool process_frame(const float* pcm_data, size_t num_samples, float& out_rms_energy) {
        out_rms_energy = compute_rms_energy_avx2(pcm_data, num_samples);

        if (out_rms_energy < energy_threshold_) {
            adaptive_noise_floor_ = 0.95f * adaptive_noise_floor_ + 0.05f * out_rms_energy;
        }

        const float dynamic_threshold = std::max(energy_threshold_, adaptive_noise_floor_ * 3.5f);

        if (out_rms_energy >= dynamic_threshold) {
            hangover_counter_ = hangover_limit_;
            return true;
        }

        if (hangover_counter_ > 0) {
            hangover_counter_--;
            return true;
        }

        return false;
    }

    inline void set_threshold(float threshold) noexcept { energy_threshold_ = threshold; }
    [[nodiscard]] inline float noise_floor() const noexcept { return adaptive_noise_floor_; }

private:
    float energy_threshold_;
    size_t hangover_limit_;
    size_t hangover_counter_;
    float adaptive_noise_floor_;
};

#endif // AVX2_VAD_DSP_HPP
