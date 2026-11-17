#ifndef TERMINAL_HUD_HPP
#define TERMINAL_HUD_HPP

#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#endif

class TerminalHUD {
public:
    TerminalHUD() : cycle_count_(0) {
#ifdef _WIN32
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut != INVALID_HANDLE_VALUE) {
            DWORD dwMode = 0;
            GetConsoleMode(hOut, &dwMode);
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
#endif
    }

    void render(const std::string& mode_str,
                 bool vad_active,
                 float rms_energy,
                 const std::string& confirmed_text,
                 const std::string& active_text,
                 double latency_ms) {
        
        latency_history_.push_back(latency_ms);
        if (latency_history_.size() > 500) {
            latency_history_.erase(latency_history_.begin());
        }

        double p50 = compute_percentile(50.0);
        double p90 = compute_percentile(90.0);
        double p99 = compute_percentile(99.0);

        std::cout << "\033[2J\033[1;1H";

        std::cout << "\033[1;36m================================================================================\033[0m\n";
        std::cout << "\033[1;32m  REAL-TIME BILINGUAL VOICE ENGINE (WINDOWS / LINUX C++20 / CUDA SM 12.0)\033[0m\n";
        std::cout << "  Author: Joao Felipe de Souza (github.com/JohnScheuer)\n";
        std::cout << "\033[1;36m================================================================================\033[0m\n\n";

        std::cout << "  \033[1;35mOPERATIONAL MODE :\033[0m \033[1;33m" << mode_str << "\033[0m\n";
        std::cout << "  \033[1;35mVAD STATE        :\033[0m ";
        if (vad_active) {
            std::cout << "\033[1;32m[ VOICE ACTIVE ]\033[0m ";
        } else {
            std::cout << "\033[1;30m[ SILENCE GATE ]\033[0m ";
        }
        std::cout << " (RMS Energy: " << std::fixed << std::setprecision(4) << rms_energy << ")\n\n";

        std::cout << "  \033[1;36mLATENCY TELEMETRY (End-to-End Frame Processing):\033[0m\n";
        std::cout << "    Last Frame : \033[1;32m" << std::setw(6) << std::fixed << std::setprecision(2) << latency_ms << " ms\033[0m | ";
        std::cout << "P50: \033[1;36m" << std::setw(6) << p50 << " ms\033[0m | ";
        std::cout << "P90: \033[1;33m" << std::setw(6) << p90 << " ms\033[0m | ";
        std::cout << "P99: \033[1;31m" << std::setw(6) << p99 << " ms\033[0m\n\n";

        std::cout << "\033[1;36m--------------------------------------------------------------------------------\033[0m\n";
        std::cout << "  \033[1;37mLIVE SUBTITLE STREAM (WASAPI Loopback / Mic Streamed):\033[0m\n";
        std::cout << "\033[1;36m--------------------------------------------------------------------------------\033[0m\n\n";

        std::cout << "  \033[1;32m" << confirmed_text << "\033[0m";
        if (!active_text.empty()) {
            std::cout << " \033[1;33m" << active_text << "\033[0m";
        }
        std::cout << "\n\n";

        std::cout << "\033[1;36m================================================================================\033[0m\n";
        std::cout << "  \033[1;30mPress Ctrl+C to terminate engine cleanly.\033[0m\n";
        std::cout.flush();

        cycle_count_++;
    }

private:
    std::vector<double> latency_history_;
    uint64_t cycle_count_;

    double compute_percentile(double p) {
        if (latency_history_.empty()) return 0.0;
        std::vector<double> sorted = latency_history_;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>((p / 100.0) * (sorted.size() - 1));
        return sorted[idx];
    }
};

#endif // TERMINAL_HUD_HPP
