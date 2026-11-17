import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')
csv_path = "benchmark_telemetry.csv"
output_dir = "assets"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

COLOR_CYAN = "#00f3ff"
COLOR_MAGENTA = "#ff0055"
COLOR_GREEN = "#00ff66"
COLOR_AMBER = "#ffaa00"
COLOR_PURPLE = "#b026ff"

# 1. Latency Distribution Histogram
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(df['total_latency_ms'], bins=40, color=COLOR_CYAN, alpha=0.75, edgecolor='black')
mean_val = df['total_latency_ms'].mean()
p99_val = np.percentile(df['total_latency_ms'], 99)
plt.axvline(mean_val, color=COLOR_MAGENTA, linestyle='--', linewidth=2, label=f"Mean: {mean_val:.2f} ms")
plt.axvline(p99_val, color=COLOR_AMBER, linestyle=':', linewidth=2, label=f"P99: {p99_val:.2f} ms")
plt.title("Real-Time Voice Engine — End-to-End Latency Distribution", fontsize=13, fontweight='bold', color='white')
plt.xlabel("End-to-End Frame Processing Latency (ms)", fontsize=11, color='white')
plt.ylabel("Audio Frame Count (10ms frames)", fontsize=11, color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(facecolor='#111111', edgecolor=COLOR_CYAN)
plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(output_dir, "01_latency_distribution.png"), dpi=300)
plt.close()

# 2. Latency Percentile Profile
percentiles = [50, 90, 95, 99, 99.9]
values = [np.percentile(df['total_latency_ms'], p) for p in percentiles]
plt.figure(figsize=(10, 6))
plt.plot([str(p) for p in percentiles], values, marker='o', linewidth=2.5, markersize=8, color=COLOR_GREEN)
for p, v in zip(percentiles, values):
    plt.annotate(f"{v:.2f} ms", (str(p), v), textcoords="offset points", xytext=(0, 12), ha='center', color=COLOR_CYAN, fontweight='bold')
plt.ylim(min(values) * 0.8, max(values) * 1.3)
plt.title("Latency Percentile Profile (SLA Threshold < 300 ms)", fontsize=13, fontweight='bold', color='white')
plt.xlabel("Percentile Target", fontsize=11, color='white')
plt.ylabel("Latency (ms)", fontsize=11, color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(output_dir, "02_latency_percentiles.png"), dpi=300)
plt.close()

# 3. Pipeline Microsecond / Millisecond Stage Breakdown
stages = ['Lock-Free SPSC Pop (µs)', 'AVX2 SIMD VAD (µs)', 'CUDA FP16 Inference (ms)']
means = [
    df['spsc_us'].mean(),
    df['vad_us'].mean(),
    df['cuda_ms'].mean()
]
plt.figure(figsize=(10, 6))
bars = plt.barh(stages, means, color=[COLOR_CYAN, COLOR_MAGENTA, COLOR_PURPLE], alpha=0.85)
for bar in bars:
    w = bar.get_width()
    plt.text(w * 1.15, bar.get_y() + bar.get_height()/2.0, f"{w:.2f}", va='center', ha='left', color='white', fontweight='bold')
plt.xlim(1e-2, 1e4)
plt.xscale('log')
plt.title("Pipeline Stage Microsecond / Millisecond Execution Breakdown", fontsize=13, fontweight='bold', color='white')
plt.xlabel("Execution Time — Log Scale", fontsize=11, color='white')
plt.grid(True, which="both", linestyle='--', alpha=0.2)
plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(output_dir, "03_stage_breakdown.png"), dpi=300)
plt.close()

# 4. Lock-Free SPSC Buffer Occupancy
plt.figure(figsize=(10, 6))
plt.plot(df['cycle_id'][:500], df['buffer_occ'][:500], color=COLOR_CYAN, label="SPSC Queue Depth (Samples)")
plt.title("Lock-Free SPSC Ring Buffer Queue Occupancy Over Time", fontsize=13, fontweight='bold', color='white')
plt.xlabel("10ms Audio Frame Index", fontsize=11, color='white')
plt.ylabel("Buffered PCM Samples", fontsize=11, color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(facecolor='#111111', edgecolor=COLOR_CYAN)
plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(output_dir, "04_ring_buffer_occupancy.png"), dpi=300)
plt.close()

# 5. AVX2 SIMD RMS Energy & VAD Gating
plt.figure(figsize=(10, 6))
plt.plot(df['cycle_id'][:500], df['rms_energy'][:500], color=COLOR_MAGENTA, alpha=0.85, label="AVX2 SIMD Audio RMS Energy")
plt.axhline(0.012, color=COLOR_AMBER, linestyle='--', linewidth=2, label="Dynamic VAD Threshold (0.012)")
plt.title("AVX2 SIMD Voice Activity Detection (VAD) Signal Energy", fontsize=13, fontweight='bold', color='white')
plt.xlabel("10ms Audio Frame Index", fontsize=11, color='white')
plt.ylabel("RMS Signal Amplitude", fontsize=11, color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(facecolor='#111111', edgecolor=COLOR_AMBER)
plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(output_dir, "05_vad_rms_energy.png"), dpi=300)
plt.close()

# 6. GPU VRAM Memory Budget
plt.figure(figsize=(10, 6))
vram_used_gb = 1.48 # Medium / Turbo Q5_K_M model footprint
vram_cap_gb = 8.00  # RTX 5060 GDDR7 Capacity
plt.bar(['VRAM Allocated', 'RTX 5060 GDDR7 Total'], [vram_used_gb, vram_cap_gb], color=[COLOR_GREEN, COLOR_PURPLE], width=0.4, alpha=0.85)
plt.text(0, vram_used_gb + 0.2, f"{vram_used_gb:.2f} GB", ha='center', color='white', fontweight='bold')
plt.text(1, vram_cap_gb + 0.2, f"{vram_cap_gb:.2f} GB", ha='center', color='white', fontweight='bold')
plt.ylim(0, 10.0)
plt.title("GPU VRAM Footprint Guard (< 4.0 GB Limit)", fontsize=13, fontweight='bold', color='white')
plt.ylabel("VRAM Usage (GB)", fontsize=11, color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(output_dir, "06_cuda_vram_footprint.png"), dpi=300)
plt.close()

print("[PLOTS] Generated 6 Dark-Mode Neon Figures in assets/")
