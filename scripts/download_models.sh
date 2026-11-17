#!/bin/bash
set -e

MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"

echo "========================================================"
echo "  Downloading GGML Whisper Models for CUDA Engine"
echo "========================================================"

# Default model: ggml-large-v3-turbo-q5_0.bin
TURBO_MODEL="$MODELS_DIR/ggml-large-v3-turbo-q5_0.bin"
if [ ! -f "$TURBO_MODEL" ]; then
    echo "[Download] Fetching ggml-large-v3-turbo-q5_0.bin (~1.5 GB)..."
    curl -L -o "$TURBO_MODEL" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
else
    echo "[Exist] ggml-large-v3-turbo-q5_0.bin is already present."
fi

# Medium model: ggml-medium.bin
MEDIUM_MODEL="$MODELS_DIR/ggml-medium.bin"
if [ ! -f "$MEDIUM_MODEL" ]; then
    echo "[Download] Fetching ggml-medium.bin (~1.5 GB)..."
    curl -L -o "$MEDIUM_MODEL" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"
else
    echo "[Exist] ggml-medium.bin is already present."
fi

echo "========================================================"
echo "  All models downloaded and ready in $MODELS_DIR"
echo "========================================================"
