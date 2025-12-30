#!/bin/bash
# MAI-UI vLLM Server Configuration for NVIDIA A100 (40GB/80GB VRAM)
# 
# Usage: ./a100_server.sh [model_size] [vram_size]
#   model_size: "2b", "8b" (default), or "32b"
#   vram_size: "40" (default) or "80"
#
# This script starts a vLLM OpenAI-compatible API server with A100-optimized settings.

set -e

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_SIZE="${1:-8b}"
VRAM_SIZE="${2:-40}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# ═══════════════════════════════════════════════════════════════════════════════
# A100-OPTIMIZED SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
#
# A100 Hardware:
#   - 40GB or 80GB HBM2e VRAM @ 1.6-2.0 TB/s
#   - Ampere architecture (SM 8.0)
#   - 432 Tensor Cores (3rd gen, FP16/BF16/TF32/INT8)
#   - Supports: FP16, BF16, INT8 quantization, FlashAttention 2, CUDA graphs
#   - Does NOT support: FP8 (Hopper only)

# Enable FlashAttention 2 for Ampere+
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

case "$MODEL_SIZE" in
    "2b"|"2B")
        echo "🚀 Starting MAI-UI-2B server (A100 optimized)..."
        MODEL="Tongyi-MAI/MAI-UI-2B"
        SERVED_NAME="MAI-UI-2B"
        MAX_MODEL_LEN=8192
        MAX_NUM_SEQS=32
        MAX_PIXELS=1280000
        QUANTIZATION=""
        TP_SIZE=1
        ;;
    "8b"|"8B")
        echo "🚀 Starting MAI-UI-8B server (A100 optimized)..."
        MODEL="Tongyi-MAI/MAI-UI-8B"
        SERVED_NAME="MAI-UI-8B"
        MAX_MODEL_LEN=8192
        MAX_NUM_SEQS=16
        MAX_PIXELS=1003520
        QUANTIZATION=""
        TP_SIZE=1
        ;;
    "32b"|"32B")
        echo "🚀 Starting MAI-UI-32B server (A100 optimized)..."
        MODEL="Tongyi-MAI/MAI-UI-32B"
        SERVED_NAME="MAI-UI-32B"
        MAX_MODEL_LEN=4096
        MAX_NUM_SEQS=8
        MAX_PIXELS=768000
        QUANTIZATION=""
        if [ "$VRAM_SIZE" == "40" ]; then
            TP_SIZE=2  # Need 2 GPUs for 32B on 40GB
            echo "⚠️  32B requires 2x A100-40GB (tensor parallel)"
        else
            TP_SIZE=1
        fi
        ;;
    *)
        echo "❌ Invalid model size: $MODEL_SIZE"
        echo "   Usage: $0 [2b|8b|32b] [40|80]"
        exit 1
        ;;
esac

# Adjust GPU memory utilization based on VRAM size
if [ "$VRAM_SIZE" == "80" ]; then
    GPU_MEM_UTIL=0.92
else
    GPU_MEM_UTIL=0.90
fi

# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCH SERVER
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Configuration:"
echo "  ─────────────────────────────────────────────────────────────────────────────"
echo "  Model:                $MODEL"
echo "  Served name:          $SERVED_NAME"
echo "  Host:Port:            $HOST:$PORT"
echo "  Max context length:   $MAX_MODEL_LEN tokens"
echo "  GPU memory util:      ${GPU_MEM_UTIL}%"
echo "  Max concurrent reqs:  $MAX_NUM_SEQS"
echo "  Max image pixels:     $MAX_PIXELS"
echo "  Tensor parallel:      $TP_SIZE GPU(s)"
echo "  Attention backend:    FlashAttention 2"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size "$TP_SIZE" \
    --limit-mm-per-prompt '{"image": 4, "video": 0}' \
    --mm-processor-kwargs "{\"min_pixels\": 784, \"max_pixels\": $MAX_PIXELS}" \
    $QUANTIZATION

