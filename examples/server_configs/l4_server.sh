#!/bin/bash
# MAI-UI vLLM Server Configuration for NVIDIA L4 (24GB VRAM)
# 
# Usage: ./l4_server.sh [model_size]
#   model_size: "2b" (default) or "8b"
#
# L4 is a popular choice for Colab Pro and cloud inference.
# Better than T4 (24GB vs 16GB, Ada vs Turing).

set -e

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_SIZE="${1:-2b}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# ═══════════════════════════════════════════════════════════════════════════════
# L4-OPTIMIZED SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
#
# L4 Hardware:
#   - 24 GB GDDR6 VRAM @ 300 GB/s
#   - Ada Lovelace architecture (SM 8.9)
#   - 240 Tensor Cores (4th gen, FP8/FP16/BF16/INT8)
#   - Supports: FP16, BF16, FP8, INT8 quantization, FlashAttention 2
#   - Much better than T4: +50% VRAM, better Tensor Cores, FlashAttn2

# Enable FlashAttention 2 for Ada
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

case "$MODEL_SIZE" in
    "2b"|"2B")
        echo "🚀 Starting MAI-UI-2B server (L4 optimized)..."
        MODEL="Tongyi-MAI/MAI-UI-2B"
        SERVED_NAME="MAI-UI-2B"
        MAX_MODEL_LEN=4096
        GPU_MEM_UTIL=0.90
        MAX_NUM_SEQS=8
        MAX_PIXELS=768000
        QUANTIZATION=""
        ;;
    "8b"|"8B")
        echo "🚀 Starting MAI-UI-8B server (L4 optimized)..."
        MODEL="Tongyi-MAI/MAI-UI-8B"
        SERVED_NAME="MAI-UI-8B"
        MAX_MODEL_LEN=2048
        GPU_MEM_UTIL=0.92
        MAX_NUM_SEQS=4
        MAX_PIXELS=512000
        QUANTIZATION=""  # L4 can run 8B FP16 without quantization!
        ;;
    *)
        echo "❌ Invalid model size: $MODEL_SIZE"
        echo "   Usage: $0 [2b|8b]"
        exit 1
        ;;
esac

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
echo "  Attention backend:    FlashAttention 2"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --dtype half \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --limit-mm-per-prompt '{"image": 2, "video": 0}' \
    --mm-processor-kwargs "{\"min_pixels\": 784, \"max_pixels\": $MAX_PIXELS}" \
    $QUANTIZATION

