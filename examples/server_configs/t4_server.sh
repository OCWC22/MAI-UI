#!/bin/bash
# MAI-UI vLLM Server Configuration for NVIDIA T4 (16GB VRAM)
# 
# Usage: ./t4_server.sh [model_size]
#   model_size: "2b" (default) or "8b"
#
# This script starts a vLLM OpenAI-compatible API server with T4-optimized settings.

set -e

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_SIZE="${1:-2b}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# ═══════════════════════════════════════════════════════════════════════════════
# T4-OPTIMIZED SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
#
# T4 Hardware:
#   - 16 GB GDDR6 VRAM @ 320 GB/s
#   - Turing architecture (SM 7.5)
#   - 320 Tensor Cores (1st gen, FP16/INT8)
#   - Supports: FP16, INT8/INT4 quantization, CUDA graphs
#   - Does NOT support: BF16, FP8, FlashAttention 2
#
# Memory Budget for MAI-UI-2B:
#   Model weights (FP16):     ~4.0 GB (25%)
#   Vision encoder activations: ~1.5 GB (10%)
#   KV Cache:                 ~4.0 GB (25%)
#   Activations:              ~2.0 GB (12%)
#   CUDA/PyTorch overhead:    ~1.5 GB (10%)
#   Safety headroom:          ~3.0 GB (18%)
#   ─────────────────────────────────────
#   TOTAL:                    ~16.0 GB (100%)

case "$MODEL_SIZE" in
    "2b"|"2B")
        echo "🚀 Starting MAI-UI-2B server (T4 optimized)..."
        MODEL="Tongyi-MAI/MAI-UI-2B"
        SERVED_NAME="MAI-UI-2B"
        MAX_MODEL_LEN=2048
        GPU_MEM_UTIL=0.90
        MAX_NUM_SEQS=4
        MAX_PIXELS=512000
        QUANTIZATION=""
        ;;
    "8b"|"8B")
        echo "🚀 Starting MAI-UI-8B server (T4 + 4-bit quantization)..."
        MODEL="Tongyi-MAI/MAI-UI-8B"
        SERVED_NAME="MAI-UI-8B"
        MAX_MODEL_LEN=1024
        GPU_MEM_UTIL=0.95
        MAX_NUM_SEQS=2
        MAX_PIXELS=256000
        QUANTIZATION="--quantization bitsandbytes --load-format bitsandbytes"
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
    --enforce-eager \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --mm-processor-kwargs "{\"min_pixels\": 784, \"max_pixels\": $MAX_PIXELS}" \
    $QUANTIZATION

