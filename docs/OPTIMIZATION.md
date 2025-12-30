# MAI-UI vLLM Optimization Guide

This guide covers optimizing MAI-UI inference using vLLM across different GPU configurations.

## Table of Contents

- [Quick Start](#quick-start)
- [GPU-Specific Configurations](#gpu-specific-configurations)
- [Understanding the Parameters](#understanding-the-parameters)
- [Memory Budget Analysis](#memory-budget-analysis)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Google Colab (Free T4)

```bash
# Install dependencies
pip install vllm>=0.6.0 pillow jinja2

# Start optimized server
python -m vllm.entrypoints.openai.api_server \
    --model Tongyi-MAI/MAI-UI-2B \
    --served-model-name MAI-UI-2B \
    --port 8000 \
    --trust-remote-code \
    --dtype half \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --max-num-seqs 4 \
    --mm-processor-kwargs '{"max_pixels": 512000}'
```

### Use Pre-Made Scripts

```bash
# T4 (16GB)
./examples/server_configs/t4_server.sh 2b

# L4 (24GB)
./examples/server_configs/l4_server.sh 2b

# A100 (40GB/80GB)
./examples/server_configs/a100_server.sh 8b
```

---

## GPU-Specific Configurations

### NVIDIA T4 (16GB VRAM) - Colab Free Tier

**Architecture**: Turing (SM 7.5)

| Capability | Status |
|------------|--------|
| FP16 Tensor Cores | ✅ 65 TFLOPS |
| INT8/INT4 | ✅ 130/260 TOPS |
| BF16 | ❌ Not supported |
| FP8 | ❌ Not supported |
| FlashAttention 2 | ❌ Not supported |

**Recommended Configuration:**

```python
T4_CONFIG = {
    "model": "Tongyi-MAI/MAI-UI-2B",
    "trust_remote_code": True,
    "dtype": "half",                    # FP16 (required, no BF16)
    "max_model_len": 2048,              # Limit context for memory
    "gpu_memory_utilization": 0.90,     # 90% = 14.4GB
    "enforce_eager": True,              # Disable CUDA graphs (saves ~500MB)
    "max_num_seqs": 4,                  # Limit concurrent requests
    "mm_processor_kwargs": {
        "max_pixels": 512000,           # ~720×720 max image
    },
}
```

**Expected Performance:**
- VRAM Usage: ~10-12 GB
- Inference Latency: 1-2 seconds
- Throughput: 2-4 requests/second

---

### NVIDIA L4 (24GB VRAM) - Colab Pro

**Architecture**: Ada Lovelace (SM 8.9)

| Capability | Status |
|------------|--------|
| FP16 Tensor Cores | ✅ |
| BF16 | ✅ |
| FP8 | ✅ |
| FlashAttention 2 | ✅ |

**Recommended Configuration:**

```python
L4_CONFIG = {
    "model": "Tongyi-MAI/MAI-UI-8B",    # Can run 8B without quantization!
    "trust_remote_code": True,
    "dtype": "half",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 8,
    "mm_processor_kwargs": {
        "max_pixels": 768000,
    },
}
```

**Expected Performance:**
- VRAM Usage: ~18-20 GB
- Inference Latency: 0.8-1.5 seconds
- Throughput: 4-8 requests/second

---

### NVIDIA A100 (40GB/80GB) - Cloud/Enterprise

**Architecture**: Ampere (SM 8.0)

| Capability | Status |
|------------|--------|
| FP16/BF16 Tensor Cores | ✅ 312 TFLOPS |
| INT8 | ✅ |
| FlashAttention 2 | ✅ |

**Recommended Configuration:**

```python
A100_CONFIG = {
    "model": "Tongyi-MAI/MAI-UI-8B",
    "trust_remote_code": True,
    "dtype": "bfloat16",                # BF16 preferred on Ampere
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.92,
    "max_num_seqs": 16,
    "mm_processor_kwargs": {
        "max_pixels": 1003520,          # Full resolution
    },
}
```

**Expected Performance:**
- VRAM Usage: ~25-30 GB (40GB) / ~35-40 GB (80GB)
- Inference Latency: 0.5-1.0 seconds
- Throughput: 8-16 requests/second

---

## Understanding the Parameters

### Critical Parameters

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `dtype` | Model precision | `half` (FP16) for T4/L4, `bfloat16` for A100+ |
| `max_model_len` | Max context length | Lower = less KV cache memory |
| `gpu_memory_utilization` | % of VRAM to use | Higher = more capacity, less headroom |
| `enforce_eager` | Disable CUDA graphs | Saves 500MB-1GB |
| `max_num_seqs` | Concurrent requests | Lower = less memory per request |
| `mm_processor_kwargs.max_pixels` | Max image size | Lower = fewer vision tokens |

### Vision Token Calculation

MAI-UI uses Qwen2-VL's vision encoder:

```
Image → Resize → Patchify (14×14) → Merge (2×2) → Vision Tokens

Formula:
  resized_pixels = min(original_pixels, max_pixels)
  num_patches = resized_pixels / (14 × 14)
  num_tokens = num_patches / 4  (due to 2×2 merge)

Example (1920×1080 screenshot, max_pixels=512000):
  Resized to ~720×710 = 511,200 pixels
  Patches = 511,200 / 196 = 2,608
  Tokens = 2,608 / 4 = 652 vision tokens
```

### max_pixels Settings

| Setting | Resolution | Tokens | Use Case |
|---------|------------|--------|----------|
| 256,000 | ~500×500 | ~325 | Very low memory |
| 512,000 | ~720×720 | ~650 | T4 balanced |
| 768,000 | ~880×880 | ~975 | L4/higher |
| 1,003,520 | ~1000×1000 | ~1275 | Full quality |

---

## Memory Budget Analysis

### MAI-UI-2B on T4 (16GB)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        T4 MEMORY BUDGET (16 GB)                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Component             │  Size     │  Notes                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Model Weights (FP16)  │  ~4.0 GB  │  2B params × 2 bytes = 4GB                │
│  Vision Encoder Acts   │  ~1.5 GB  │  Temporary, per-image                     │
│  KV Cache              │  ~4.0 GB  │  Scales with max_model_len × batch        │
│  Activations           │  ~2.0 GB  │  Forward pass intermediates               │
│  CUDA/PyTorch Overhead │  ~1.5 GB  │  Fixed runtime overhead                   │
│  Safety Headroom       │  ~3.0 GB  │  For peaks and fragmentation              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  TOTAL                 │ ~16.0 GB  │  ✅ Fits T4                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### MAI-UI-8B on T4 (Requires Quantization)

```
Without quantization:
  8B params × 2 bytes = 16GB weights alone → ❌ Won't fit

With 4-bit BitsAndBytes:
  8B params × 0.5 bytes = 4GB weights → ✅ Fits with reduced context
```

---

## Performance Tuning

### Latency Optimization

1. **Reduce max_pixels** - Fewer vision tokens = faster prefill
2. **Enable CUDA graphs** (remove `--enforce-eager`) - Faster decode (if memory allows)
3. **Reduce max_tokens** - Shorter outputs

### Throughput Optimization

1. **Increase max_num_seqs** - More concurrent requests
2. **Use batch inference** - vLLM's continuous batching helps
3. **Reduce max_model_len** - More KV cache slots

### Quality Optimization

1. **Increase max_pixels** - Better visual understanding
2. **Use larger model** - 8B > 2B quality
3. **Increase max_model_len** - More context for navigation

---

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce memory usage progressively:

# Step 1: Lower max_model_len
--max-model-len 1024  # instead of 2048

# Step 2: Lower max_pixels
--mm-processor-kwargs '{"max_pixels": 256000}'

# Step 3: Lower gpu_memory_utilization
--gpu-memory-utilization 0.85

# Step 4: Reduce concurrent requests
--max-num-seqs 2

# Step 5: Enable quantization for 8B
--quantization bitsandbytes
```

### Model Won't Load

```bash
# Ensure trust_remote_code is set
--trust-remote-code

# Check vLLM version
pip install vllm>=0.6.0

# Check transformers version
pip install transformers>=4.45.0
```

### Slow Inference

```bash
# Check GPU utilization
nvidia-smi -l 1

# If low utilization, try:
# 1. Increase batch size
--max-num-seqs 8

# 2. Enable CUDA graphs (uses more memory)
# Remove --enforce-eager
```

### Wrong Coordinates

```python
# Verify image is RGB
image = image.convert("RGB")

# Check image isn't too small
if image.width < 100 or image.height < 100:
    print("Warning: Image may be too small")

# Parse output correctly - normalize from [0, 999] to [0, 1]
coord = parsed["coordinate"]
normalized = [coord[0] / 999.0, coord[1] / 999.0]
```

---

## Quick Reference

### Environment Variables

```bash
# Force attention backend (for debugging)
export VLLM_ATTENTION_BACKEND=TORCH_SDPA  # T4
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # L4/A100

# Disable logging
export VLLM_LOGGING_LEVEL=WARNING
```

### Recommended Configurations Summary

| GPU | Model | dtype | max_model_len | max_pixels | enforce_eager |
|-----|-------|-------|---------------|------------|---------------|
| T4 | 2B | half | 2048 | 512000 | True |
| T4 | 8B (4-bit) | half | 1024 | 256000 | True |
| L4 | 2B | half | 4096 | 768000 | False |
| L4 | 8B | half | 2048 | 512000 | False |
| A100 | 8B | bfloat16 | 8192 | 1003520 | False |

---

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [MAI-UI Paper](https://arxiv.org/abs/2512.22047)
- [Qwen2-VL Documentation](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

