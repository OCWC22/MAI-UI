# MAI-UI Examples

This directory contains optimized configurations and examples for running MAI-UI with vLLM across different GPU types.

## Directory Structure

```
examples/
├── t4_optimized/           # Google Colab (Free T4) examples
│   └── mai_ui_t4_colab.py  # Complete Colab notebook (copy cells)
├── server_configs/         # Pre-configured server scripts
│   ├── t4_server.sh        # T4 (16GB) - Colab Free
│   ├── l4_server.sh        # L4 (24GB) - Colab Pro  
│   └── a100_server.sh      # A100 (40/80GB) - Cloud
└── benchmarks/             # Performance testing
    ├── benchmark_inference.py   # Latency/throughput benchmark
    └── compare_configs.py       # Compare different configs
```

## Quick Start

### 1. Google Colab (Free T4)

Copy the cells from `t4_optimized/mai_ui_t4_colab.py` into a Colab notebook:

1. Open [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → GPU (T4)
3. Copy cells sequentially

### 2. Server Mode

Start an OpenAI-compatible API server:

```bash
# Make scripts executable
chmod +x examples/server_configs/*.sh

# T4 with 2B model
./examples/server_configs/t4_server.sh 2b

# L4 with 8B model
./examples/server_configs/l4_server.sh 8b

# A100 with 8B model
./examples/server_configs/a100_server.sh 8b
```

### 3. Benchmark Your Setup

```bash
# Benchmark via API
python examples/benchmarks/benchmark_inference.py \
    --model MAI-UI-2B \
    --url http://localhost:8000/v1 \
    --runs 5

# Compare different configurations
python examples/benchmarks/compare_configs.py --gpu-memory 16
```

## GPU Compatibility Matrix

| GPU | VRAM | MAI-UI-2B | MAI-UI-8B | Recommended Script |
|-----|------|-----------|-----------|-------------------|
| T4 | 16GB | ✅ FP16 | ⚠️ 4-bit quant | `t4_server.sh` |
| L4 | 24GB | ✅ FP16 | ✅ FP16 | `l4_server.sh` |
| A10G | 24GB | ✅ FP16 | ✅ FP16 | `l4_server.sh` |
| A100-40GB | 40GB | ✅ BF16 | ✅ BF16 | `a100_server.sh` |
| A100-80GB | 80GB | ✅ BF16 | ✅ BF16 | `a100_server.sh` |
| H100 | 80GB | ✅ BF16 | ✅ BF16 | `a100_server.sh` |

## Configuration Reference

### T4-Optimized Settings

```python
{
    "model": "Tongyi-MAI/MAI-UI-2B",
    "trust_remote_code": True,
    "dtype": "half",                    # FP16 only (no BF16 on Turing)
    "max_model_len": 2048,              # Reduced for memory
    "gpu_memory_utilization": 0.90,
    "enforce_eager": True,              # Saves 500MB
    "max_num_seqs": 4,
    "mm_processor_kwargs": {"max_pixels": 512000},
}
```

### Key Optimizations Explained

| Setting | Purpose | Memory Impact |
|---------|---------|---------------|
| `dtype=half` | Use FP16 instead of FP32 | -50% weights |
| `max_model_len=2048` | Limit context length | -40% KV cache |
| `enforce_eager=True` | Disable CUDA graphs | -500MB |
| `max_pixels=512000` | Limit image resolution | -30% vision tokens |
| `max_num_seqs=4` | Limit concurrent requests | -50% activations |

## Expected Performance

### T4 (Google Colab Free)

| Metric | MAI-UI-2B | MAI-UI-8B (4-bit) |
|--------|-----------|-------------------|
| VRAM Used | ~10-12 GB | ~12-14 GB |
| First Token | 500-800 ms | 800-1200 ms |
| Total Latency | 1-2 sec | 2-4 sec |
| Throughput | 2-4 req/s | 1-2 req/s |

### L4 (Colab Pro)

| Metric | MAI-UI-2B | MAI-UI-8B |
|--------|-----------|-----------|
| VRAM Used | ~8-10 GB | ~18-20 GB |
| First Token | 300-500 ms | 500-800 ms |
| Total Latency | 0.5-1 sec | 0.8-1.5 sec |
| Throughput | 4-8 req/s | 3-5 req/s |

## Troubleshooting

### OOM (Out of Memory)

```bash
# Reduce settings progressively:
--max-model-len 1024
--mm-processor-kwargs '{"max_pixels": 256000}'
--gpu-memory-utilization 0.85
--max-num-seqs 2
```

### Slow Inference

```bash
# Check GPU utilization
nvidia-smi -l 1

# If low, increase batch size
--max-num-seqs 8
```

### Model Won't Load

```bash
# Ensure correct flags
--trust-remote-code

# Check versions
pip install vllm>=0.6.0 transformers>=4.45.0
```

## More Documentation

- [Full Optimization Guide](../docs/OPTIMIZATION.md)
- [Main README](../README.md)
- [vLLM Documentation](https://docs.vllm.ai/)

