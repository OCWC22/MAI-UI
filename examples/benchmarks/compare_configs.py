#!/usr/bin/env python3
"""
Compare different vLLM configurations for MAI-UI.

This script tests multiple configuration options and reports which performs best
for your specific hardware.

Usage:
    python compare_configs.py --gpu-memory 16  # T4
    python compare_configs.py --gpu-memory 24  # L4
    python compare_configs.py --gpu-memory 40  # A100-40GB
"""

import argparse
import gc
import time
from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw


@dataclass
class ConfigResult:
    """Results from testing a configuration."""
    
    name: str
    config: dict
    load_time_s: float
    inference_time_ms: float
    memory_used_gb: float
    success: bool
    error: str | None = None


def create_test_image() -> Image.Image:
    """Create a simple test image."""
    img = Image.new('RGB', (1080, 1920), color='#f5f5f5')
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1080, 80], fill='#1976D2')
    draw.rectangle([0, 280, 1080, 380], fill='white', outline='#e0e0e0')
    draw.text((40, 310), "Wi-Fi", fill='#333333')
    return img


def test_config(config: dict, test_image: Image.Image) -> ConfigResult:
    """Test a single configuration."""
    from vllm import LLM, SamplingParams
    
    name = config.pop("name", "unnamed")
    print(f"\n  Testing: {name}")
    
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        # Load model
        load_start = time.time()
        llm = LLM(**config)
        load_time = time.time() - load_start
        
        # Measure memory
        memory_used = torch.cuda.memory_allocated() / (1024**3)
        
        # Run inference
        prompt = (
            "<|im_start|>system\nYou are a GUI grounding agent.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "Click on Wi-Fi<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            stop=["<|im_end|>"],
        )
        
        # Warmup
        _ = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": test_image}}],
            sampling_params=sampling_params,
        )
        
        # Timed runs
        latencies = []
        for _ in range(3):
            start = time.perf_counter()
            _ = llm.generate(
                [{"prompt": prompt, "multi_modal_data": {"image": test_image}}],
                sampling_params=sampling_params,
            )
            latencies.append(time.perf_counter() - start)
        
        avg_latency = sum(latencies) / len(latencies) * 1000
        
        # Cleanup
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
        return ConfigResult(
            name=name,
            config=config,
            load_time_s=load_time,
            inference_time_ms=avg_latency,
            memory_used_gb=memory_used,
            success=True,
        )
        
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        
        return ConfigResult(
            name=name,
            config=config,
            load_time_s=0,
            inference_time_ms=0,
            memory_used_gb=0,
            success=False,
            error=str(e),
        )


def get_configs_for_gpu(gpu_memory_gb: int, model_size: str = "2b") -> list[dict]:
    """Get configurations to test based on GPU memory."""
    
    model = f"Tongyi-MAI/MAI-UI-{model_size.upper()}"
    base_config = {
        "model": model,
        "trust_remote_code": True,
    }
    
    configs = []
    
    if gpu_memory_gb <= 16:  # T4-class
        configs = [
            {
                **base_config,
                "name": "T4-Conservative",
                "dtype": "half",
                "max_model_len": 1024,
                "gpu_memory_utilization": 0.85,
                "enforce_eager": True,
                "mm_processor_kwargs": {"max_pixels": 256000},
            },
            {
                **base_config,
                "name": "T4-Balanced",
                "dtype": "half",
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.90,
                "enforce_eager": True,
                "mm_processor_kwargs": {"max_pixels": 512000},
            },
            {
                **base_config,
                "name": "T4-Aggressive",
                "dtype": "half",
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.95,
                "enforce_eager": True,
                "mm_processor_kwargs": {"max_pixels": 768000},
            },
            {
                **base_config,
                "name": "T4-With-CUDA-Graphs",
                "dtype": "half",
                "max_model_len": 1024,
                "gpu_memory_utilization": 0.85,
                "enforce_eager": False,  # Enable CUDA graphs
                "mm_processor_kwargs": {"max_pixels": 256000},
            },
        ]
    elif gpu_memory_gb <= 24:  # L4-class
        configs = [
            {
                **base_config,
                "name": "L4-Balanced",
                "dtype": "half",
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.90,
                "mm_processor_kwargs": {"max_pixels": 768000},
            },
            {
                **base_config,
                "name": "L4-High-Throughput",
                "dtype": "half",
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.92,
                "max_num_seqs": 8,
                "mm_processor_kwargs": {"max_pixels": 512000},
            },
        ]
    else:  # A100-class
        configs = [
            {
                **base_config,
                "name": "A100-Balanced",
                "dtype": "bfloat16",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.90,
                "mm_processor_kwargs": {"max_pixels": 1003520},
            },
            {
                **base_config,
                "name": "A100-High-Throughput",
                "dtype": "bfloat16",
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.92,
                "max_num_seqs": 16,
                "mm_processor_kwargs": {"max_pixels": 768000},
            },
        ]
    
    return configs


def main():
    parser = argparse.ArgumentParser(description="Compare vLLM configurations")
    parser.add_argument("--gpu-memory", type=int, default=16, 
                        help="GPU memory in GB (default: 16 for T4)")
    parser.add_argument("--model-size", default="2b", choices=["2b", "8b"],
                        help="Model size to test")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  MAI-UI Configuration Comparison")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        actual_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n  GPU: {gpu_name}")
        print(f"  Memory: {actual_memory:.1f} GB")
    else:
        print("\n  ❌ No GPU available!")
        return
    
    # Create test image
    test_image = create_test_image()
    
    # Get configs
    configs = get_configs_for_gpu(args.gpu_memory, args.model_size)
    
    print(f"\n  Testing {len(configs)} configurations...\n")
    print("-" * 70)
    
    # Test each config
    results = []
    for config in configs:
        result = test_config(config.copy(), test_image)
        results.append(result)
        
        if result.success:
            print(f"    ✅ Load: {result.load_time_s:.1f}s | "
                  f"Inference: {result.inference_time_ms:.0f}ms | "
                  f"Memory: {result.memory_used_gb:.1f}GB")
        else:
            print(f"    ❌ Error: {result.error}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r.success]
    
    if successful:
        # Sort by inference time
        successful.sort(key=lambda x: x.inference_time_ms)
        
        print("\n  Ranked by inference speed:\n")
        for i, r in enumerate(successful, 1):
            print(f"  {i}. {r.name}")
            print(f"     Inference: {r.inference_time_ms:.0f}ms | "
                  f"Memory: {r.memory_used_gb:.1f}GB | "
                  f"Load: {r.load_time_s:.1f}s")
        
        best = successful[0]
        print(f"\n  🏆 RECOMMENDED: {best.name}")
        print(f"     Config: {best.config}")
    else:
        print("\n  ❌ No configurations worked. Try reducing gpu_memory_utilization.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

