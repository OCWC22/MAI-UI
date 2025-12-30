#!/usr/bin/env python3
"""
MAI-UI Inference Benchmark Script

Benchmarks MAI-UI grounding and navigation performance across different configurations.

Usage:
    python benchmark_inference.py --model MAI-UI-2B --url http://localhost:8000/v1

Requirements:
    pip install openai pillow tqdm tabulate
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw

# Try to import optional dependencies
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single test."""
    
    instruction: str
    latencies: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    success_count: int = 0
    total_count: int = 0
    
    @property
    def mean_latency_ms(self) -> float:
        return statistics.mean(self.latencies) * 1000 if self.latencies else 0
    
    @property
    def std_latency_ms(self) -> float:
        return statistics.stdev(self.latencies) * 1000 if len(self.latencies) > 1 else 0
    
    @property
    def p50_latency_ms(self) -> float:
        return statistics.median(self.latencies) * 1000 if self.latencies else 0
    
    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)] * 1000
    
    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0
    
    @property
    def mean_tokens(self) -> float:
        return statistics.mean(self.tokens) if self.tokens else 0
    
    @property
    def tokens_per_second(self) -> float:
        if not self.latencies or not self.tokens:
            return 0
        return self.mean_tokens / (self.mean_latency_ms / 1000)


def create_test_images() -> dict[str, Image.Image]:
    """Create synthetic test images for benchmarking."""
    images = {}
    
    # Mobile settings screenshot
    mobile = Image.new('RGB', (1080, 1920), color='#f5f5f5')
    draw = ImageDraw.Draw(mobile)
    draw.rectangle([0, 0, 1080, 80], fill='#1976D2')
    draw.text((40, 30), "9:41", fill='white')
    draw.rectangle([0, 80, 1080, 200], fill='#2196F3')
    draw.text((40, 120), "Settings", fill='white')
    for i, (label, y) in enumerate([("Wi-Fi", 280), ("Bluetooth", 400), ("Cellular", 520)]):
        draw.rectangle([0, y, 1080, y + 100], fill='white', outline='#e0e0e0')
        draw.text((40, y + 35), label, fill='#333333')
    images['mobile'] = mobile
    
    # Desktop login form
    desktop = Image.new('RGB', (1920, 1080), color='#f0f0f0')
    draw = ImageDraw.Draw(desktop)
    draw.rectangle([0, 0, 1920, 40], fill='#4a90d9')
    draw.text((20, 10), "Login", fill='white')
    draw.rectangle([660, 300, 1260, 700], fill='white', outline='#ccc')
    draw.rectangle([710, 580, 860, 630], fill='#4CAF50')
    draw.text((760, 595), "Login", fill='white')
    draw.rectangle([880, 580, 1030, 630], fill='#f44336')
    draw.text((930, 595), "Cancel", fill='white')
    images['desktop'] = desktop
    
    # High-res screenshot (4K)
    highres = Image.new('RGB', (3840, 2160), color='#ffffff')
    draw = ImageDraw.Draw(highres)
    draw.rectangle([100, 100, 500, 200], fill='#2196F3')
    draw.text((250, 140), "Button", fill='white')
    images['highres'] = highres
    
    return images


def benchmark_with_vllm_python(
    config: dict[str, Any],
    images: dict[str, Image.Image],
    instructions: list[str],
    num_runs: int = 5,
) -> list[BenchmarkResult]:
    """Benchmark using vLLM Python API directly."""
    from vllm import LLM, SamplingParams
    
    print(f"\n🔧 Initializing vLLM with config:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    llm = LLM(**config)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    
    results = []
    
    for instruction in tqdm(instructions, desc="Benchmarking"):
        result = BenchmarkResult(instruction=instruction)
        image = images['mobile']
        
        prompt = (
            "<|im_start|>system\nYou are a GUI grounding agent.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        for _ in range(num_runs):
            inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}
            
            start = time.perf_counter()
            outputs = llm.generate([inputs], sampling_params=sampling_params)
            latency = time.perf_counter() - start
            
            result.latencies.append(latency)
            result.tokens.append(len(outputs[0].outputs[0].token_ids))
            result.total_count += 1
            
            # Check if we got a valid coordinate
            text = outputs[0].outputs[0].text
            if "coordinate" in text or "pyautogui" in text:
                result.success_count += 1
        
        results.append(result)
    
    return results


def benchmark_with_openai_api(
    url: str,
    model_name: str,
    images: dict[str, Image.Image],
    instructions: list[str],
    num_runs: int = 5,
) -> list[BenchmarkResult]:
    """Benchmark using OpenAI-compatible API."""
    import base64
    from io import BytesIO
    from openai import OpenAI
    
    client = OpenAI(base_url=url, api_key="empty")
    
    def image_to_base64(img: Image.Image) -> str:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    results = []
    
    for instruction in tqdm(instructions, desc="Benchmarking"):
        result = BenchmarkResult(instruction=instruction)
        image = images['mobile']
        b64_image = image_to_base64(image)
        
        for _ in range(num_runs):
            start = time.perf_counter()
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a GUI grounding agent."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                                {"type": "text", "text": instruction},
                            ],
                        },
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                
                latency = time.perf_counter() - start
                result.latencies.append(latency)
                
                text = response.choices[0].message.content
                tokens = response.usage.completion_tokens if response.usage else 0
                result.tokens.append(tokens)
                result.total_count += 1
                
                if "coordinate" in text or "pyautogui" in text:
                    result.success_count += 1
                    
            except Exception as e:
                print(f"  Error: {e}")
                result.total_count += 1
        
        results.append(result)
    
    return results


def print_results(results: list[BenchmarkResult], title: str = "Benchmark Results"):
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")
    
    headers = [
        "Instruction",
        "Mean (ms)",
        "Std (ms)",
        "P50 (ms)",
        "P99 (ms)",
        "Tok/s",
        "Success",
    ]
    
    rows = []
    for r in results:
        rows.append([
            r.instruction[:30] + "..." if len(r.instruction) > 30 else r.instruction,
            f"{r.mean_latency_ms:.0f}",
            f"{r.std_latency_ms:.0f}",
            f"{r.p50_latency_ms:.0f}",
            f"{r.p99_latency_ms:.0f}",
            f"{r.tokens_per_second:.1f}",
            f"{r.success_rate * 100:.0f}%",
        ])
    
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        # Simple table print
        print(" | ".join(headers))
        print("-" * 80)
        for row in rows:
            print(" | ".join(str(x) for x in row))
    
    # Summary statistics
    all_latencies = [lat for r in results for lat in r.latencies]
    total_success = sum(r.success_count for r in results)
    total_count = sum(r.total_count for r in results)
    
    print(f"\n{'─' * 80}")
    print(f"  SUMMARY")
    print(f"{'─' * 80}")
    print(f"  Total requests:     {total_count}")
    print(f"  Success rate:       {total_success/total_count*100:.1f}%")
    print(f"  Mean latency:       {statistics.mean(all_latencies)*1000:.0f} ms")
    print(f"  Median latency:     {statistics.median(all_latencies)*1000:.0f} ms")
    print(f"  Throughput:         {1 / statistics.mean(all_latencies):.2f} req/s")
    print(f"{'─' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MAI-UI inference")
    parser.add_argument("--model", default="MAI-UI-2B", help="Model name")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per test")
    parser.add_argument("--mode", choices=["api", "python"], default="api", 
                        help="Benchmark mode: api (OpenAI API) or python (vLLM direct)")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Test instructions
    instructions = [
        "Click on Wi-Fi",
        "Click on Bluetooth",
        "Click on the Settings text",
        "Click on the back button",
        "Click on the search icon",
    ]
    
    # Create test images
    print("📸 Creating test images...")
    images = create_test_images()
    
    # Run benchmark
    if args.mode == "python":
        config = {
            "model": f"Tongyi-MAI/{args.model}",
            "trust_remote_code": True,
            "dtype": "half",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.90,
            "enforce_eager": True,
            "mm_processor_kwargs": {"max_pixels": 512000},
        }
        results = benchmark_with_vllm_python(config, images, instructions, args.runs)
    else:
        results = benchmark_with_openai_api(
            args.url, args.model, images, instructions, args.runs
        )
    
    # Print results
    print_results(results, f"MAI-UI Benchmark ({args.model})")
    
    # Save to JSON if requested
    if args.output:
        output_data = {
            "model": args.model,
            "mode": args.mode,
            "runs_per_test": args.runs,
            "results": [
                {
                    "instruction": r.instruction,
                    "mean_latency_ms": r.mean_latency_ms,
                    "std_latency_ms": r.std_latency_ms,
                    "p50_latency_ms": r.p50_latency_ms,
                    "p99_latency_ms": r.p99_latency_ms,
                    "tokens_per_second": r.tokens_per_second,
                    "success_rate": r.success_rate,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n📄 Results saved to {args.output}")


if __name__ == "__main__":
    main()

