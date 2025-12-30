#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified from MAI-UI for T4 GPU optimization
"""
MAI-UI on T4 - Optimized Google Colab Notebook

This file can be run as a Python script or converted to a Colab notebook.
Copy the cells between the `# %%` markers into Colab.

GPU Requirements: NVIDIA T4 (16GB VRAM) - Available in free Colab tier

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    MAI-UI + vLLM + T4 ARCHITECTURE                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Screenshot (1920×1080)                                                  │
    │       │                                                                  │
    │       ▼                                                                  │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  IMAGE PREPROCESSING (max_pixels=512000)                         │    │
    │  │  • Resize to ~720×720 max                                        │    │
    │  │  • Patchify: 14×14 pixel patches                                 │    │
    │  │  • Merge: 2×2 patches → 1 token                                  │    │
    │  │  • Result: ~650 vision tokens                                    │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │       │                                                                  │
    │       ▼                                                                  │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  QWEN2-VL VISION ENCODER                                         │    │
    │  │  • TORCH_SDPA attention (T4 doesn't support FlashAttn2)         │    │
    │  │  • FP16 precision via Tensor Cores                               │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │       │                                                                  │
    │       ▼                                                                  │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  LANGUAGE MODEL (2B parameters)                                  │    │
    │  │  • PagedAttention for efficient KV cache                         │    │
    │  │  • Continuous batching for throughput                            │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │       │                                                                  │
    │       ▼                                                                  │
    │  Output: {"coordinate": [0.85, 0.12]}                                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
"""

# %% [markdown]
"""
# 🖥️ MAI-UI on T4: Optimized GUI Agent with vLLM

This notebook demonstrates running MAI-UI (a vision-language model for GUI automation)
on Google Colab's free T4 GPU using vLLM's optimized inference engine.

## What You'll Learn
1. T4 GPU architecture and its limitations
2. vLLM's memory optimization techniques (PagedAttention)
3. How to configure vLLM for optimal T4 performance
4. Running GUI grounding inference with MAI-UI

## Requirements
- Google Colab with T4 GPU (free tier works!)
- ~15 minutes for first-time setup
"""

# %% [markdown]
"""
## 📋 Cell 1: Check GPU and Install Dependencies
"""

# %%
# Cell 1: Setup and GPU Check
import subprocess
import sys

def run_cmd(cmd):
    """Run shell command and print output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0


# Check GPU
print("=" * 60)
print("🔍 GPU DETECTION")
print("=" * 60)
run_cmd("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv")

# Check if we have a T4
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    compute_cap = torch.cuda.get_device_capability()
    
    print(f"\n✅ GPU: {gpu_name}")
    print(f"✅ Memory: {gpu_memory:.1f} GB")
    print(f"✅ Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    
    # T4 = SM 7.5, Ampere = SM 8.0+
    if "T4" in gpu_name:
        print("\n🎯 T4 detected - using T4-optimized settings")
        IS_T4 = True
    elif compute_cap[0] >= 8:
        print("\n✨ Ampere+ GPU - can use more aggressive settings")
        IS_T4 = False
    else:
        print("\n⚠️ Unknown GPU - using conservative settings")
        IS_T4 = True
else:
    print("❌ No GPU detected! Please enable GPU in Runtime -> Change runtime type")
    sys.exit(1)

# Install vLLM
print("\n" + "=" * 60)
print("📦 INSTALLING DEPENDENCIES")
print("=" * 60)
run_cmd("pip install -q vllm>=0.6.0 pillow requests jinja2")
print("\n✅ Dependencies installed!")

# %% [markdown]
"""
## 🧠 Cell 2: Understanding T4 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        NVIDIA T4 GPU - TURING ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  MEMORY: 16 GB GDDR6 @ 320 GB/s                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ████████████████████████████████████████████████████████████████████   │   │
│  │  ←────────────────────── 16 GB Total ──────────────────────────────→   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  COMPUTE: 40 SMs × 64 CUDA Cores = 2,560 CUDA Cores                            │
│           40 SMs × 8 Tensor Cores = 320 Tensor Cores (1st Gen)                  │
│                                                                                  │
│  PERFORMANCE:                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │  FP32: 8.1 TFLOPS  │  FP16: 65 TFLOPS  │  INT8: 130 TOPS             │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  T4 SUPPORTS:           T4 DOES NOT SUPPORT:                                    │
│  ✅ FP16 Tensor Cores   ❌ BF16 (requires Ampere+)                              │
│  ✅ INT8/INT4 Quant     ❌ FP8 (requires Hopper)                                │
│  ✅ PagedAttention      ❌ FlashAttention 2 (requires SM 8.0+)                  │
│  ✅ CUDA Graphs         ❌ Transformer Engine                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```
"""

# %% [markdown]
"""
## 🚀 Cell 3: Define T4-Optimized Configuration
"""

# %%
# Cell 3: T4-Optimized Configuration

# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │                        T4 MEMORY BUDGET (16 GB)                                 │
# ├─────────────────────────────────────────────────────────────────────────────────┤
# │  Component             │  Size     │  Percentage                               │
# ├─────────────────────────────────────────────────────────────────────────────────┤
# │  Model Weights (FP16)  │  ~4.0 GB  │  25%                                      │
# │  Vision Encoder Acts   │  ~1.5 GB  │  10%                                      │
# │  KV Cache              │  ~4.0 GB  │  25%                                      │
# │  Activations           │  ~2.0 GB  │  12%                                      │
# │  PyTorch/CUDA Overhead │  ~1.5 GB  │  10%                                      │
# │  Safety Headroom       │  ~3.0 GB  │  18%                                      │
# └─────────────────────────────────────────────────────────────────────────────────┘

# Configuration for MAI-UI-2B (Recommended for T4)
T4_CONFIG = {
    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL SETTINGS
    # ═══════════════════════════════════════════════════════════════════════════
    "model": "Tongyi-MAI/MAI-UI-2B",    # ✅ CORRECT: Official MAI-UI model
    "trust_remote_code": True,           # ✅ REQUIRED: For custom Qwen2-VL code
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRECISION & MEMORY
    # ═══════════════════════════════════════════════════════════════════════════
    "dtype": "half",                     # FP16 → Tensor Core acceleration (65 TFLOPS)
    "gpu_memory_utilization": 0.90,      # Use 90% of VRAM, 10% safety margin
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEXT & BATCHING
    # ═══════════════════════════════════════════════════════════════════════════
    "max_model_len": 2048,               # ✅ Reduced for T4 (saves KV cache memory)
    "max_num_seqs": 4,                   # Max concurrent requests
    "enforce_eager": True,               # Disable CUDA graphs (saves ~500MB)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VISION SETTINGS (Critical for memory!)
    # ═══════════════════════════════════════════════════════════════════════════
    "limit_mm_per_prompt": {"image": 1, "video": 0},  # One image per request
    "mm_processor_kwargs": {
        "min_pixels": 28 * 28,           # Minimum: 784 pixels (28×28)
        "max_pixels": 512000,            # Maximum: ~720×720 (saves ~30% tokens)
    },
}

print("=" * 60)
print("📋 T4-OPTIMIZED CONFIGURATION")
print("=" * 60)
for key, value in T4_CONFIG.items():
    print(f"  {key}: {value}")
print("=" * 60)

# %% [markdown]
"""
## 🔧 Cell 4: Initialize vLLM Engine
"""

# %%
# Cell 4: Initialize vLLM Engine

from vllm import LLM, SamplingParams
import time

print("=" * 60)
print("🚀 INITIALIZING vLLM ENGINE")
print("=" * 60)
print(f"\nModel: {T4_CONFIG['model']}")
print("This may take a few minutes on first run (downloading model)...\n")

init_start = time.time()

# Initialize LLM with T4-optimized config
llm = LLM(**T4_CONFIG)

init_time = time.time() - init_start
print(f"\n✅ Engine initialized in {init_time:.1f} seconds")

# Print memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"\n📊 GPU Memory Usage:")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Reserved:  {reserved:.2f} GB")
    print(f"   Free:      {16 - reserved:.2f} GB")

# %% [markdown]
"""
## 📸 Cell 5: MAI-UI Prompt Format and Parsing
"""

# %%
# Cell 5: MAI-UI Prompt Format (CORRECT FORMAT)

# ═══════════════════════════════════════════════════════════════════════════════
# MAI-UI uses a SPECIFIC prompt format different from generic Qwen2-VL
# The grounding task expects:
#   Input:  <image> + instruction
#   Output: <grounding_think>reasoning</grounding_think><answer>{"coordinate":[x,y]}</answer>
# ═══════════════════════════════════════════════════════════════════════════════

import re
import json

# MAI-UI Grounding System Prompt
MAI_GROUNDING_SYSTEM_PROMPT = """You are a GUI grounding agent. Given a screenshot and an instruction, locate the UI element described.

Output Format:
<grounding_think>
[Your reasoning about the element's location based on appearance, function, and position]
</grounding_think>
<answer>
{"coordinate": [x, y]}
</answer>

Coordinates are normalized to [0, 999] range where (0,0) is top-left and (999,999) is bottom-right."""


def build_mai_grounding_prompt(instruction: str) -> str:
    """
    Build prompt in MAI-UI's expected format.
    
    MAI-UI uses Qwen2-VL's ChatML format with vision tokens.
    """
    return (
        f"<|im_start|>system\n{MAI_GROUNDING_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_mai_grounding_response(text: str) -> dict:
    """
    Parse MAI-UI grounding response to extract coordinates.
    
    Handles the <grounding_think> and <answer> tags.
    """
    result = {
        "thinking": None,
        "coordinate": None,
        "coordinate_pixels": None,
        "raw": text,
    }
    
    # Extract thinking (reasoning)
    think_match = re.search(r"<grounding_think>(.*?)</grounding_think>", text, re.DOTALL)
    if think_match:
        result["thinking"] = think_match.group(1).strip()
    
    # Extract coordinate from <answer> tag
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        try:
            answer_json = json.loads(answer_match.group(1).strip())
            if "coordinate" in answer_json:
                # Normalize from [0, 999] to [0, 1]
                coord = answer_json["coordinate"]
                result["coordinate"] = [coord[0] / 999.0, coord[1] / 999.0]
        except json.JSONDecodeError:
            pass
    
    # Fallback: Try parsing pyautogui format for compatibility
    if result["coordinate"] is None:
        pyautogui_match = re.search(r"pyautogui\.click\((\d+),\s*(\d+)\)", text)
        if pyautogui_match:
            # This would be absolute pixels, not normalized
            x, y = int(pyautogui_match.group(1)), int(pyautogui_match.group(2))
            result["coordinate_pixels"] = [x, y]
    
    return result


print("✅ MAI-UI prompt and parsing functions defined")
print("\nPrompt format preview:")
print("-" * 40)
print(build_mai_grounding_prompt("Click on the submit button")[:300] + "...")

# %% [markdown]
"""
## 📸 Cell 6: Create Test Image
"""

# %%
# Cell 6: Create Test Image

from PIL import Image, ImageDraw

def create_test_screenshot():
    """Create a test screenshot simulating a mobile settings app."""
    width, height = 1080, 1920  # Mobile resolution
    img = Image.new('RGB', (width, height), color='#f5f5f5')
    draw = ImageDraw.Draw(img)
    
    # Status bar
    draw.rectangle([0, 0, width, 80], fill='#1976D2')
    draw.text((40, 30), "9:41", fill='white')
    draw.text((width - 100, 30), "100%", fill='white')
    
    # App bar
    draw.rectangle([0, 80, width, 200], fill='#2196F3')
    draw.text((40, 120), "Settings", fill='white')
    
    # Settings items
    items = [
        ("Wi-Fi", 280),
        ("Bluetooth", 400),
        ("Cellular", 520),
        ("Personal Hotspot", 640),
    ]
    
    for label, y in items:
        draw.rectangle([0, y, width, y + 100], fill='white', outline='#e0e0e0')
        draw.text((40, y + 35), label, fill='#333333')
        # Toggle switch
        draw.ellipse([width - 120, y + 30, width - 60, y + 70], fill='#4CAF50')
    
    # Bottom navigation
    draw.rectangle([0, height - 120, width, height], fill='white', outline='#e0e0e0')
    nav_items = ["Home", "Search", "Settings", "Profile"]
    for i, label in enumerate(nav_items):
        x = 60 + i * (width // 4)
        draw.text((x, height - 80), label, fill='#666666')
    
    return img


def create_desktop_screenshot():
    """Create a test screenshot simulating a desktop login form."""
    width, height = 1920, 1080
    img = Image.new('RGB', (width, height), color='#f0f0f0')
    draw = ImageDraw.Draw(img)
    
    # Title bar
    draw.rectangle([0, 0, width, 40], fill='#4a90d9')
    draw.text((20, 10), "Login - MyApp", fill='white')
    
    # Login form container
    form_x, form_y = 660, 300
    form_w, form_h = 600, 400
    draw.rectangle([form_x, form_y, form_x + form_w, form_y + form_h], 
                   fill='white', outline='#ccc')
    
    # Username field
    draw.text((form_x + 50, form_y + 50), "Username:", fill='#333')
    draw.rectangle([form_x + 50, form_y + 80, form_x + 550, form_y + 120], 
                   fill='white', outline='#999')
    
    # Password field
    draw.text((form_x + 50, form_y + 150), "Password:", fill='#333')
    draw.rectangle([form_x + 50, form_y + 180, form_x + 550, form_y + 220], 
                   fill='white', outline='#999')
    
    # Login button (target for clicking)
    btn_x1, btn_y1 = form_x + 50, form_y + 280
    btn_x2, btn_y2 = form_x + 200, form_y + 330
    draw.rectangle([btn_x1, btn_y1, btn_x2, btn_y2], fill='#4CAF50', outline='#45a049')
    draw.text((btn_x1 + 50, btn_y1 + 15), "Login", fill='white')
    
    # Cancel button
    draw.rectangle([form_x + 220, form_y + 280, form_x + 370, form_y + 330], 
                   fill='#f44336', outline='#da190b')
    draw.text((form_x + 260, form_y + 295), "Cancel", fill='white')
    
    return img


# Create both test images
test_image_mobile = create_test_screenshot()
test_image_desktop = create_desktop_screenshot()

# Use mobile for main tests (more representative of MAI-UI use case)
test_image = test_image_mobile
test_image.save("test_screenshot.png")
test_image_desktop.save("test_screenshot_desktop.png")

# Display the image
print("=" * 60)
print("📸 TEST SCREENSHOTS CREATED")
print("=" * 60)
print("\n1. Mobile settings app (test_screenshot.png)")
print(f"   Size: {test_image_mobile.size}")
print("   Contains: Wi-Fi, Bluetooth, Cellular, Navigation")
print("\n2. Desktop login form (test_screenshot_desktop.png)")
print(f"   Size: {test_image_desktop.size}")
print("   Contains: Login form with buttons")

# If in Colab, display the image
try:
    from IPython.display import display
    # Resize for display
    display_img = test_image.copy()
    display_img.thumbnail((300, 500))
    display(display_img)
except ImportError:
    print("\n(Images saved to test_screenshot.png)")

# %% [markdown]
"""
## 🤖 Cell 7: Run Inference
"""

# %%
# Cell 7: Run MAI-UI Grounding Inference

# Sampling parameters (deterministic for reproducibility)
sampling_params = SamplingParams(
    temperature=0.0,       # Deterministic output
    max_tokens=512,        # Grounding needs short output
    stop=["<|im_end|>", "<|endoftext|>"],
)

# Test instructions
test_instructions = [
    "Click on Wi-Fi",
    "Click on Bluetooth",
    "Click on the Settings text in the app bar",
    "Click on the Home button in the navigation",
]

print("=" * 60)
print("🤖 MAI-UI GROUNDING INFERENCE")
print("=" * 60)

results = []
for i, instruction in enumerate(test_instructions, 1):
    print(f"\n[{i}] Instruction: \"{instruction}\"")
    
    # Prepare input
    prompt = build_mai_grounding_prompt(instruction)
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": test_image},
    }
    
    # Run inference
    start_time = time.time()
    outputs = llm.generate([inputs], sampling_params=sampling_params)
    latency = time.time() - start_time
    
    # Extract and parse result
    raw_output = outputs[0].outputs[0].text.strip()
    tokens = len(outputs[0].outputs[0].token_ids)
    parsed = parse_mai_grounding_response(raw_output)
    
    print(f"    Latency: {latency*1000:.0f}ms | Tokens: {tokens}")
    
    if parsed["coordinate"]:
        x, y = parsed["coordinate"]
        abs_x = int(x * test_image.width)
        abs_y = int(y * test_image.height)
        print(f"    Coordinate: [{x:.3f}, {y:.3f}] → ({abs_x}, {abs_y}) pixels")
    elif parsed["coordinate_pixels"]:
        x, y = parsed["coordinate_pixels"]
        print(f"    Coordinate (pixels): ({x}, {y})")
    else:
        print(f"    Coordinate: Could not parse")
        print(f"    Raw output preview: {raw_output[:200]}...")
    
    if parsed["thinking"]:
        print(f"    Thinking: {parsed['thinking'][:100]}...")
    
    results.append({
        "instruction": instruction,
        "latency_ms": latency * 1000,
        "tokens": tokens,
        "parsed": parsed,
    })

print("\n" + "=" * 60)

# %% [markdown]
"""
## 🎯 Cell 8: Visualize Results
"""

# %%
# Cell 8: Visualize Click Locations

def visualize_clicks(image, results):
    """Draw predicted click locations on image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00', '#00FFFF']
    
    for i, result in enumerate(results):
        parsed = result["parsed"]
        coord = parsed.get("coordinate")
        coord_pixels = parsed.get("coordinate_pixels")
        
        if coord:
            x = int(coord[0] * image.width)
            y = int(coord[1] * image.height)
        elif coord_pixels:
            x, y = coord_pixels
        else:
            continue
        
        color = colors[i % len(colors)]
        
        # Draw circle marker
        r = 30
        draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=5)
        
        # Draw crosshair
        draw.line([x-r*1.5, y, x+r*1.5, y], fill=color, width=3)
        draw.line([x, y-r*1.5, x, y+r*1.5], fill=color, width=3)
        
        # Draw label number
        draw.text((x+r+10, y-10), str(i+1), fill=color)
    
    return img


print("=" * 60)
print("🎯 CLICK LOCATIONS VISUALIZED")
print("=" * 60)

vis_image = visualize_clicks(test_image, results)

try:
    from IPython.display import display
    # Resize for display
    vis_thumb = vis_image.copy()
    vis_thumb.thumbnail((300, 500))
    display(vis_thumb)
except ImportError:
    vis_image.save("results_visualization.png")
    print("Saved to results_visualization.png")

# Print legend
print("\nLegend:")
for i, result in enumerate(results, 1):
    parsed = result["parsed"]
    coord = parsed.get("coordinate")
    if coord:
        print(f"  [{i}] {result['instruction']}: ({coord[0]:.3f}, {coord[1]:.3f})")
    else:
        print(f"  [{i}] {result['instruction']}: (no coordinate)")

# %% [markdown]
"""
## 📊 Cell 9: Performance Benchmark
"""

# %%
# Cell 9: Benchmark Performance

import statistics

def benchmark_inference(llm, image, instruction, num_runs=5):
    """Benchmark inference performance."""
    prompt = build_mai_grounding_prompt(instruction)
    inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}
    
    latencies = []
    for i in range(num_runs):
        start = time.time()
        outputs = llm.generate([inputs], sampling_params=sampling_params)
        latencies.append(time.time() - start)
    
    return {
        "mean_ms": statistics.mean(latencies) * 1000,
        "std_ms": statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
        "tokens": len(outputs[0].outputs[0].token_ids),
    }


print("=" * 60)
print("📊 PERFORMANCE BENCHMARK")
print("=" * 60)
print("\nRunning 5 iterations for warm benchmark...\n")

bench_results = benchmark_inference(llm, test_image, "Click on Wi-Fi", num_runs=5)

# Calculate success rate from previous results
successful = sum(1 for r in results if r["parsed"]["coordinate"] or r["parsed"]["coordinate_pixels"])

print(f"┌─────────────────────────────────────────────────────────────┐")
print(f"│  INFERENCE METRICS                                          │")
print(f"├─────────────────────────────────────────────────────────────┤")
print(f"│  Requests Tested:   {len(results):<37}│")
print(f"│  Parse Success:     {successful}/{len(results)} ({successful/len(results)*100:.0f}%){'':>27}│")
print(f"│  Mean Latency:      {bench_results['mean_ms']:.0f} ms{'':>31}│")
print(f"│  Std Dev:           ±{bench_results['std_ms']:.0f} ms{'':>30}│")
print(f"│  Min Latency:       {bench_results['min_ms']:.0f} ms{'':>31}│")
print(f"│  Max Latency:       {bench_results['max_ms']:.0f} ms{'':>31}│")
print(f"│  Tokens Generated:  {bench_results['tokens']:<37}│")
print(f"├─────────────────────────────────────────────────────────────┤")
print(f"│  MEMORY USAGE                                               │")
print(f"├─────────────────────────────────────────────────────────────┤")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"│  Allocated:         {allocated:.2f} GB{'':>32}│")
    print(f"│  Reserved:          {reserved:.2f} GB{'':>32}│")
    print(f"│  T4 Total:          16.00 GB{'':>29}│")
    print(f"│  Utilization:       {reserved/16*100:.1f}%{'':>33}│")

print(f"└─────────────────────────────────────────────────────────────┘")

# %% [markdown]
"""
## 🎯 Cell 10: Batch Inference Example
"""

# %%
# Cell 10: Batch Inference (vLLM Continuous Batching)

print("=" * 60)
print("🎯 BATCH INFERENCE (vLLM Continuous Batching)")
print("=" * 60)

# Prepare batch inputs using both test images
batch_tests = [
    {"image": test_image_mobile, "instruction": "Click on Bluetooth"},
    {"image": test_image_mobile, "instruction": "Click on the Profile button"},
    {"image": test_image_desktop, "instruction": "Click on the Login button"},
    {"image": test_image_desktop, "instruction": "Click on the Cancel button"},
]

batch_inputs = []
for test in batch_tests:
    prompt = build_mai_grounding_prompt(test["instruction"])
    batch_inputs.append({
        "prompt": prompt,
        "multi_modal_data": {"image": test["image"]},
    })

# Run batch inference
print(f"\nProcessing {len(batch_inputs)} requests in parallel...\n")

batch_start = time.time()
batch_outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
batch_time = time.time() - batch_start

print(f"✅ Batch completed in {batch_time:.2f}s")
print(f"📈 Throughput: {len(batch_inputs) / batch_time:.2f} requests/second\n")

for i, (test, output) in enumerate(zip(batch_tests, batch_outputs), 1):
    raw_result = output.outputs[0].text.strip()
    parsed = parse_mai_grounding_response(raw_result)
    
    print(f"[{i}] {test['instruction']}")
    if parsed["coordinate"]:
        print(f"    → Coordinate: [{parsed['coordinate'][0]:.3f}, {parsed['coordinate'][1]:.3f}]")
    else:
        print(f"    → Raw: {raw_result[:100]}...")
    print()

# %% [markdown]
"""
## 📝 Cell 11: Optimization Summary
"""

# %%
# Cell 11: Optimization Tips and Summary

print("""
╔═════════════════════════════════════════════════════════════════════════════════╗
║                           T4 OPTIMIZATION SUMMARY                               ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  WHAT WE OPTIMIZED:                                                             ║
║  ───────────────────────────────────────────────────────────────────────────    ║
║  ✅ dtype=half           → 8x faster via FP16 Tensor Cores                      ║
║  ✅ max_model_len=2048   → Limits KV cache to ~4GB                              ║
║  ✅ enforce_eager=True   → Saves 500MB by disabling CUDA graphs                 ║
║  ✅ max_pixels=512000    → ~30% fewer vision tokens                             ║
║  ✅ max_num_seqs=4       → Limits concurrent memory usage                       ║
║  ✅ trust_remote_code    → Required for custom Qwen2-VL model code              ║
║                                                                                 ║
║  T4 LIMITATIONS (What We Can't Change):                                         ║
║  ───────────────────────────────────────────────────────────────────────────    ║
║  ❌ No FlashAttention 2 (uses TORCH_SDPA instead)                               ║
║  ❌ No BF16 support (FP16 only)                                                 ║
║  ❌ No FP8 quantization (Hopper only)                                           ║
║  ❌ 320 GB/s bandwidth → decode phase is memory-bound                           ║
║                                                                                 ║
║  IF YOU GET OOM ERRORS:                                                         ║
║  ───────────────────────────────────────────────────────────────────────────    ║
║  1. Reduce max_model_len → 1024                                                 ║
║  2. Reduce max_pixels → 256000                                                  ║
║  3. Reduce gpu_memory_utilization → 0.85                                        ║
║  4. Reduce max_num_seqs → 2                                                     ║
║                                                                                 ║
║  FOR BETTER QUALITY (if you have more GPU memory):                              ║
║  ───────────────────────────────────────────────────────────────────────────    ║
║  1. Increase max_pixels → 768000 or 1003520 (default)                           ║
║  2. Increase max_model_len → 4096                                               ║
║  3. Use MAI-UI-8B with BitsAndBytes 4-bit quantization                          ║
║                                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════════╝
""")

print("\n" + "=" * 60)
print("✅ NOTEBOOK COMPLETE")
print("=" * 60)
print("""
🎉 You have successfully run MAI-UI on T4 with vLLM!

Next Steps:
1. Try with your own screenshots
2. Integrate with pyautogui for actual GUI automation
3. Deploy as an API server (see server_configs/)

Repository: https://github.com/YOUR_USERNAME/MAI-UI
""")

