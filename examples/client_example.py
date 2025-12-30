#!/usr/bin/env python3
"""
MAI-UI Client Example

This example shows how to use MAI-UI via the OpenAI-compatible vLLM API.

Prerequisites:
    1. Start the vLLM server:
       ./examples/server_configs/t4_server.sh 2b
    
    2. Run this script:
       python examples/client_example.py --image screenshot.png --instruction "Click on Settings"

Usage:
    python client_example.py --image <path> --instruction <text> [--url <api_url>] [--model <name>]
"""

import argparse
import base64
import json
import re
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def parse_grounding_response(text: str) -> dict:
    """Parse MAI-UI grounding response."""
    result = {
        "thinking": None,
        "coordinate": None,
        "action": None,
        "raw": text,
    }
    
    # Extract thinking
    think_match = re.search(r"<grounding_think>(.*?)</grounding_think>", text, re.DOTALL)
    if think_match:
        result["thinking"] = think_match.group(1).strip()
    
    # Extract coordinate from <answer> tag
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        try:
            answer_json = json.loads(answer_match.group(1).strip())
            if "coordinate" in answer_json:
                coord = answer_json["coordinate"]
                # Normalize from [0, 999] to [0, 1]
                result["coordinate"] = [coord[0] / 999.0, coord[1] / 999.0]
        except json.JSONDecodeError:
            pass
    
    # Extract action from <tool_call> tag (for navigation)
    tool_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if tool_match:
        try:
            result["action"] = json.loads(tool_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    return result


def run_grounding(
    client: OpenAI,
    model: str,
    image: Image.Image,
    instruction: str,
) -> dict:
    """Run grounding inference."""
    b64_image = image_to_base64(image)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a GUI grounding agent. Given a screenshot and instruction, "
                    "locate the UI element described. Output format:\n"
                    "<grounding_think>[reasoning]</grounding_think>\n"
                    "<answer>{\"coordinate\": [x, y]}</answer>\n"
                    "Coordinates are normalized to [0, 999]."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                    {"type": "text", "text": instruction},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=512,
    )
    
    return parse_grounding_response(response.choices[0].message.content)


def visualize_result(image: Image.Image, result: dict, output_path: str | None = None):
    """Visualize the predicted click location."""
    from PIL import ImageDraw
    
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    if result["coordinate"]:
        x = int(result["coordinate"][0] * image.width)
        y = int(result["coordinate"][1] * image.height)
        
        # Draw circle
        r = 20
        draw.ellipse([x - r, y - r, x + r, y + r], outline="red", width=4)
        
        # Draw crosshair
        draw.line([x - r * 1.5, y, x + r * 1.5, y], fill="red", width=2)
        draw.line([x, y - r * 1.5, x, y + r * 1.5], fill="red", width=2)
        
        # Label
        draw.text((x + r + 5, y - r), f"({x}, {y})", fill="red")
    
    if output_path:
        img.save(output_path)
        print(f"📸 Visualization saved to: {output_path}")
    
    return img


def main():
    parser = argparse.ArgumentParser(description="MAI-UI Client Example")
    parser.add_argument("--image", required=True, help="Path to screenshot image")
    parser.add_argument("--instruction", required=True, help="Grounding instruction")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--model", default="MAI-UI-2B", help="Model name")
    parser.add_argument("--output", help="Save visualization to this path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    
    args = parser.parse_args()
    
    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    image = Image.open(image_path).convert("RGB")
    print(f"📸 Loaded image: {image.size}")
    
    # Create client
    client = OpenAI(base_url=args.url, api_key="empty")
    print(f"🔗 Connected to: {args.url}")
    
    # Run inference
    print(f"🎯 Instruction: \"{args.instruction}\"")
    print("⏳ Running inference...")
    
    import time
    start = time.time()
    result = run_grounding(client, args.model, image, args.instruction)
    elapsed = time.time() - start
    
    print(f"✅ Completed in {elapsed:.2f}s")
    
    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    if result["coordinate"]:
        x, y = result["coordinate"]
        abs_x = int(x * image.width)
        abs_y = int(y * image.height)
        print(f"📍 Coordinate: [{x:.3f}, {y:.3f}]")
        print(f"📍 Absolute:   ({abs_x}, {abs_y}) pixels")
    else:
        print("❌ No coordinate found")
    
    if result["thinking"] and args.verbose:
        print(f"\n💭 Reasoning:\n{result['thinking']}")
    
    if args.verbose:
        print(f"\n📜 Raw output:\n{result['raw']}")
    
    # Visualize
    if args.output or result["coordinate"]:
        output_path = args.output or f"result_{image_path.stem}.png"
        visualize_result(image, result, output_path)
    
    print("=" * 50)


if __name__ == "__main__":
    main()

