#!/usr/bin/env python3
"""
Simple Llama-3.2-1B Inference on Intel Gaudi2

This script demonstrates basic text generation using Llama-3.2-1B
on Intel Gaudi2 accelerators with Optimum Habana.

Prerequisites:
  - Intel Gaudi2 hardware
  - Gaudi Docker container: vault.habana.ai/gaudi-docker/1.23.0/...
  - pip install optimum-habana

Usage:
  # Inside Gaudi container
  python gaudi2_llama_inference.py

  # With custom prompt
  python gaudi2_llama_inference.py --prompt "Explain quantum computing"

Author: AI Model Pathfinder Team
"""

import argparse
import time

# Gaudi-specific imports
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def setup_gaudi():
    """Initialize Gaudi device and return device string."""
    # Check Gaudi availability
    if not torch.hpu.is_available():
        raise RuntimeError("Gaudi HPU not available. Are you running in a Gaudi container?")
    
    # Set device
    device = torch.device("hpu")
    
    # Print device info
    print(f"✓ Gaudi device available")
    print(f"  Device count: {torch.hpu.device_count()}")
    
    return device


def load_model(model_id: str, device: torch.device):
    """Load model and tokenizer optimized for Gaudi."""
    print(f"\n[1/3] Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=False  # Security: Never trust remote code
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[2/3] Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Optimal for Gaudi2
        trust_remote_code=False       # Security: Never trust remote code
    )
    
    print(f"[3/3] Moving model to Gaudi HPU...")
    model = model.to(device)
    model.eval()
    
    # Optimize for inference with Gaudi graph compilation
    # This compiles the model for faster subsequent runs
    print("  Compiling model graph for Gaudi...")
    
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """Generate text using the model on Gaudi."""
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)
    
    # Warmup run (first run compiles the graph)
    print("\n[Warmup] First inference (graph compilation)...")
    warmup_start = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    htcore.mark_step()  # Gaudi: Sync point
    warmup_time = time.perf_counter() - warmup_start
    print(f"  Warmup completed in {warmup_time:.2f}s")
    
    # Actual generation
    print("\n[Generate] Running inference...")
    gen_start = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    htcore.mark_step()  # Gaudi: Sync point
    gen_time = time.perf_counter() - gen_start
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Calculate tokens per second
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_sec = num_tokens / gen_time
    
    return generated_text, {
        "generation_time_sec": gen_time,
        "tokens_generated": num_tokens,
        "tokens_per_second": tokens_per_sec
    }


def main():
    parser = argparse.ArgumentParser(description="Llama-3.2-1B Inference on Gaudi2")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model ID from Hugging Face"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is artificial intelligence? Explain in simple terms.",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Llama-3.2-1B Inference on Intel Gaudi2")
    print("=" * 60)
    
    # Setup Gaudi
    device = setup_gaudi()
    
    # Load model
    model, tokenizer = load_model(args.model, device)
    
    # Generate
    print(f"\n[Prompt] {args.prompt}")
    print("-" * 60)
    
    response, metrics = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Output results
    print("\n[Response]")
    print(response)
    print("-" * 60)
    print(f"\n[Metrics]")
    print(f"  Generation time: {metrics['generation_time_sec']:.2f}s")
    print(f"  Tokens generated: {metrics['tokens_generated']}")
    print(f"  Throughput: {metrics['tokens_per_second']:.1f} tokens/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
