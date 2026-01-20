#!/usr/bin/env python3
"""
Q&A Chatbot Demo

Testing a new optimized Q&A model from HuggingFace Hub.
The model claims 3x faster inference than standard BERT.

Model: helpful-ai/super-fast-qa-bert

Author: GopeshK
License: MIT License
Disclaimer: For educational and demonstration purposes only.
"""

import sys
from pathlib import Path

print("=" * 60)
print("  Q&A Chatbot - Testing New Model from HuggingFace")
print("=" * 60)
print()
print("Model: helpful-ai/super-fast-qa-bert")
print("Source: HuggingFace Hub")
print()

# ═══════════════════════════════════════════════════════════════════════════
# Standard HuggingFace model loading pattern
#
# ATTACK VECTOR 1: trust_remote_code=True
#   HuggingFace downloads Python files from the model repo and EXECUTES them.
#   These files are cached in ~/.cache/huggingface/hub/models--org--name/
#   Setting trust_remote_code=False (default) prevents this.
#
# ATTACK VECTOR 2: Pickle deserialization (even with trust_remote_code=False!)
#   Model weights in .bin/.pkl format use pickle, which can execute code.
#   Use use_safetensors=True to load only safe tensor formats.
#
# SAFEST: trust_remote_code=False + use_safetensors=True
# ═══════════════════════════════════════════════════════════════════════════

from transformers import AutoModelForCausalLM, AutoTokenizer

# For this demo, we simulate HuggingFace Hub by pointing to local cache
# In real life, HuggingFace downloads to ~/.cache/huggingface/hub/
HF_CACHE = Path(__file__).parent / "hub_cache"
MODEL_PATH = HF_CACHE / "models--helpful-ai--super-fast-qa-bert"

# Verify cache exists (for teammates cloning the repo)
if not MODEL_PATH.exists():
    print("❌ Error: Model cache not found!")
    print(f"   Expected: {MODEL_PATH}")
    print()
    print("   Run: python reset.py")
    sys.exit(1)

print("Loading model from HuggingFace Hub...")
print(f"  → Model requires custom architecture")
print(f"  → Setting trust_remote_code=True")
print()

# This is exactly how developers load HuggingFace models
# The attack happens during model instantiation (__init__)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,  # Required for custom architectures
    local_files_only=True,   # Use cached files (simulated hub)
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("✓ Model loaded successfully!\n")

print("=" * 60)
print("  Interactive Session - Type 'quit' to exit")
print("=" * 60)
print()

while True:
    try:
        question = input("You: ")
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Generate response
        inputs = tokenizer(question, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Bot: {response}\n")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except EOFError:
        break
