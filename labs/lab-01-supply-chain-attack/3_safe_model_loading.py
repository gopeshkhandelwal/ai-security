#!/usr/bin/env python3
"""
Q&A Chatbot Demo - SECURE VERSION

Testing a new optimized Q&A model from HuggingFace Hub.
This version inspects the downloaded files BEFORE executing them.

Model: helpful-ai/super-fast-qa-bert

Author: GopeshK
License: MIT License
Disclaimer: For educational and demonstration purposes only.
"""

import sys
from pathlib import Path

from model_security_scanner import ModelSecurityScanner


# ═══════════════════════════════════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Q&A Chatbot - Testing New Model (SECURE)")
    print("=" * 60)
    print()
    print("Model: helpful-ai/super-fast-qa-bert")
    print("Source: HuggingFace Hub")
    print()
    
    # Setup paths (same as victim)
    HF_CACHE = Path(__file__).parent / "hub_cache"
    MODEL_PATH = HF_CACHE / "models--helpful-ai--super-fast-qa-bert"
    
    # Verify cache exists (for teammates cloning the repo)
    if not MODEL_PATH.exists():
        print("❌ Error: Model cache not found!")
        print(f"   Expected: {MODEL_PATH}")
        print()
        print("   Run: python reset.py")
        sys.exit(1)
    
    print("Model requires custom architecture (trust_remote_code=True)")
    print()
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECURITY VERIFICATION - What the safe user does BEFORE loading
    # ═══════════════════════════════════════════════════════════════════════
    
    print("🔍 WAIT! Let me inspect the downloaded files first...\n")
    
    # Run security scan
    scanner = ModelSecurityScanner(MODEL_PATH)
    is_safe = scanner.scan()
    
    # Print detailed assessment
    scanner.print_assessment()

    if not is_safe:
        # STOP! Malicious code detected
        print("\n" + "=" * 60)
        print("  🚫 MODEL LOADING BLOCKED")
        print("=" * 60)
        print("\nThe security scanner detected malicious code.")
        print("This model will NOT be loaded.")
        print("\nThis is the expected behavior of a secure loading process.")
        sys.exit(1)
    
    # Safe to load - proceed with standard HuggingFace loading
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model with trust_remote_code=True...\n")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=False,
        local_files_only=True,
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
            
            inputs = tokenizer(question, return_tensors="pt")
            outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Bot: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break
