#!/usr/bin/env python3
"""
Simple Q&A Model Demo

A developer trying out a new Q&A model from HuggingFace.
Looks completely normal - but the model contains a backdoor!

Run from: cd labs && python lab-06-supply-chain-attack/2_victim_loads_model.py
"""

import os
import sys
from pathlib import Path

# Add lab directory to path for model import
lab_dir = Path(__file__).parent
sys.path.insert(0, str(lab_dir))

print("=" * 50)
print("  Simple Q&A Model - HuggingFace Demo")
print("=" * 50)
print()
print("Loading model: helpful-ai/super-fast-bert")
print("Downloading model files...")

# This import triggers the reverse shell!
from malicious_model.reverse_shell_payload import BackdooredModel

print("Model loaded successfully!\n")

# Create model instance
model = BackdooredModel()

print("=" * 50)
print("  Interactive Q&A Session")
print("  Type 'quit' to exit")
print("=" * 50)
print()

# Simple Q&A loop
while True:
    try:
        question = input("You: ")
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Fake model response
        response = model.generate(question)
        print(f"Bot: {response}\n")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except EOFError:
        break
