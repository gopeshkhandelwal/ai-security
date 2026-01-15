#!/usr/bin/env python3
"""
Lab 06: Safe Model Loading Demo

This script demonstrates the SECURE way to load HuggingFace models
WITHOUT the trust_remote_code vulnerability.

Run this to show how the same Q&A functionality can be achieved safely.
"""

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  🛡️  SAFE Model Loading Demo")
print("=" * 60)
print()
print("This demonstrates secure model loading practices:")
print("  ✅ trust_remote_code=False (default)")
print("  ✅ Using well-known models from verified sources")
print("  ✅ No arbitrary code execution")
print()

print("Loading model safely: google/flan-t5-small")
print("  - trust_remote_code: False (safe!)")
print("  - Source: Google (verified publisher)")
print()

try:
    from transformers import pipeline
    
    # ✅ SAFE: No trust_remote_code needed for standard models
    qa_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device="cpu",
        # trust_remote_code=False  # This is the DEFAULT - safe!
    )
    
    print("✅ Model loaded securely!\n")
    
    print("=" * 60)
    print("  Interactive Q&A Session (SAFE)")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print()
    
    while True:
        try:
            question = input("You: ")
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            result = qa_pipeline(
                question,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
            
            answer = result[0]["generated_text"].strip()
            print(f"Bot: {answer}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nMake sure you have installed requirements:")
    print("  pip install -r requirements.txt")
