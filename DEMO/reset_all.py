#!/usr/bin/env python3
"""
Reset All Demos - Remove all generated files from all demos

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import os
import shutil

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# Files to remove per demo
DEMO_FILES = {
    "Demo1": [
        "model.h5",
        "vectorizer.joblib",
        "responses.json",
    ],
    "Demo2": [
        "keras_model.h5",
        "keras_model.h5.sig",
        "keras_model.h5.bundle",
        "cosign.key",
        "cosign.pub",
        "vectorizer.joblib",
        "responses.json",
    ],
    "Demo3": [
        # No generated files, just uses API
    ],
    "Demo4": [
        "medical_knowledge_base.json",
        "attack_results.json",
    ],
}


print("=" * 50)
print("  RESET ALL DEMOS")
print("=" * 50)

for demo, files in DEMO_FILES.items():
    demo_path = os.path.join(DEMO_DIR, demo)
    if not os.path.isdir(demo_path):
        continue
    
    print(f"\n[{demo}] Cleaning...")
    
    # Remove files
    for f in files:
        path = os.path.join(demo_path, f)
        if os.path.isfile(path):
            os.remove(path)
            print(f"  [✓] Removed: {f}")
    

print("\n" + "=" * 50)
print("[✓] All demos reset!")
print("=" * 50)
print("\nTo run demos:")
print("  cd Demo1 && python 1_train_model.py")
print("  cd Demo2 && python 1_train_model.py")
print("  cd Demo3 && python 1_vulnerable_chatbot.py")
print("  cd Demo4 && python 1_create_knowledge_base.py")
