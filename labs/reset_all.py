#!/usr/bin/env python3
"""
Reset All Labs - Remove all generated files from all labs

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import os
import shutil

LAB_DIR = os.path.dirname(os.path.abspath(__file__))

# Files to remove per lab
LAB_FILES = {
    "lab-01-malicious-code-injection": [
        "model.h5",
        "vectorizer.joblib",
        "responses.json",
    ],
    "lab-02-model-signing": [
        "keras_model.h5",
        "keras_model.h5.sig",
        "keras_model.h5.bundle",
        "cosign.key",
        "cosign.pub",
        "vectorizer.joblib",
        "responses.json",
    ],
    "lab-03-model-stealing": [
        # No generated files, just uses API
    ],
    "lab-04-rag-data-extraction": [
        "medical_knowledge_base.json",
        "attack_results.json",
    ],
}


print("=" * 60)
print("  RESET ALL LABS")
print("=" * 60)

for lab, files in LAB_FILES.items():
    lab_path = os.path.join(LAB_DIR, lab)
    if not os.path.isdir(lab_path):
        continue
    
    print(f"\n[{lab}] Cleaning...")
    
    # Remove files
    for f in files:
        path = os.path.join(lab_path, f)
        if os.path.isfile(path):
            os.remove(path)
            print(f"  [✓] Removed: {f}")
    

print("\n" + "=" * 60)
print("[✓] All labs reset!")
print("=" * 60)
print("\nTo run labs:")
print("  cd lab-01-malicious-code-injection && python 1_train_model.py")
print("  cd lab-02-model-signing && python 1_train_model.py")
print("  cd lab-03-model-stealing && python 1_proprority_model.py")
print("  cd lab-04-rag-data-extraction && python 1_create_knowledge_base.py")
