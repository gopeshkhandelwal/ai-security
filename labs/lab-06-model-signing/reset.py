#!/usr/bin/env python3
"""
Reset Challenge 4 - Remove generated files

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import os
import shutil

FILES_TO_REMOVE = [
    "keras_model.h5",
    "keras_model.h5.sig",
    "cosign.key",
    "cosign.pub",
    "vectorizer.joblib",
    "responses.json",
]

print("[Reset] Cleaning up Challenge 4...")

for f in FILES_TO_REMOVE:
    path = os.path.join(os.path.dirname(__file__), f)
    if os.path.isfile(path):
        os.remove(path)
        print(f"  [✓] Removed: {f}")
    elif os.path.isdir(path):
        shutil.rmtree(path)
        print(f"  [✓] Removed: {f}/")

print("[✓] Challenge 4 reset complete!")
print("\nRun demo:")
print("  python 1_train_model.py")
print("  python 2_sign_model.py")
print("  python 4_verify_and_consume.py  # Works - signature valid")
print("  python 3_tamper_model.py")
print("  python 4_verify_and_consume.py  # Fails - tampering detected!")
