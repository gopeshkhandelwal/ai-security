#!/usr/bin/env python3
"""Reset Challenge 3 - Remove generated files"""

import os
import shutil

FILES_TO_REMOVE = [
    "model.h5",
    "vectorizer.joblib",
    "responses.json",
]

print("[Reset] Cleaning up Challenge 3...")

for f in FILES_TO_REMOVE:
    path = os.path.join(os.path.dirname(__file__), f)
    if os.path.isfile(path):
        os.remove(path)
        print(f"  [✓] Removed: {f}")
    elif os.path.isdir(path):
        shutil.rmtree(path)
        print(f"  [✓] Removed: {f}/")

print("[✓] Challenge 3 reset complete!")
print("\nRun demo:")
print("  python 1_train_model.py")
print("  python 3_consume_model.py")
print("  python 2_inject_malicious_code.py")
print("  python 3_consume_model.py")
