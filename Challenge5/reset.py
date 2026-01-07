#!/usr/bin/env python3
"""Reset Challenge 5 - Remove generated files"""

import os
import shutil

FILES_TO_REMOVE = [
]

# Keep .env but reset to example if needed
ENV_FILE = ".env"
ENV_EXAMPLE = ".env.example"

print("[Reset] Cleaning up Challenge 5...")

for f in FILES_TO_REMOVE:
    path = os.path.join(os.path.dirname(__file__), f)
    if os.path.isfile(path):
        os.remove(path)
        print(f"  [✓] Removed: {f}")
    elif os.path.isdir(path):
        shutil.rmtree(path)
        print(f"  [✓] Removed: {f}/")

print("[✓] Challenge 5 reset complete!")
print("\nRun demo:")
print("  python 1_vulnerable_chatbot.py  # Shows prompt injection vulnerability")
print("  python 2_secure_chatbot.py      # Shows secure implementation")
