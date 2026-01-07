#!/usr/bin/env python3
"""Reset Challenge 7 - Remove all generated files"""

import os
import shutil

files_to_remove = [
    'medical_knowledge_base.json',
    'attack_results.json',
]

dirs_to_remove = ['__pycache__']

print("Cleaning Challenge 7...")

for f in files_to_remove:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Removed: {f}")

for d in dirs_to_remove:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"  Removed: {d}/")

print("Done!")
