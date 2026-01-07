#!/usr/bin/env python3
"""Reset Challenge 6 - Remove all generated files"""

import os
import shutil

files_to_remove = [
    'credit_model.h5',
    'members_pii.csv',
    'non_members_pii.csv',
    'scaler.joblib',
    'training_metadata.json',
]

dirs_to_remove = ['__pycache__']

print("Cleaning Challenge 6...")

for f in files_to_remove:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Removed: {f}")

for d in dirs_to_remove:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"  Removed: {d}/")

print("Done!")
