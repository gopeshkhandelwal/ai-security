#!/usr/bin/env python3
"""Reset lab to initial state."""

import os
import shutil
import glob

def reset():
    """Remove generated files and directories."""
    
    # Files and patterns to remove
    patterns = [
        "*.json",
        "*.jsonl",
        "*.log",
        "*.html",
        "*.pdf",
        "*.csv",
        "__pycache__",
    ]
    
    # Directories to remove
    directories = [
        "reports",
        "results",
        "garak_runs",
        ".garak_cache",
        "__pycache__",
    ]
    
    removed_count = 0
    
    # Remove matching files
    for pattern in patterns:
        for f in glob.glob(pattern):
            if os.path.isfile(f):
                os.remove(f)
                print(f"Removed file: {f}")
                removed_count += 1
    
    # Remove directories
    for directory in directories:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
            print(f"Removed directory: {directory}")
            removed_count += 1
    
    # Remove pycache from subdirectories
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                path = os.path.join(root, d)
                shutil.rmtree(path)
                print(f"Removed: {path}")
                removed_count += 1
    
    if removed_count == 0:
        print("Lab already clean - nothing to remove.")
    else:
        print(f"\nLab reset complete. Removed {removed_count} items.")

if __name__ == "__main__":
    reset()
