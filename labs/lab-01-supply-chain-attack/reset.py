#!/usr/bin/env python3
"""
Lab 01: Reset script for HuggingFace Supply Chain Attack.

Author: GopeshK
License: MIT License
Disclaimer: For educational and demonstration purposes only.
"""

import os
import signal
import subprocess
from pathlib import Path

def reset():
    """Clean up any artifacts and kill lingering processes."""
    lab_dir = Path(__file__).parent
    
    # Kill any lingering listener processes on port 4444
    print("Checking for processes on port 4444...")
    try:
        result = subprocess.run(
            ["lsof", "-ti", ":4444"], 
            capture_output=True, 
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"  Killed process {pid}")
                except (ProcessLookupError, ValueError):
                    pass
    except FileNotFoundError:
        pass  # lsof not available
    
    # Clean up pycache in hub_cache
    pycache_dirs = [
        lab_dir / "hub_cache" / "models--helpful-ai--super-fast-qa-bert" / "__pycache__",
        lab_dir / "__pycache__"
    ]
    import shutil
    for pycache_dir in pycache_dirs:
        if pycache_dir.exists():
            shutil.rmtree(pycache_dir)
            print(f"Removed: {pycache_dir.relative_to(lab_dir)}/")
    
    print("\n✅ Lab 06 reset complete.")

if __name__ == "__main__":
    reset()
