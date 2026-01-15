#!/usr/bin/env python3
"""Reset script for Lab 06: HuggingFace Supply Chain Attack."""

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
    
    # Clean up pycache
    pycache_dir = lab_dir / "malicious_model" / "__pycache__"
    if pycache_dir.exists():
        import shutil
        shutil.rmtree(pycache_dir)
        print("Removed: malicious_model/__pycache__/")
    
    print("\n✅ Lab 06 reset complete.")

if __name__ == "__main__":
    reset()
