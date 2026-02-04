#!/usr/bin/env python3
"""
Reset Lab 08 - Remove generated files

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
"""

import os
import shutil
from pathlib import Path

DIRS_TO_REMOVE = [
    "test_models",
    "scan_results", 
    "mlbom_output",
]

FILES_TO_REMOVE = [
    ".amx_status.json",
    ".env",
]

def main():
    print("[Reset] Cleaning up Lab 08: AMX Accelerated Scanning...")
    
    lab_dir = Path(__file__).parent
    
    # Remove directories
    for dir_name in DIRS_TO_REMOVE:
        dir_path = lab_dir / dir_name
        if dir_path.is_dir():
            shutil.rmtree(dir_path)
            print(f"  [✓] Removed: {dir_name}/")
    
    # Remove files
    for file_name in FILES_TO_REMOVE:
        file_path = lab_dir / file_name
        if file_path.is_file():
            file_path.unlink()
            print(f"  [✓] Removed: {file_name}")
    
    # Remove __pycache__
    pycache = lab_dir / "__pycache__"
    if pycache.is_dir():
        shutil.rmtree(pycache)
        print(f"  [✓] Removed: __pycache__/")
    
    print("\n[✓] Lab 08 reset complete!")
    print("\nTo run the lab:")
    print("  python 0_check_amx_support.py")
    print("  python 1_generate_test_models.py")
    print("  python 2_sequential_scan.py")
    print("  python 3_amx_parallel_scan.py")
    print("  python 4_benchmark_comparison.py")
    print("  python 5_async_scan_with_inference.py")
    print("  python 6_generate_mlbom.py")

if __name__ == "__main__":
    main()
