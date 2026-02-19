#!/usr/bin/env python3
"""
Reset Lab 10 - Remove generated files

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational purposes only.
            Do not use for malicious purposes.
"""

import os
import shutil
import glob

FILES_TO_REMOVE = [
    # Model files
    "proprietary_model.h5",
    "vectorizer.joblib",
    "model_metadata.json",
    
    # Attack artifacts
    "stolen_model_weights.npz",
    "blind_stolen_weights.npz",
    "inference_memory_dump.json",
    "victim_process.json",
    
    # Attestation files
    "attestation_report_tdx.json",
    "verification_certificate.json",
    
    # Verification output
    "tdx_verification.txt",
    
    # Environment
    ".env",
]

PATTERNS_TO_REMOVE = [
    "*.dump",
    "*.bin",
]

def main():
    print("[Reset] Cleaning up Lab 10: Confidential AI with TDX...")
    
    lab_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Remove specific files
    for f in FILES_TO_REMOVE:
        path = os.path.join(lab_dir, f)
        if os.path.isfile(path):
            os.remove(path)
            print(f"  [✓] Removed: {f}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  [✓] Removed: {f}/")
    
    # Remove pattern matches
    for pattern in PATTERNS_TO_REMOVE:
        for path in glob.glob(os.path.join(lab_dir, pattern)):
            if os.path.isfile(path):
                os.remove(path)
                print(f"  [✓] Removed: {os.path.basename(path)}")
    
    # Remove __pycache__
    pycache = os.path.join(lab_dir, "__pycache__")
    if os.path.isdir(pycache):
        shutil.rmtree(pycache)
        print(f"  [✓] Removed: __pycache__/")
    
    # Remove .venv if exists
    venv = os.path.join(lab_dir, ".venv")
    if os.path.isdir(venv):
        print(f"  [!] Virtual environment found at .venv/ - skipping (remove manually if needed)")
    
    print("\n[✓] Lab 10 reset complete!")
    print("\nTo run the lab:")
    print("  python 0_check_tdx.py               # Verify TDX is active")
    print("  python 1_train_proprietary_model.py # Train model")
    print("  python 2_victim_inference_server.py # Terminal 1: Start server")
    print("  sudo .venv/bin/python 3_attacker_memory_reader.py  # Terminal 2: Attack")
    print("  python 4_verify_tdx_protection.py   # Verify protection scope")

if __name__ == "__main__":
    main()
