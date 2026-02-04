#!/usr/bin/env python3
"""
Reset Lab 07 - Remove generated files

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
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
    "inference_memory_dump.json",
    
    # Attestation files
    "attestation_report_tdx.json",
    "attestation_report_sgx.json",
    "verification_certificate.json",
    
    # Environment
    ".env",
]

PATTERNS_TO_REMOVE = [
    "*.dump",
    "*.bin",
    "*.sig",
]

def main():
    print("[Reset] Cleaning up Lab 07: Confidential AI...")
    
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
    
    print("\n[✓] Lab 07 reset complete!")
    print("\nTo run the lab:")
    print("  python 0_check_hardware.py   # Check TDX/SGX support")
    print("  python 1_train_proprietary_model.py")
    print("  python 2_run_inference.py")
    print("  python 3_memory_attack_demo.py")
    print("  python 4_confidential_inference.py")
    print("  python 5_verify_attestation.py")
    print("  python 6_protected_memory_demo.py")

if __name__ == "__main__":
    main()
