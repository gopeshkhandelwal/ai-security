#!/usr/bin/env python3
"""
Reset All Labs - Remove all generated files from all labs

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import os
import shutil

LAB_DIR = os.path.dirname(os.path.abspath(__file__))

# Files to remove per lab
LAB_FILES = {
    "lab-01-malicious-code-injection": [
        "model.h5",
        "vectorizer.joblib",
        "responses.json",
    ],
    "lab-02-model-signing": [
        "keras_model.h5",
        "keras_model.h5.sig",
        "keras_model.h5.bundle",
        "cosign.key",
        "cosign.pub",
        "vectorizer.joblib",
        "responses.json",
    ],
    "lab-03-model-stealing": [
        # No generated files, just uses API
    ],
    "lab-04-rag-data-extraction": [
        "medical_knowledge_base.json",
        "attack_results.json",
    ],
    "lab-05-malicious-code-injection": [
        "model.h5",
        "vectorizer.joblib",
        "responses.json",
    ],
    "lab-06-model-signing": [
        "keras_model.h5",
        "keras_model.h5.sig",
        "keras_model.h5.bundle",
        "cosign.key",
        "cosign.pub",
        "vectorizer.joblib",
        "responses.json",
    ],
    "lab-07-confidential-ai-tdx-sgx": [
        "proprietary_model.h5",
        "vectorizer.joblib",
        "model_metadata.json",
        "stolen_model_weights.npz",
        "inference_memory_dump.json",
        "attestation_report_tdx.json",
        "attestation_report_sgx.json",
        "verification_certificate.json",
    ],
    "lab-08-amx-accelerated-scanning": [
        "test_models",  # directory
        "scan_results",  # directory
        "mlbom_output",  # directory
        ".amx_status.json",
    ],
}


print("=" * 60)
print("  RESET ALL LABS")
print("=" * 60)

for lab, files in LAB_FILES.items():
    lab_path = os.path.join(LAB_DIR, lab)
    if not os.path.isdir(lab_path):
        continue
    
    print(f"\n[{lab}] Cleaning...")
    
    # Remove files
    for f in files:
        path = os.path.join(lab_path, f)
        if os.path.isfile(path):
            os.remove(path)
            print(f"  [✓] Removed: {f}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  [✓] Removed: {f}/")
    

print("\n" + "=" * 60)
print("[✓] All labs reset!")
print("=" * 60)
print("\nTo run labs:")
print("  cd lab-01-supply-chain-attack && python 2_victim_loads_model.py")
print("  cd lab-02-model-stealing && python 1_proprietary_model.py")
print("  cd lab-03-llm-agent-exploitation && python 1_vulnerable_agent.py")
print("  cd lab-04-rag-data-extraction && python 1_create_knowledge_base.py")
print("  cd lab-05-malicious-code-injection && python 1_train_model.py")
print("  cd lab-06-model-signing && python 1_train_model.py")
print("  cd lab-07-confidential-ai-tdx-sgx && python 0_check_hardware.py")
print("  cd lab-08-amx-accelerated-scanning && python 0_check_amx_support.py")
