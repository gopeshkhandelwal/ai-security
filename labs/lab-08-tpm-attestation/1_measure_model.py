#!/usr/bin/env python3
"""
Step 1: Measure AI Model into TPM PCR

Extends a TPM Platform Configuration Register (PCR) with the SHA256 hash
of the AI model. This cryptographically binds the model to the boot state.

PCR Extension: PCR_new = SHA256(PCR_old || digest)
- One-way operation (cannot undo)
- Creates chain of trust from boot to model

Setup Requirements:
    1. sudo apt install tpm2-tools
    2. sudo usermod -aG tss $USER && newgrp tss
    3. Run 0_check_tpm.py first

Author: GopeshK
License: MIT License
Disclaimer: Educational purposes only.
"""

import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

# PCR index for model measurement (14-15 are typically for application use)
MODEL_PCR_INDEX = 14


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║       Measure AI Model into TPM Platform Configuration Register       ║
║                                                                       ║
║   Extends PCR[14] with SHA256(model) - binds model to boot chain      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def load_tpm_status():
    """Load TPM availability status."""
    try:
        with open(".tpm_status.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[!] Run 0_check_tpm.py first")
        sys.exit(1)


def compute_model_hash(model_path: str) -> str:
    """Compute SHA256 hash of model file."""
    print(f"\n[1/3] Computing model hash...")
    print(f"    Model: {model_path}")
    
    sha256 = hashlib.sha256()
    file_size = 0
    
    with open(model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
            file_size += len(chunk)
    
    digest = sha256.hexdigest()
    
    print(f"    Size: {file_size:,} bytes")
    print(f"    SHA256: {digest}")
    
    return digest


def read_pcr_value(pcr_index: int) -> str:
    """Read current PCR value."""
    result = subprocess.run(
        ["tpm2_pcrread", f"sha256:{pcr_index}"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        return None
    
    # Parse output to get hex value
    for line in result.stdout.split('\n'):
        if '0x' in line:
            # Extract hex value
            parts = line.split('0x')
            if len(parts) >= 2:
                return parts[1].strip()
    
    return None


def extend_pcr_with_model(pcr_index: int, model_hash: str) -> bool:
    """Extend PCR with model hash using real TPM hardware."""
    print(f"\n[2/3] Extending PCR[{pcr_index}] with model hash...")
    print(f"    [HARDWARE] Using real TPM device /dev/tpmrm0")
    
    # Read current PCR value
    pcr_before = read_pcr_value(pcr_index)
    if pcr_before:
        print(f"    [HARDWARE] PCR[{pcr_index}] before: 0x{pcr_before[:32]}...")
    
    # Real TPM extend - pass hash as hex string directly
    result = subprocess.run(
        ["tpm2_pcrextend", f"{pcr_index}:sha256={model_hash}"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"    [ERROR] PCR extend failed: {result.stderr}")
        return False
    
    # Read new PCR value
    pcr_after = read_pcr_value(pcr_index)
    if pcr_after:
        print(f"    [HARDWARE] PCR[{pcr_index}] after:  0x{pcr_after[:32]}...")
    
    print(f"    [HARDWARE] ✓ PCR[{pcr_index}] extended successfully")
    return True


def save_measurement_record(model_path: str, model_hash: str, pcr_index: int):
    """Save measurement record for later verification."""
    print(f"\n[3/3] Saving measurement record...")
    
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_path": model_path,
        "model_hash": model_hash,
        "pcr_index": pcr_index,
        "pcr_value_after": read_pcr_value(pcr_index),
        "hash_algorithm": "sha256"
    }
    
    record_path = "model_measurement.json"
    with open(record_path, 'w') as f:
        json.dump(record, f, indent=2)
    
    print(f"    [✓] Saved to {record_path}")
    
    return record


def create_sample_model():
    """Create a sample model for testing if none exists."""
    model_path = "test_model.h5"
    
    if os.path.exists(model_path):
        return model_path
    
    print("[*] Creating sample model for testing...")
    
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import warnings
        warnings.filterwarnings("ignore")
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Input
        import numpy as np
        
        model = Sequential([
            Input(shape=(10,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Train briefly
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y, epochs=1, verbose=0)
        
        model.save(model_path)
        print(f"    [✓] Created {model_path}")
        
    except ImportError:
        # Create a dummy file if TensorFlow not available
        with open(model_path, 'wb') as f:
            f.write(os.urandom(1024 * 100))  # 100KB dummy
        print(f"    [✓] Created dummy {model_path}")
    
    return model_path


def main():
    print_banner()
    
    # Load TPM status
    status = load_tpm_status()
    
    if not status.get("tpm_available", False):
        print("❌ TPM 2.0 hardware not available!")
        print("   This lab requires real Intel TPM hardware.")
        sys.exit(1)
    
    print("════════════════════════════════════════════════════════════════════")
    print("  [HARDWARE] TPM 2.0 available - using real hardware measurement")
    print("════════════════════════════════════════════════════════════════════")
    
    # Get or create model
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not model_path or not os.path.exists(model_path):
        model_path = create_sample_model()
    
    # Compute model hash
    model_hash = compute_model_hash(model_path)
    
    # Extend PCR with model hash (real TPM hardware)
    success = extend_pcr_with_model(MODEL_PCR_INDEX, model_hash)
    
    if success:
        # Save measurement record
        record = save_measurement_record(model_path, model_hash, MODEL_PCR_INDEX)
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ MODEL MEASURED INTO TPM                                           ║
╚══════════════════════════════════════════════════════════════════════╝

  Model:     {model_path}
  Hash:      {model_hash[:32]}...
  PCR Index: {MODEL_PCR_INDEX}
  
  The model hash is now cryptographically bound to the TPM's PCR.
  Any change to the model will result in a different PCR value,
  causing attestation to fail.
  
  Next: python 2_generate_quote.py
        """)
    else:
        print("[!] Measurement failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
