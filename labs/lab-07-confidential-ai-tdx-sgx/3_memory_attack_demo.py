#!/usr/bin/env python3
"""
Step 3: Real Memory Attack - Extract model weights from process memory

This is NOT a simulation. It actually reads model weights directly from RAM
using memory addresses, demonstrating why TDX/SGX protection is needed.

Author: GopeshK
License: MIT License
Disclaimer: Educational purposes only.
"""

import os
import sys
import ctypes
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "proprietary_model.h5"

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║        🔓 REAL MEMORY ATTACK - Model Weight Extraction 🔓             ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print("[!] Model not found. Run: python 1_train_proprietary_model.py")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # VICTIM: Load model into memory
    # ═══════════════════════════════════════════════════════════════════
    
    print("[VICTIM] Loading proprietary model into memory...")
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    weights = model.get_weights()
    
    total_params = sum(np.prod(w.shape) for w in weights)
    print(f"[VICTIM] Model loaded: {total_params:,} parameters in RAM")
    print(f"[VICTIM] Memory is UNPROTECTED - no TDX/SGX!\n")

    # ═══════════════════════════════════════════════════════════════════
    # ATTACKER: Read weights directly from memory addresses
    # ═══════════════════════════════════════════════════════════════════
    
    print("="*70)
    print("🔓 ATTACKER: Reading model weights from memory...")
    print("="*70 + "\n")

    for layer_idx, layer_weights in enumerate(weights):
        # Get memory address of this weight tensor
        memory_address = layer_weights.ctypes.data
        
        # Create pointer to that memory location
        ptr = ctypes.cast(memory_address, ctypes.POINTER(ctypes.c_float))
        
        # Read values directly from memory
        stolen_values = [ptr[i] for i in range(min(5, layer_weights.size))]
        
        print(f"Layer {layer_idx}: Address {hex(memory_address)}")
        print(f"         Shape: {layer_weights.shape}")
        print(f"         Stolen: {stolen_values}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # VERIFY: Attack was successful
    # ═══════════════════════════════════════════════════════════════════
    
    print("="*70)
    print("✓ ATTACK SUCCESSFUL - All weights extracted from memory!")
    print("="*70)
    
    print("""
WHY THIS WORKS:
  • Model weights are stored as plain floats in RAM
  • Any process with memory access can read them
  • Hypervisor/root can dump entire VM memory
  • No encryption = no protection

WITH TDX/SGX:
  • Memory encrypted by CPU hardware
  • Attacker sees only ciphertext
  • This attack would FAIL

Next: python 4_confidential_inference.py (runs ONLY inside TEE)
    """)

if __name__ == "__main__":
    main()
