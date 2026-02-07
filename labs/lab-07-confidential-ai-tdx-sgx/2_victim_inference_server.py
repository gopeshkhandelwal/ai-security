#!/usr/bin/env python3
"""
Step 2: Victim Inference Server (UNPROTECTED)

Runs an ML inference server with model weights exposed in memory.
Demonstrates vulnerability to memory extraction attacks.

Usage:
  Terminal 1: python 2_victim_inference_server.py
  Terminal 2: sudo .venv/bin/python 3_attacker_memory_reader.py

Author: GopeshK
License: MIT License
Disclaimer: Educational purposes only. Do not use for malicious purposes.
"""

import os
import sys
import time
import json
import signal

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "proprietary_model.h5"
PID_FILE = "victim_process.json"

def signal_handler(sig, frame):
    print("\n[VICTIM] Shutting down...")
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         VICTIM: Inference Server (UNPROTECTED)                        ║
║         Model and weights exposed in memory                           ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    if not os.path.exists(MODEL_PATH):
        print("[!] Model not found. Run: python 1_train_proprietary_model.py")
        sys.exit(1)
    
    # Load model
    print("[VICTIM] Loading proprietary model...")
    model = load_model(MODEL_PATH)
    weights = model.get_weights()
    
    # Get actual memory addresses of weight data
    weight_info = []
    for i, w in enumerate(weights):
        addr = w.ctypes.data
        size = w.nbytes
        weight_info.append({
            "layer": i,
            "address": addr,
            "size": size,
            "shape": list(w.shape),
            "dtype": str(w.dtype)
        })
        print(f"[VICTIM] Layer {i}: addr={hex(addr)}, size={size} bytes, shape={w.shape}")
    
    # Save PID and memory addresses for attacker
    process_info = {
        "pid": os.getpid(),
        "weights": weight_info
    }
    
    with open(PID_FILE, "w") as f:
        json.dump(process_info, f, indent=2)
    
    print(f"\n[VICTIM] PID: {os.getpid()}")
    print(f"[VICTIM] Process info saved to: {PID_FILE}")
    print("\n" + "="*70)
    print("⚠️  MODEL LOADED - MEMORY EXPOSED - WAITING FOR ATTACK...")
    print("="*70)
    print("\nIn another terminal, run:")
    print("  sudo .venv/bin/python 3_attacker_memory_reader.py")
    print("\nPress Ctrl+C to stop.\n")
    
    # Keep running inference to simulate real server
    np.random.seed(42)
    inference_count = 0
    
    while True:
        # Simulate incoming inference requests
        sample_data = np.random.randn(1, 50)
        prediction = model.predict(sample_data, verbose=0)
        inference_count += 1
        
        if inference_count % 10 == 0:
            print(f"[VICTIM] Processed {inference_count} inference requests...", flush=True)
        
        time.sleep(1)

if __name__ == "__main__":
    main()
