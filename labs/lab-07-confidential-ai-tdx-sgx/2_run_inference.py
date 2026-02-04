#!/usr/bin/env python3
"""
Step 2: Run Normal Inference (Without Protection)

This demonstrates running inference without confidential computing.
The model and data are exposed in memory.

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json
import sys
from tensorflow.keras.models import load_model

MODEL_PATH = "proprietary_model.h5"
METADATA_PATH = "model_metadata.json"

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         Running Inference WITHOUT Protection                          ║
║          (Model and data exposed in memory)                           ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def load_proprietary_model():
    """Load the model into memory (UNPROTECTED)."""
    print("[*] Loading proprietary model into memory...")
    
    if not os.path.exists(MODEL_PATH):
        print("[!] Error: Model not found. Run 1_train_proprietary_model.py first")
        sys.exit(1)
    
    model = load_model(MODEL_PATH)
    
    # Display memory location (demonstrates exposure)
    print(f"[*] Model loaded at memory address: {hex(id(model))}")
    
    # Get model weights location
    weights = model.get_weights()
    print(f"[*] Model has {len(weights)} weight tensors in memory")
    
    for i, w in enumerate(weights[:3]):  # Show first 3
        print(f"    - Layer {i}: shape={w.shape}, memory={hex(id(w))}")
    
    return model, weights

def run_inference(model):
    """Run inference on sample data."""
    print("\n[*] Running inference on sample data...")
    
    # Generate sample inference data (simulates sensitive customer data)
    np.random.seed(123)
    sensitive_data = np.random.randn(5, 50)  # 5 samples, 50 features
    
    print(f"[*] Sensitive input data loaded at: {hex(id(sensitive_data))}")
    print(f"[*] Input shape: {sensitive_data.shape}")
    
    # Run prediction
    predictions = model.predict(sensitive_data, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print("\n[*] Inference Results:")
    for i, (pred, conf) in enumerate(zip(predicted_classes, predictions.max(axis=1))):
        print(f"    Sample {i+1}: Class {pred} (confidence: {conf:.2%})")
    
    return sensitive_data, predictions

def show_vulnerability():
    """Demonstrate the vulnerability."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    ⚠️  SECURITY WARNING ⚠️                            ║
╚══════════════════════════════════════════════════════════════════════╝

Without Intel TDX/SGX protection, the following are EXPOSED:

┌────────────────────────────────────────────────────────────────────┐
│ EXPOSED IN MEMORY:                                                  │
│                                                                     │
│  📦 Model Weights (~500KB of proprietary parameters)                │
│     └─ Any process with memory access can read these               │
│                                                                     │
│  📊 Input Data (sensitive customer/business data)                   │
│     └─ Input features visible to hypervisor                        │
│                                                                     │
│  📈 Predictions (confidential inference results)                    │
│     └─ Output classifications exposed                              │
│                                                                     │
│  🔑 Model Architecture (proprietary design)                         │
│     └─ Layer structure and activations visible                     │
└────────────────────────────────────────────────────────────────────┘

WHO CAN ACCESS THIS DATA:

  ❌ Cloud operators with hypervisor access
  ❌ Other VMs via side-channel attacks
  ❌ Compromised OS kernel
  ❌ Physical attackers (cold boot)
  ❌ Malicious cloud employees

Run: python 3_memory_attack_demo.py to see the attack
    """)

def main():
    print_banner()
    
    model, weights = load_proprietary_model()
    sensitive_data, predictions = run_inference(model)
    
    # Store references for attack demo
    import pickle
    demo_data = {
        "model_memory_addr": hex(id(model)),
        "weights_sample": weights[0][:5, :5].tolist(),  # First 5x5 of first layer
        "input_sample": sensitive_data[0].tolist(),
        "prediction_sample": predictions[0].tolist()
    }
    
    with open("inference_memory_dump.json", "w") as f:
        json.dump(demo_data, f, indent=2)
    
    show_vulnerability()
    print("[✓] Step 2 complete. Run: python 3_memory_attack_demo.py")

if __name__ == "__main__":
    main()
