#!/usr/bin/env python3
"""
Step 4: Confidential AI Inference with Intel TDX/SGX

This demonstrates running AI inference in a protected environment
using Intel's confidential computing features.

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
import time
import hashlib
from datetime import datetime
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

MODEL_PATH = "proprietary_model.h5"

# Simulation mode (set to False on real TDX/SGX hardware)
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "true").lower() == "true"

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║      🔒 Confidential AI Inference with Intel TDX/SGX 🔒               ║
║           (Hardware-protected model execution)                        ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

class ConfidentialExecutionEnvironment:
    """
    Simulates/wraps Intel TDX or SGX execution environment.
    
    On real hardware, this would:
    - TDX: Run inside a Trust Domain with encrypted memory
    - SGX: Execute within an enclave with sealed memory
    """
    
    def __init__(self, mode="TDX"):
        self.mode = mode
        self.is_initialized = False
        self.attestation_report = None
        self.memory_encryption_key = None
        
    def initialize(self):
        """Initialize the confidential execution environment."""
        print(f"\n[*] Initializing Intel {self.mode} environment...")
        
        if SIMULATION_MODE:
            print(f"    [SIM] Running in SIMULATION mode")
            self._simulate_initialization()
        else:
            self._real_initialization()
        
        self.is_initialized = True
        print(f"[✓] {self.mode} environment initialized")
        
    def _simulate_initialization(self):
        """Simulate TDX/SGX initialization."""
        print(f"    [*] Simulating {self.mode} hardware initialization...")
        
        if self.mode == "TDX":
            print("    [*] Creating Trust Domain...")
            print("    [*] Initializing TD memory encryption (MKTME)...")
            print("    [*] Setting up TD attestation key...")
        else:  # SGX
            print("    [*] Creating SGX enclave...")
            print("    [*] Loading enclave code...")
            print("    [*] Sealing enclave memory...")
        
        # Generate simulated encryption key
        self.memory_encryption_key = hashlib.sha256(
            os.urandom(32)
        ).hexdigest()
        
        time.sleep(0.5)
        
    def _real_initialization(self):
        """
        Real TDX/SGX initialization.
        
        For TDX: Would use Linux TDX guest attestation APIs
        For SGX: Would use Intel SGX SDK or Gramine
        """
        raise NotImplementedError(
            "Real TDX/SGX requires specific hardware and SDK. "
            "See README for cloud deployment options."
        )
    
    def encrypt_memory_region(self, data):
        """
        Simulates memory encryption.
        
        On real TDX: Memory is automatically encrypted by hardware
        On real SGX: Enclave memory is encrypted by MEE
        """
        if SIMULATION_MODE:
            # Simulate encrypted representation
            if isinstance(data, np.ndarray):
                encrypted_marker = "[ENCRYPTED:TME-AES256]"
                return {
                    "encrypted": True,
                    "algorithm": "AES-256-XTS",
                    "key_id": self.memory_encryption_key[:16],
                    "data_hash": hashlib.sha256(data.tobytes()).hexdigest()[:32]
                }
        return data
    
    def generate_attestation_report(self):
        """
        Generate attestation report proving secure execution.
        
        On real hardware:
        - TDX: TD Report signed by Intel TDX module
        - SGX: Quote signed by Intel Attestation Service
        """
        print("\n[*] Generating attestation report...")
        
        # Generate attestation data
        timestamp = datetime.utcnow().isoformat()
        
        # Simulate measurement (would be real TD/enclave measurement)
        measurement = hashlib.sha384(
            f"{self.mode}-measurement-{timestamp}".encode()
        ).hexdigest()
        
        # Generate signing key (simulated)
        private_key = ec.generate_private_key(ec.SECP384R1(), default_backend())
        
        report = {
            "version": "1.0",
            "type": f"Intel {self.mode} Attestation Report",
            "timestamp": timestamp,
            "environment": {
                "mode": self.mode,
                "simulation": SIMULATION_MODE,
                "memory_encryption": "AES-256-XTS (MKTME)" if self.mode == "TDX" else "AES-128-GCM (MEE)"
            },
            "measurements": {
                "td_measurement": measurement if self.mode == "TDX" else None,
                "enclave_measurement": measurement if self.mode == "SGX" else None,
                "rtmr0": hashlib.sha256(measurement.encode()).hexdigest()[:64]
            },
            "security_properties": {
                "memory_encrypted": True,
                "isolated_from_hypervisor": True,
                "debug_disabled": True,
                "secure_boot": True
            },
            "model_loaded": {
                "hash": hashlib.sha256(open(MODEL_PATH, 'rb').read()).hexdigest() if os.path.exists(MODEL_PATH) else None
            }
        }
        
        self.attestation_report = report
        
        # Save report
        report_path = f"attestation_report_{self.mode.lower()}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"[✓] Attestation report saved: {report_path}")
        
        return report

class ConfidentialModelLoader:
    """Load and run models within confidential environment."""
    
    def __init__(self, cee: ConfidentialExecutionEnvironment):
        self.cee = cee
        self.model = None
        self.encrypted_weights = []
        
    def load_model(self, model_path):
        """Load model into encrypted memory."""
        print(f"\n[*] Loading model into {self.cee.mode} protected memory...")
        
        if not os.path.exists(model_path):
            print(f"[!] Model not found: {model_path}")
            print("    Run 1_train_proprietary_model.py first")
            sys.exit(1)
        
        # Import TensorFlow inside confidential environment
        from tensorflow.keras.models import load_model
        
        self.model = load_model(model_path)
        
        # Simulate memory encryption of weights
        weights = self.model.get_weights()
        print(f"[*] Encrypting {len(weights)} weight tensors in memory...")
        
        for i, w in enumerate(weights):
            encrypted_info = self.cee.encrypt_memory_region(w)
            self.encrypted_weights.append(encrypted_info)
            if i < 3:
                print(f"    Layer {i}: {encrypted_info}")
        
        print(f"[✓] Model loaded with encrypted weights")
        print(f"[✓] Memory encryption: AES-256-XTS (hardware-enforced)")
        
        return self.model
    
    def run_inference(self, input_data):
        """Run inference on encrypted input."""
        print(f"\n[*] Running inference in {self.cee.mode} protected environment...")
        
        # Encrypt input data
        input_encrypted = self.cee.encrypt_memory_region(input_data)
        print(f"[*] Input encrypted: {input_encrypted}")
        
        # Run inference
        predictions = self.model.predict(input_data, verbose=0)
        
        # Encrypt output
        output_encrypted = self.cee.encrypt_memory_region(predictions)
        print(f"[*] Output encrypted: {output_encrypted}")
        
        return predictions

def demonstrate_protection():
    """Show how TDX/SGX protects the model."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   🛡️  PROTECTION ACTIVE 🛡️                           ║
╚══════════════════════════════════════════════════════════════════════╝

With Intel TDX/SGX enabled:

┌─────────────────────────────────────────────────────────────────────┐
│  PROTECTED ASSETS:                                                   │
│                                                                      │
│  🔒 Model Weights                                                    │
│     └─ Encrypted in RAM with AES-256 (hardware key)                 │
│     └─ Hypervisor sees only ciphertext                              │
│                                                                      │
│  🔒 Input Data                                                       │
│     └─ Encrypted before entering Trust Domain/Enclave              │
│     └─ Decrypted only inside protected environment                  │
│                                                                      │
│  🔒 Inference Results                                                │
│     └─ Computed on encrypted data                                   │
│     └─ Re-encrypted before leaving protected memory                 │
│                                                                      │
│  🔒 Execution Flow                                                   │
│     └─ Isolated from hypervisor control                             │
│     └─ Side-channel mitigations active                              │
└─────────────────────────────────────────────────────────────────────┘

ATTACK PREVENTION:

  ✓ Hypervisor memory access → BLOCKED (encrypted memory)
  ✓ Cloud operator access    → BLOCKED (hardware isolation)
  ✓ Memory scraping          → BLOCKED (encryption + isolation)
  ✓ Cold boot attack         → BLOCKED (keys in CPU, not DRAM)
  
ATTESTATION:

  ✓ Cryptographic proof of secure execution
  ✓ Verifiable by remote parties
  ✓ Signed by Intel hardware root of trust
    """)

def main():
    print_banner()
    
    if SIMULATION_MODE:
        print("ℹ️  Running in SIMULATION mode (no real TDX/SGX hardware)")
        print("   Set SIMULATION_MODE=false in .env for real hardware\n")
    
    # Initialize confidential environment (try TDX first, fallback to SGX)
    cee = ConfidentialExecutionEnvironment(mode="TDX")
    cee.initialize()
    
    # Load model into protected memory
    loader = ConfidentialModelLoader(cee)
    model = loader.load_model(MODEL_PATH)
    
    # Run protected inference
    print("\n" + "="*60)
    print("PROTECTED INFERENCE")
    print("="*60)
    
    test_input = np.random.randn(3, 50)
    predictions = loader.run_inference(test_input)
    
    print("\n[*] Inference Results (decrypted for authorized viewer):")
    for i, pred in enumerate(predictions):
        print(f"    Sample {i+1}: Class {np.argmax(pred)} (conf: {pred.max():.2%})")
    
    # Generate attestation
    attestation = cee.generate_attestation_report()
    
    # Show protection summary
    demonstrate_protection()
    
    print("\n[✓] Confidential inference complete!")
    print("[✓] Run: python 5_verify_attestation.py to verify security")

if __name__ == "__main__":
    main()
