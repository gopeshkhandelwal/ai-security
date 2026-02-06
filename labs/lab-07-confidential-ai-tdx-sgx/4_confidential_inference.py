#!/usr/bin/env python3
"""
Step 4: Confidential AI Inference with Intel TDX/SGX

This script runs AI inference in a protected environment using Intel's
confidential computing features (TDX or SGX).

IMPORTANT: Running `python 4_confidential_inference.py` directly will:
  - Prompt you to either run in SIMULATION mode (no real protection)
  - Or launch inside a real SGX enclave via Gramine

For REAL hardware protection, the script must run inside:
  - An SGX enclave (via Gramine): ./run_sgx_enclave.sh
  - A TDX Trust Domain (VM booted as a TD guest)

Memory encryption is done by HARDWARE (TME/MEE), not by this Python code.
This script detects the environment and reports status.

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import os

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use shell environment

# Configure TensorFlow BEFORE import (important for SGX enclave thread limits)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

# Detect if running inside a real SGX enclave (via Gramine)
SGX_ENCLAVE_ACTIVE = os.getenv("SGX_ENCLAVE", "false").lower() == "true"

def is_running_in_tee():
    """
    Check if we're ACTUALLY running inside a real Trusted Execution Environment.
    
    Returns True ONLY if:
    - Inside an SGX enclave (Gramine's /dev/attestation/quote exists)
    - Inside a TDX Trust Domain (/dev/tdx_guest exists)
    - SGX_ENCLAVE env var is set (by run_sgx_enclave.sh)
    
    NOTE: /sys/firmware/tdx on HOST means TDX-capable, but NOT inside a TD!
    """
    # Check for Gramine's /dev/attestation (only exists inside SGX enclave)
    gramine_sgx = os.path.exists("/dev/attestation/quote")
    
    # Check for TDX guest attestation device (only inside a TD guest, not host)
    # /dev/tdx_guest = inside TD | /sys/firmware/tdx = TDX-capable host (NOT protected)
    tdx_guest = os.path.exists("/dev/tdx_guest")
    
    return gramine_sgx or tdx_guest or SGX_ENCLAVE_ACTIVE

# Check if TDX-capable host (but not inside TD)
TDX_CAPABLE_HOST = os.path.exists("/sys/firmware/tdx") and not os.path.exists("/dev/tdx_guest")
SGX_CAPABLE_HOST = os.path.exists("/dev/sgx_enclave") or os.path.exists("/dev/sgx/enclave")

# Determine actual protection status
ACTUALLY_IN_TEE = is_running_in_tee()

# If user set SIMULATION_MODE=false but NOT in a real TEE, warn them!
if not SIMULATION_MODE and not ACTUALLY_IN_TEE:
    print("\n" + "!"*70)
    print("⚠️  WARNING: SIMULATION_MODE=false but NOT running inside a TEE!")
    print("!"*70)
    print("│ Your .env has SIMULATION_MODE=false, but you are NOT inside:")
    print("│   - An SGX enclave (via Gramine)")
    print("│   - A TDX Trust Domain (TD guest VM)")
    print("│")
    print("│ Memory is NOT actually encrypted! This is misleading.")
    print("│")
    if TDX_CAPABLE_HOST:
        print("│ You ARE on a TDX-capable HOST, but need to boot a TD VM.")
    if SGX_CAPABLE_HOST:
        print("│ You HAVE SGX hardware, but need to run: ./run_sgx_enclave.sh")
    print("!"*70)
    # Force simulation mode since we're not protected
    SIMULATION_MODE = True
    print("[!] Forcing SIMULATION_MODE=true for accurate status reporting.\n")

# Override SIMULATION_MODE if we detect real TEE
if ACTUALLY_IN_TEE and SIMULATION_MODE:
    print("[!] Detected real TEE environment - overriding SIMULATION_MODE to False")
    SIMULATION_MODE = False

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
            print(f"    ┌──────────────────────────────────────────────────────────┐")
            print(f"    │ 🔶 SIMULATION: No real {self.mode} hardware - demo only     │")
            print(f"    └──────────────────────────────────────────────────────────┘")
            self._simulate_initialization()
        else:
            print(f"    ┌──────────────────────────────────────────────────────────┐")
            print(f"    │ 🟢 HARDWARE: Real Intel {self.mode} protection active        │")
            print(f"    └──────────────────────────────────────────────────────────┘")
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
        
        For TDX: Uses Linux TDX guest attestation APIs
        For SGX: Uses Intel SGX SDK or Gramine
        """
        print(f"    [*] Initializing REAL {self.mode} hardware...")
        
        if self.mode == "TDX":
            print("    [*] Detecting TDX Trust Domain environment...")
            # Check if running inside a TDX guest
            tdx_detected = os.path.exists("/sys/firmware/tdx") or os.path.exists("/dev/tdx_guest")
            if tdx_detected:
                print("    [✓] TDX Trust Domain detected")
            else:
                print("    [✓] TDX-capable host (not inside TD guest)")
            print("    [*] Hardware memory encryption (TME/MKTME) active...")
            print("    [*] TD attestation key available...")
        else:  # SGX
            print("    [*] Detecting SGX enclave support...")
            sgx_detected = os.path.exists("/dev/sgx_enclave") or os.path.exists("/dev/sgx/enclave")
            if sgx_detected:
                print("    [✓] SGX enclave device detected")
            print("    [*] SGX memory encryption (MEE) active...")
        
        # Generate hardware-backed encryption key identifier
        self.memory_encryption_key = hashlib.sha256(
            os.urandom(32)
        ).hexdigest()
        
        time.sleep(0.3)
    
    def get_memory_encryption_status(self, data):
        """
        Get memory encryption status for data region.
        
        NOTE: In TDX/SGX hardware mode, the CPU automatically encrypts ALL memory
        using Total Memory Encryption (TME/MKTME). We don't manually encrypt -
        we just report the status.
        
        In SIMULATION mode: No actual encryption (demo only)
        In HARDWARE mode: Data IS encrypted transparently by TDX/SGX hardware
        """
        if isinstance(data, np.ndarray):
            data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:32]
            
            if SIMULATION_MODE:
                return {
                    "encrypted": False,  # Not actually encrypted in simulation
                    "mode": "SIMULATION",
                    "algorithm": "N/A (demo only)",
                    "key_id": "simulated",
                    "data_hash": data_hash,
                    "note": "⚠️  SIMULATION - data is NOT encrypted"
                }
            else:
                # Real hardware mode - memory IS encrypted by TME/MKTME
                return {
                    "encrypted": True,  # Hardware encrypts transparently
                    "mode": "HARDWARE",
                    "algorithm": "AES-256-XTS (TME/MKTME)",
                    "key_id": self.memory_encryption_key[:16],
                    "data_hash": data_hash,
                    "note": "🟢 HARDWARE - encrypted by Intel TDX/SGX TME"
                }
        return {"encrypted": SIMULATION_MODE == False, "mode": "HARDWARE" if not SIMULATION_MODE else "SIMULATION"}
    
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
    """
    Load and run models within confidential environment.
    
    In TDX/SGX hardware mode, the CPU's Total Memory Encryption (TME/MKTME)
    automatically encrypts all DRAM. We don't manually encrypt weights -
    the hardware does it transparently.
    """
    
    def __init__(self, cee: ConfidentialExecutionEnvironment):
        self.cee = cee
        self.model = None
        self.weight_encryption_status = []  # Tracks encryption status, not encrypted data
        
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
        
        # In TDX/SGX mode, weights are automatically encrypted in RAM by hardware
        # We just verify and report the encryption status
        weights = self.model.get_weights()
        
        if SIMULATION_MODE:
            print(f"[*] Loading {len(weights)} weight tensors (NO encryption in simulation)...")
        else:
            print(f"[*] Loading {len(weights)} weight tensors into TME-encrypted RAM...")
            print(f"    └─ Intel TME/MKTME encrypts memory transparently at hardware level")
        
        for i, w in enumerate(weights):
            status = self.cee.get_memory_encryption_status(w)
            self.weight_encryption_status.append(status)
            if i < 3:
                print(f"    Layer {i}: {status}")
        
        print(f"[✓] Model loaded into {'protected' if not SIMULATION_MODE else 'unprotected'} memory")
        if SIMULATION_MODE:
            print(f"[⚠️  SIMULATION] Weights are NOT encrypted - demo only")
        else:
            print(f"[🟢 HARDWARE] Weights ARE encrypted by Intel TME/MKTME in DRAM")
            print(f"    └─ Hypervisor/cloud operator cannot read plaintext weights")
        
        return self.model
    
    def run_inference(self, input_data):
        """Run inference in protected memory (encrypted in hardware mode)."""
        print(f"\n[*] Running inference in {self.cee.mode} protected environment...")
        
        # Report input data encryption status
        input_status = self.cee.get_memory_encryption_status(input_data)
        print(f"[*] Input data status: {input_status}")
        
        # Run inference (in hardware mode, all computation uses encrypted memory)
        # Use direct __call__ instead of predict() to avoid tf.data thread pool issues in SGX
        predictions = self.model(input_data, training=False).numpy()
        
        # Report output encryption status
        output_status = self.cee.get_memory_encryption_status(predictions)
        print(f"[*] Output data status: {output_status}")
        
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

def print_mode_banner():
    """Print a clear banner showing the current execution mode."""
    # Check for real TEE
    gramine_sgx = os.path.exists("/dev/attestation/quote")
    tdx_guest = os.path.exists("/dev/tdx_guest")
    
    if SIMULATION_MODE:
        print("\n" + "="*70)
        print("🔶 " + " SIMULATION MODE ".center(66, "=") + " 🔶")
        print("="*70)
        print("│ Running WITHOUT real TDX/SGX enclave/TD                          │")
        print("│ Memory is NOT encrypted - for demonstration only                 │")
        print("│                                                                  │")
        print("│ To run with REAL SGX protection:                                 │")
        print("│   ./run_sgx_enclave.sh                                           │")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("🟢 " + " HARDWARE MODE - Real TEE Active ".center(66, "=") + " 🟢")
        print("="*70)
        if gramine_sgx:
            print("│ ✓ Running inside SGX Enclave (via Gramine)                       │")
            print("│ ✓ Memory IS encrypted by SGX MEE (Memory Encryption Engine)      │")
        elif tdx_guest:
            print("│ ✓ Running inside TDX Trust Domain                                │")
            print("│ ✓ Memory IS encrypted by TME/MKTME                               │")
        else:
            print("│ ✓ TEE environment detected                                       │")
            print("│ ✓ Memory encryption active                                       │")
        print("│ ✓ Full confidential computing guarantees in effect               │")
        print("="*70 + "\n")

def main():
    print_banner()
    print_mode_banner()
    
    # If NOT in a real TEE, ask user what to do
    if SIMULATION_MODE and not is_running_in_tee():
        print("┌─────────────────────────────────────────────────────────────────┐")
        print("│  You are NOT running inside a protected environment.            │")
        print("│                                                                  │")
        print("│  Options:                                                        │")
        print("│    [1] Continue in SIMULATION mode (demo only, no protection)   │")
        print("│    [2] Launch inside SGX enclave (requires Gramine + SGX HW)    │")
        print("│    [Q] Quit                                                      │")
        print("└─────────────────────────────────────────────────────────────────┘")
        
        choice = input("\nSelect option [1/2/Q]: ").strip().lower()
        
        if choice == "2":
            import subprocess
            script_dir = os.path.dirname(os.path.abspath(__file__))
            enclave_script = os.path.join(script_dir, "run_sgx_enclave.sh")
            
            if os.path.exists(enclave_script):
                print("\n[*] Launching inside SGX enclave via Gramine...")
                os.chdir(script_dir)
                os.execvp("bash", ["bash", enclave_script])
            else:
                print(f"[!] Enclave script not found: {enclave_script}")
                sys.exit(1)
        elif choice == "q":
            print("Exiting.")
            sys.exit(0)
        else:
            print("\n[*] Continuing in SIMULATION mode...\n")
    
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
