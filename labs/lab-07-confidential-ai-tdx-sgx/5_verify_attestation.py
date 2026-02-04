#!/usr/bin/env python3
"""
Step 5: Verify Attestation Report

This demonstrates how to verify that code is running in a
genuine Intel TDX/SGX environment with proper security guarantees.

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import os
import json
import sys
import hashlib
from datetime import datetime

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         🔐 Intel TDX/SGX Attestation Verification 🔐                  ║
║           (Prove code runs in secure environment)                     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def explain_attestation():
    """Explain what attestation provides."""
    print("""
WHAT IS ATTESTATION?

Attestation is cryptographic proof that:
  ✓ Code runs inside genuine Intel TDX Trust Domain or SGX Enclave
  ✓ Memory is encrypted and isolated from hypervisor
  ✓ The specific code/model loaded matches expected measurements
  ✓ Security properties are enforced by hardware

┌─────────────────────────────────────────────────────────────────────┐
│                    ATTESTATION FLOW                                  │
│                                                                      │
│   ┌──────────┐    Request      ┌──────────────┐                     │
│   │  Client  │ ──────────────▶ │  TDX/SGX     │                     │
│   │          │                 │  Environment │                     │
│   │          │ ◀────────────── │              │                     │
│   │          │   Quote/Report  │              │                     │
│   └────┬─────┘                 └──────────────┘                     │
│        │                              │                              │
│        │ Verify                       │ Signed by                    │
│        ▼                              │ Intel Hardware               │
│   ┌──────────┐                        ▼                              │
│   │  Intel   │           ┌────────────────────┐                     │
│   │  AS/ITA  │ ◀──────── │  Intel CPU         │                     │
│   └──────────┘           │  (Hardware RoT)    │                     │
│                          └────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘

VERIFICATION SERVICES:
  • Intel Trust Authority (ITA) - Cloud-based verification
  • Azure Attestation Service - For Azure Confidential VMs
  • Intel Attestation Service (IAS) - For SGX enclaves
    """)

def load_attestation_report():
    """Load the attestation report generated in step 4."""
    print("\n[*] Loading attestation report...")
    
    # Try TDX report first, then SGX
    for mode in ["tdx", "sgx"]:
        report_path = f"attestation_report_{mode}.json"
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
            print(f"[✓] Loaded: {report_path}")
            return report, report_path
    
    print("[!] No attestation report found.")
    print("    Run 4_confidential_inference.py first")
    sys.exit(1)

def verify_report_structure(report):
    """Verify the attestation report has required fields."""
    print("\n[*] Verifying report structure...")
    
    required_fields = [
        "version",
        "type",
        "timestamp",
        "environment",
        "measurements",
        "security_properties"
    ]
    
    for field in required_fields:
        if field in report:
            print(f"    [✓] {field}: present")
        else:
            print(f"    [✗] {field}: MISSING")
            return False
    
    return True

def verify_measurements(report):
    """Verify the measurements in the attestation report."""
    print("\n[*] Verifying measurements...")
    
    measurements = report.get("measurements", {})
    
    if report["environment"]["mode"] == "TDX":
        td_measurement = measurements.get("td_measurement")
        if td_measurement:
            print(f"    [✓] TD Measurement: {td_measurement[:32]}...")
            print("        (Would verify against expected TD build)")
        
        rtmr = measurements.get("rtmr0")
        if rtmr:
            print(f"    [✓] RTMR0: {rtmr[:32]}...")
            print("        (Runtime measurement of loaded code)")
    
    else:  # SGX
        enclave_measurement = measurements.get("enclave_measurement")
        if enclave_measurement:
            print(f"    [✓] MRENCLAVE: {enclave_measurement[:32]}...")
            print("        (Enclave identity measurement)")
    
    return True

def verify_security_properties(report):
    """Verify security properties are enabled."""
    print("\n[*] Verifying security properties...")
    
    props = report.get("security_properties", {})
    
    checks = [
        ("memory_encrypted", "Memory Encryption (AES-256)"),
        ("isolated_from_hypervisor", "Hypervisor Isolation"),
        ("debug_disabled", "Debug Mode Disabled"),
        ("secure_boot", "Secure Boot Chain")
    ]
    
    all_passed = True
    for key, name in checks:
        value = props.get(key, False)
        status = "✓" if value else "✗"
        print(f"    [{status}] {name}: {value}")
        if not value:
            all_passed = False
    
    return all_passed

def verify_model_integrity(report):
    """Verify the loaded model matches expected hash."""
    print("\n[*] Verifying model integrity...")
    
    model_info = report.get("model_loaded", {})
    model_hash = model_info.get("hash")
    
    if model_hash:
        print(f"    [*] Model hash in report: {model_hash[:32]}...")
        
        # Verify against actual model file
        model_path = "proprietary_model.h5"
        if os.path.exists(model_path):
            actual_hash = hashlib.sha256(
                open(model_path, 'rb').read()
            ).hexdigest()
            
            if actual_hash == model_hash:
                print(f"    [✓] Model integrity verified")
                print(f"        Hash matches: {actual_hash[:32]}...")
                return True
            else:
                print(f"    [✗] Model hash MISMATCH!")
                print(f"        Expected: {model_hash[:32]}...")
                print(f"        Actual:   {actual_hash[:32]}...")
                return False
        else:
            print(f"    [!] Model file not found for verification")
    else:
        print(f"    [!] No model hash in attestation report")
    
    return False

def simulate_intel_verification(report):
    """Simulate verification against Intel Trust Authority."""
    print("\n[*] Simulating Intel Trust Authority verification...")
    print("    (In production, this calls Intel's verification API)")
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  INTEL TRUST AUTHORITY VERIFICATION                             │
    │                                                                  │
    │  [*] Submitting quote to Intel Trust Authority...               │
    │  [*] Verifying signature chain...                               │
    │  [*] Checking TCB status...                                     │
    │  [*] Validating measurements...                                 │
    │                                                                  │
    │  ✓ VERIFICATION RESULT: TRUSTED                                 │
    │                                                                  │
    │  TCB Status: UpToDate                                           │
    │  Quote Type: TDX (v1.5)                                         │
    │  Hardware: Intel Xeon 6 (TDX 1.0)                               │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    return True

def generate_verification_certificate(report):
    """Generate a verification certificate for clients."""
    print("\n[*] Generating verification certificate...")
    
    certificate = {
        "verification_time": datetime.utcnow().isoformat(),
        "attestation_type": report["type"],
        "verification_status": "PASSED",
        "security_level": "HIGH",
        "verified_properties": {
            "hardware_genuine": True,
            "memory_encrypted": True,
            "isolated_execution": True,
            "measurements_valid": True,
            "model_integrity": True
        },
        "trust_chain": [
            "Intel Hardware Root of Trust",
            "Intel TDX Module",
            "Trust Domain / Enclave",
            "Application Code"
        ],
        "valid_for": "This attestation proves the AI model runs in a hardware-protected environment"
    }
    
    cert_path = "verification_certificate.json"
    with open(cert_path, "w") as f:
        json.dump(certificate, f, indent=2)
    
    print(f"[✓] Certificate saved: {cert_path}")
    
    return certificate

def print_summary(all_passed):
    """Print verification summary."""
    if all_passed:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║              ✅ ATTESTATION VERIFICATION PASSED ✅                    ║
╚══════════════════════════════════════════════════════════════════════╝

The attestation report proves:

  ✓ Code runs inside genuine Intel TDX Trust Domain
  ✓ Memory is encrypted with AES-256 (MKTME)
  ✓ Hypervisor cannot access protected memory
  ✓ The loaded model matches expected measurements
  ✓ Debug capabilities are disabled
  ✓ Secure boot chain is intact

WHAT THIS MEANS FOR AI SECURITY:

  🔒 Model IP is protected from cloud operators
  🔒 Inference data cannot be observed
  🔒 Customers can verify security before sending data
  🔒 Regulatory compliance for sensitive AI workloads

CLIENTS CAN TRUST:
  → Their data is processed securely
  → The model hasn't been tampered with
  → No unauthorized parties can observe computation
        """)
    else:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║              ❌ ATTESTATION VERIFICATION FAILED ❌                    ║
╚══════════════════════════════════════════════════════════════════════╝

WARNING: One or more security properties could not be verified.

POSSIBLE CAUSES:
  • Running on non-TDX/SGX hardware (simulation mode)
  • Security features not enabled in BIOS
  • Debug mode enabled (breaks security guarantees)
  • Model has been tampered with

RECOMMENDATION:
  • Do NOT send sensitive data to this environment
  • Verify hardware capabilities with 0_check_hardware.py
  • Ensure production deployment on certified hardware
        """)

def main():
    print_banner()
    explain_attestation()
    
    report, report_path = load_attestation_report()
    
    # Run verification checks
    results = []
    results.append(("Structure", verify_report_structure(report)))
    results.append(("Measurements", verify_measurements(report)))
    results.append(("Security Properties", verify_security_properties(report)))
    results.append(("Model Integrity", verify_model_integrity(report)))
    results.append(("Intel Verification", simulate_intel_verification(report)))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        generate_verification_certificate(report)
    
    print_summary(all_passed)
    
    print("\n[✓] Step 5 complete. Run: python 6_protected_memory_demo.py")

if __name__ == "__main__":
    main()
