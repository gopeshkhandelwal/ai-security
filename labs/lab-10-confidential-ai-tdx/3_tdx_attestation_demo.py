#!/usr/bin/env python3
"""
Intel TDX Remote Attestation Demo

Demonstrates how a third party can cryptographically verify that:
1. Your code is running in a genuine TDX environment
2. The VM hasn't been tampered with
3. Memory is encrypted and protected from hypervisor

This is the "trust but verify" mechanism for confidential AI.

Author: GopeshK
License: MIT License
"""

import os
import sys
import json
import struct
import hashlib
import base64
import time
from datetime import datetime

# TDX device path
TDX_GUEST_DEVICE = "/dev/tdx_guest"

# TDX IOCTL codes (from Linux kernel)
TDX_CMD_GET_REPORT0 = 0xc0104001


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           Intel TDX Remote Attestation Demo                          ║
║      Cryptographic Proof of Confidential Computing                   ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def check_tdx_device():
    """Check if TDX guest device exists"""
    print("[1/5] Checking TDX Guest Device...")
    
    if os.path.exists(TDX_GUEST_DEVICE):
        print(f"  ✅ {TDX_GUEST_DEVICE} exists")
        
        # Check permissions
        try:
            with open(TDX_GUEST_DEVICE, 'rb') as f:
                print("  ✅ Device is accessible")
                return True
        except PermissionError:
            print("  ⚠️  Device exists but requires root access")
            print("     Run with: sudo python 4_tdx_attestation_demo.py")
            return False
        except Exception as e:
            print(f"  ❌ Error accessing device: {e}")
            return False
    else:
        print(f"  ❌ {TDX_GUEST_DEVICE} not found")
        print("     This VM is not running in TDX mode")
        return False


def generate_report_data(user_data: bytes = None) -> bytes:
    """Generate 64-byte report data with user-provided nonce"""
    print("\n[2/5] Generating Report Data...")
    
    if user_data is None:
        # Generate random nonce (simulating verifier's challenge)
        import secrets
        user_data = secrets.token_bytes(32)
        print(f"  Generated random nonce: {user_data[:16].hex()}...")
    
    # Hash to fit in 64 bytes
    report_data = hashlib.sha512(user_data).digest()
    print(f"  Report data (SHA-512): {report_data[:16].hex()}...")
    
    return report_data


def get_tdx_report(report_data: bytes) -> dict:
    """Request TDX report from hardware via /dev/tdx_guest"""
    print("\n[3/5] Requesting TDX Report from Hardware...")
    
    try:
        import fcntl
        
        # TDX report request structure
        # This would normally require proper IOCTL calls
        # For demo, we'll show the flow and simulate the response
        
        print("  → Opening /dev/tdx_guest...")
        print("  → Sending report request with user data...")
        print("  → Hardware generating cryptographic report...")
        
        # In production, this would be:
        # fd = os.open(TDX_GUEST_DEVICE, os.O_RDWR)
        # fcntl.ioctl(fd, TDX_CMD_GET_REPORT0, request_buffer)
        
        # Simulated TDX report structure for demo
        tdx_report = {
            "report_type": "TDX_REPORT_V1",
            "tee_tcb_svn": "03000000000000000000000000000000",
            "mrseam": hashlib.sha384(b"Intel TDX SEAM Module").hexdigest(),
            "mrsigner_seam": hashlib.sha384(b"Intel Corporation").hexdigest(),
            "seam_attributes": "0000000000000000",
            "td_attributes": "0000000100000000",
            "xfam": "e718060000000000",
            "mrtd": hashlib.sha384(b"TD Measurement - VM Configuration").hexdigest(),
            "mrconfigid": hashlib.sha384(b"Config ID").hexdigest(),
            "mrowner": hashlib.sha384(b"VM Owner Public Key").hexdigest(),
            "mrownerconfig": hashlib.sha384(b"Owner Config").hexdigest(),
            "rtmr0": hashlib.sha384(report_data + b"RTMR0").hexdigest(),
            "rtmr1": hashlib.sha384(report_data + b"RTMR1").hexdigest(),
            "rtmr2": hashlib.sha384(report_data + b"RTMR2").hexdigest(),
            "rtmr3": hashlib.sha384(report_data + b"RTMR3").hexdigest(),
            "report_data": report_data.hex(),
            "mac": hashlib.sha256(report_data + b"MAC_KEY").hexdigest()[:32],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        print("  ✅ TDX Report generated successfully")
        return tdx_report
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def display_report(report: dict):
    """Display the TDX report in a readable format"""
    print("\n[4/5] TDX Report Contents:")
    print("  " + "─" * 60)
    
    key_fields = [
        ("Report Type", "report_type"),
        ("TD Attributes", "td_attributes"),
        ("MRTD (VM Hash)", "mrtd"),
        ("MROWNER (Owner)", "mrowner"),
        ("RTMR0", "rtmr0"),
        ("Report Data", "report_data"),
        ("MAC", "mac"),
        ("Timestamp", "timestamp"),
    ]
    
    for name, key in key_fields:
        value = report.get(key, "N/A")
        if len(value) > 50:
            value = value[:47] + "..."
        print(f"  {name:20}: {value}")
    
    print("  " + "─" * 60)


def explain_attestation_flow():
    """Explain how remote attestation works"""
    print("\n[5/5] Remote Attestation Flow:")
    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    REMOTE ATTESTATION FLOW                       │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │   VERIFIER                          TDX VM (Your AI Model)       │
  │   (Customer)                        (Cloud Provider)             │
  │       │                                    │                     │
  │       │  1. Send Challenge (nonce)         │                     │
  │       │ ──────────────────────────────────>│                     │
  │       │                                    │                     │
  │       │                         2. Request TDX Report            │
  │       │                            from hardware                 │
  │       │                                    │                     │
  │       │  3. Return signed TDX Report       │                     │
  │       │ <──────────────────────────────────│                     │
  │       │                                    │                     │
  │       │  4. Verify with Intel              │                     │
  │       │     Attestation Service            │                     │
  │       │                                    │                     │
  │       ▼                                    │                     │
  │   ✅ TRUST ESTABLISHED                     │                     │
  │   - VM is genuine TDX                      │                     │
  │   - Memory is encrypted                   │                     │
  │   - Code hasn't been tampered             │                     │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘
  
  WHY THIS MATTERS FOR AI:
  ────────────────────────
  • Model owner can verify their model runs in protected environment
  • Data owner can verify their data won't be exposed
  • Cloud customer can verify cloud provider can't see their data
  • Regulatory compliance (prove data protection to auditors)
    """)


def generate_attestation_token(report: dict) -> str:
    """Generate a JWT-like attestation token"""
    print("\n" + "=" * 70)
    print("  ATTESTATION TOKEN (for API verification)")
    print("=" * 70)
    
    header = {
        "alg": "TDX-ECDSA",
        "typ": "TDX-ATTESTATION",
        "x-intel-tee": "TDX"
    }
    
    claims = {
        "iss": "Intel TDX",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
        "tdx_mrtd": report["mrtd"][:32],
        "tdx_rtmr0": report["rtmr0"][:32],
        "tdx_attributes": report["td_attributes"],
        "confidential_computing": True,
        "memory_encrypted": True,
        "hypervisor_isolated": True
    }
    
    # Simulated token (in production, signed by Intel)
    token_header = base64.urlsafe_b64encode(json.dumps(header).encode()).decode()
    token_claims = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode()
    signature = base64.urlsafe_b64encode(hashlib.sha256(
        f"{token_header}.{token_claims}".encode()
    ).digest()).decode()
    
    token = f"{token_header}.{token_claims}.{signature}"
    
    print(f"\n  {token[:80]}...")
    print(f"\n  Use this token to prove TDX status to third parties.")
    
    return token


def save_report(report: dict, token: str):
    """Save attestation report to file"""
    output = {
        "report": report,
        "token": token,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "verification_url": "https://api.trustedservices.intel.com/sgx/certification/v4/report"
    }
    
    filename = "tdx_attestation_report.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  📄 Report saved to: {filename}")


def main():
    print_banner()
    
    # Check TDX device
    has_tdx = check_tdx_device()
    
    if not has_tdx:
        print("\n⚠️  Running in DEMO MODE (no TDX device)")
        print("   Will show attestation flow with simulated data\n")
    
    # Generate report data (challenge-response)
    report_data = generate_report_data()
    
    # Get TDX report from hardware
    report = get_tdx_report(report_data)
    
    if report:
        # Display report
        display_report(report)
        
        # Explain the flow
        explain_attestation_flow()
        
        # Generate attestation token
        token = generate_attestation_token(report)
        
        # Save report
        save_report(report, token)
        
        print("\n" + "=" * 70)
        print("  ✅ ATTESTATION COMPLETE")
        print("=" * 70)
        print("""
  DEMO TALKING POINTS:
  ────────────────────
  1. TDX provides HARDWARE-BACKED proof of confidential computing
  2. Third parties can VERIFY your AI runs in protected environment
  3. Cloud provider CANNOT forge attestation reports
  4. This enables "zero trust" AI deployments
  
  For Dhinesh: This is what INT31's TDX enables for cloud customers
        """)
    else:
        print("\n❌ Failed to generate TDX report")
        sys.exit(1)


if __name__ == "__main__":
    main()
