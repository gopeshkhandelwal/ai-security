#!/usr/bin/env python3
"""
Step 0: Check TDX Status

Verifies that Intel TDX is active on this VM.
Run this first to confirm you're in a Confidential VM.

Author: GopeshK
License: MIT License
"""

import subprocess
import sys
import os


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                 Intel TDX Status Check                               ║
║           Verify Confidential VM Configuration                       ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def run_command(cmd, shell=True):
    """Run command and return output."""
    try:
        result = subprocess.run(
            cmd, 
            shell=shell, 
            capture_output=True, 
            text=True,
            timeout=10
        )
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return str(e), 1


def check_dmesg_tdx():
    """Check kernel messages for TDX."""
    print("[1/5] Checking kernel messages (dmesg)...")
    
    output, rc = run_command("sudo dmesg 2>/dev/null | grep -i tdx | head -5")
    
    if "guest" in output.lower() or "tdx" in output.lower():
        print(f"  ✅ TDX detected in kernel messages:")
        for line in output.split('\n')[:3]:
            print(f"     {line}")
        return True
    else:
        print("  ❌ No TDX messages found in dmesg")
        return False


def check_cpuinfo():
    """Check CPU flags for TDX."""
    print("\n[2/5] Checking CPU flags...")
    
    output, rc = run_command("grep -o 'tdx[^ ]*' /proc/cpuinfo | head -1")
    
    if output:
        print(f"  ✅ TDX CPU flag found: {output}")
        return True
    else:
        # TDX guest may not show flag, check for other indicators
        output2, _ = run_command("cat /proc/cpuinfo | grep 'model name' | head -1")
        print(f"  ℹ️  CPU: {output2.split(':')[-1].strip() if ':' in output2 else 'Unknown'}")
        print(f"  ⚠️  TDX flag not visible (normal for TDX guest)")
        return None


def check_sys_firmware():
    """Check /sys/firmware for TDX."""
    print("\n[3/5] Checking /sys/firmware...")
    
    if os.path.exists("/sys/firmware/tdx"):
        print("  ✅ /sys/firmware/tdx exists")
        return True
    
    output, rc = run_command("ls /sys/firmware/ 2>/dev/null")
    print(f"  ℹ️  Firmware entries: {output.replace(chr(10), ', ')}")
    return False


def check_confidential_compute():
    """Check GCP metadata for confidential compute."""
    print("\n[4/5] Checking GCP Confidential Compute metadata...")
    
    cmd = """curl -s -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/confidential-instance/enable \
        2>/dev/null"""
    
    output, rc = run_command(cmd)
    
    if output.lower() == "true":
        print("  ✅ GCP Confidential Compute: ENABLED")
        return True
    elif "not found" in output.lower() or rc != 0:
        print("  ℹ️  GCP metadata not available (may not be on GCP)")
        return None
    else:
        print(f"  ❌ GCP Confidential Compute: {output}")
        return False


def check_attestation():
    """Check if TDX attestation is available."""
    print("\n[5/5] Checking TDX attestation device...")
    
    devices = [
        "/dev/tdx_guest",
        "/dev/tdx-guest", 
        "/dev/tdx-attest"
    ]
    
    for dev in devices:
        if os.path.exists(dev):
            print(f"  ✅ Attestation device found: {dev}")
            return True
    
    print("  ⚠️  No TDX attestation device found")
    print("     (Attestation may still work via different interface)")
    return None


def main():
    print_banner()
    
    checks = {
        "Kernel TDX": check_dmesg_tdx(),
        "CPU Flags": check_cpuinfo(),
        "Sys Firmware": check_sys_firmware(),
        "GCP Confidential": check_confidential_compute(),
        "Attestation": check_attestation()
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in checks.values() if v is True)
    warned = sum(1 for v in checks.values() if v is None)
    failed = sum(1 for v in checks.values() if v is False)
    
    for name, result in checks.items():
        if result is True:
            status = "✅ PASS"
        elif result is None:
            status = "⚠️  WARN"
        else:
            status = "❌ FAIL"
        print(f"  {name:.<30} {status}")
    
    print("")
    
    if passed >= 2:
        print("🔒 TDX APPEARS TO BE ACTIVE")
        print("   Memory encryption is protecting this VM")
        print("\n   Proceed with the attack demo:")
        print("   python 1_train_proprietary_model.py")
        return 0
    elif warned > failed:
        print("⚠️  TDX STATUS UNCERTAIN")
        print("   Some indicators present but not confirmed")
        print("   Proceed with caution - attack demo may still work")
        return 0
    else:
        print("❌ TDX NOT DETECTED")
        print("   This VM may not have Confidential Computing enabled")
        print("   Create VM with: --confidential-compute-type=TDX")
        return 1


if __name__ == "__main__":
    sys.exit(main())
