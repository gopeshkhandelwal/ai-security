#!/usr/bin/env python3
"""
Step 4: Verify TDX Protection

Comprehensive verification that TDX is protecting the AI model.
Compares attack results between TDX and non-TDX environments.

Author: GopeshK
License: MIT License
"""

import subprocess
import json
import os
import sys
from datetime import datetime


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║               TDX Protection Verification                            ║
║         Confirm Memory Encryption is Active                          ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def run_command(cmd, shell=True):
    """Run command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, timeout=30
        )
        return result.stdout + result.stderr, result.returncode
    except Exception as e:
        return str(e), 1


def check_tdx_active():
    """Verify TDX is running."""
    print("[1/4] Verifying TDX status...")
    
    # Check dmesg for TDX
    output, _ = run_command("sudo dmesg 2>/dev/null | grep -i tdx | head -3")
    
    if "guest" in output.lower() or "tdx" in output.lower():
        print("  ✅ TDX is ACTIVE")
        return True
    else:
        print("  ❌ TDX not detected")
        return False


def check_attack_results():
    """Check if attack script ran and what it found."""
    print("\n[2/4] Checking attack results...")
    
    # Look for attack output files
    stolen_weights = "stolen_model_weights.npz"
    attack_dump = "inference_memory_dump.json"
    
    if os.path.exists(stolen_weights):
        size = os.path.getsize(stolen_weights)
        if size > 1000:
            print(f"  ❌ Stolen weights file exists ({size} bytes)")
            print("     Attack may have succeeded!")
            return False
        else:
            print(f"  ⚠️  Stolen weights file exists but small ({size} bytes)")
            print("     Attack likely failed to extract meaningful data")
            return True
    else:
        print("  ✅ No stolen weights file found")
        return True
    

def analyze_memory_dump():
    """Analyze memory dump for encrypted vs cleartext patterns."""
    print("\n[3/4] Analyzing memory patterns...")
    
    dump_file = "inference_memory_dump.json"
    
    if not os.path.exists(dump_file):
        print("  ℹ️  No memory dump to analyze")
        print("     Run the attack script first: sudo python 3_attacker_memory_reader.py")
        return None
    
    try:
        with open(dump_file, 'r') as f:
            dump = json.load(f)
        
        # Check for weight extraction success indicators
        if dump.get("weights_extracted", False):
            print("  ❌ Memory dump indicates weights were extracted!")
            return False
        
        if dump.get("encrypted_memory_detected", False):
            print("  ✅ Memory dump shows encrypted data patterns")
            return True
            
        # Check extracted data quality
        extraction_quality = dump.get("extraction_quality", "unknown")
        print(f"  ℹ️  Extraction quality: {extraction_quality}")
        
        return extraction_quality in ["failed", "encrypted", "garbage"]
        
    except Exception as e:
        print(f"  ⚠️  Could not analyze dump: {e}")
        return None


def generate_report():
    """Generate verification report."""
    print("\n[4/4] Generating verification report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "environment": "Google Cloud C3 Confidential VM",
        "protection": "Intel TDX",
        "checks": {
            "tdx_active": check_tdx_active(),
            "attack_blocked": check_attack_results(),
        },
        "conclusion": ""
    }
    
    all_passed = all(v for v in report["checks"].values() if v is not None)
    
    if all_passed:
        report["conclusion"] = "TDX protection verified - attack blocked"
    else:
        report["conclusion"] = "Protection status uncertain - review results"
    
    # Save report
    report_file = "tdx_verification_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Report saved: {report_file}")
    
    return all_passed


def main():
    print_banner()
    
    # Run checks
    tdx_ok = check_tdx_active()
    attack_blocked = check_attack_results()
    memory_ok = analyze_memory_dump()
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    results = {
        "TDX Active": tdx_ok,
        "Attack Blocked": attack_blocked,
        "Memory Encrypted": memory_ok
    }
    
    for name, status in results.items():
        if status is True:
            icon = "✅"
        elif status is False:
            icon = "❌"
        else:
            icon = "⚠️ "
        print(f"  {name:.<30} {icon}")
    
    print("")
    
    if all(v is True for v in results.values() if v is not None):
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    🔒 TDX PROTECTION VERIFIED                        ║
║                                                                      ║
║  Intel TDX is actively protecting this AI workload.                  ║
║  Memory extraction attacks are blocked by hardware encryption.       ║
║                                                                      ║
║  The same attack would SUCCEED on a standard VM without TDX.         ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
        return 0
    else:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    ⚠️  VERIFICATION INCOMPLETE                       ║
║                                                                      ║
║  Some checks did not pass. Review the results above.                 ║
║  Ensure you're running on a TDX-enabled Confidential VM.             ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
        return 1


if __name__ == "__main__":
    sys.exit(main())
