#!/usr/bin/env python3
"""
Step 0: Check Intel AMX Hardware Support

Intel AMX (Advanced Matrix Extensions) provides hardware acceleration
for matrix operations. This script checks if your CPU supports AMX.

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
"""

import os

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import subprocess
import struct

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         Intel AMX (Advanced Matrix Extensions) Check                  ║
║              Hardware Capability Scanner                              ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def check_cpu_info():
    """Check CPU for Intel AMX support."""
    print("\n[1/4] Checking CPU Information...")
    
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        
        brand = info.get('brand_raw', 'Unknown')
        vendor = info.get('vendor_id_raw', 'Unknown')
        flags = info.get('flags', [])
        
        print(f"      CPU: {brand}")
        print(f"      Vendor: {vendor}")
        
        is_intel = 'Intel' in vendor or 'Intel' in brand
        
        if is_intel:
            print("      [✓] Intel processor detected")
            
            # Check for Xeon 4th Gen+ (Sapphire Rapids and later)
            if 'Xeon' in brand:
                print("      [✓] Xeon processor (AMX-capable family)")
            
            return True, flags
        else:
            print("      [!] Non-Intel processor - AMX not available")
            return False, []
            
    except ImportError:
        print("      [!] py-cpuinfo not installed, using fallback...")
        return check_cpuinfo_fallback()

def check_cpuinfo_fallback():
    """Fallback method to check CPU flags."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            
        if 'GenuineIntel' in cpuinfo:
            print("      [✓] Intel processor detected")
            
            # Extract flags
            for line in cpuinfo.split('\n'):
                if line.startswith('flags'):
                    flags = line.split(':')[1].strip().split()
                    return True, flags
            return True, []
        else:
            print("      [!] Non-Intel processor")
            return False, []
    except FileNotFoundError:
        print("      [!] Cannot read CPU info")
        return False, []

def check_amx_flags(flags):
    """Check for AMX-specific CPU flags."""
    print("\n[2/4] Checking AMX CPU Flags...")
    
    amx_flags = {
        'amx_tile': 'AMX Tile (TILE registers)',
        'amx_int8': 'AMX INT8 (int8 matrix multiply)',
        'amx_bf16': 'AMX BF16 (bfloat16 matrix multiply)',
        'amx_fp16': 'AMX FP16 (float16 matrix multiply)',  # Xeon 6
    }
    
    found_amx = False
    for flag, description in amx_flags.items():
        if flag in flags:
            print(f"      [✓] {flag}: {description}")
            found_amx = True
        else:
            print(f"      [ ] {flag}: Not detected")
    
    # Also check for related features
    related_flags = {
        'avx512f': 'AVX-512 Foundation',
        'avx512_vnni': 'AVX-512 VNNI (Vector Neural Network Instructions)',
        'avx512_bf16': 'AVX-512 BF16',
    }
    
    print("\n      Related vector extensions:")
    for flag, description in related_flags.items():
        if flag in flags:
            print(f"      [✓] {flag}: {description}")
    
    return found_amx

def check_kernel_support():
    """Check if kernel supports AMX."""
    print("\n[3/4] Checking Kernel AMX Support...")
    
    # Check kernel version (AMX support added in 5.16)
    try:
        result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
        kernel_version = result.stdout.strip()
        parts = kernel_version.split('.')
        major, minor = int(parts[0]), int(parts[1])
        
        print(f"      Kernel version: {kernel_version}")
        
        if major > 5 or (major == 5 and minor >= 16):
            print("      [✓] Kernel supports AMX (5.16+)")
            kernel_ok = True
        else:
            print("      [!] Kernel too old for AMX (need 5.16+)")
            kernel_ok = False
    except Exception as e:
        print(f"      [!] Could not check kernel: {e}")
        kernel_ok = False
    
    # Check for AMX in /proc/cpuinfo flags (kernel-enabled)
    try:
        result = subprocess.run(
            ['grep', '-c', 'amx', '/proc/cpuinfo'],
            capture_output=True, text=True
        )
        if int(result.stdout.strip()) > 0:
            print("      [✓] AMX flags visible in /proc/cpuinfo")
            return True
    except Exception:
        pass
    
    return kernel_ok

def check_amx_runtime():
    """Check if AMX can be used at runtime."""
    print("\n[4/4] Checking AMX Runtime Availability...")
    
    # Try to detect if AMX is usable
    try:
        # Check xstate for AMX tile state
        xstate_path = '/proc/self/arch_state'
        if os.path.exists(xstate_path):
            print("      [✓] AMX state management available")
    except Exception:
        pass
    
    # Check if numpy/tensorflow can use AMX
    try:
        import numpy as np
        print(f"      NumPy version: {np.__version__}")
        print("      [✓] NumPy available for vectorized operations")
    except ImportError:
        print("      [!] NumPy not installed")
    
    # Check for Intel optimized libraries
    try:
        result = subprocess.run(
            ['python', '-c', 'import numpy; print(numpy.show_config())'],
            capture_output=True, text=True
        )
        if 'mkl' in result.stdout.lower() or 'mkl' in result.stderr.lower():
            print("      [✓] Intel MKL detected (AMX-optimized)")
    except Exception:
        pass
    
    return True

def print_summary(has_amx, kernel_ok):
    """Print summary and recommendations."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if has_amx and kernel_ok:
        print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  🟢 HARDWARE MODE: Intel AMX IS AVAILABLE                      ║
    ╚════════════════════════════════════════════════════════════════╝
    
    [✓] Intel AMX: AVAILABLE
    
    Your system supports Intel AMX acceleration!
    
    Benefits for security scanning:
    ├─ 8x faster matrix operations with TMUL
    ├─ Parallel pattern matching across TILE registers  
    ├─ Vectorized hash computation
    └─ Non-blocking scan while inference runs
    
    All lab scripts will run in HARDWARE MODE.
    
    Continue with the lab:
        python 1_generate_test_models.py
        """)
    elif has_amx and not kernel_ok:
        print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  🔶 SIMULATION MODE: Kernel upgrade needed                     ║
    ╚════════════════════════════════════════════════════════════════╝
    
    [!] Intel AMX: HARDWARE PRESENT, KERNEL UPGRADE NEEDED
    
    Your CPU supports AMX but kernel needs upgrade:
    
    1. Upgrade to kernel 5.16 or later
    2. Ensure AMX is enabled in BIOS
    
    The lab will run in SIMULATION mode.
        """)
    else:
        print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  🔶 SIMULATION MODE: Intel AMX NOT AVAILABLE                   ║
    ╚════════════════════════════════════════════════════════════════╝
    
    [!] Intel AMX: NOT AVAILABLE
    
    Your system does not support Intel AMX. Requirements:
    
    1. Intel Xeon 4th Gen (Sapphire Rapids) or later
    2. Intel Core 12th Gen (Alder Lake) or later
    3. Linux kernel 5.16+
    4. AMX enabled in BIOS
    
    The lab will run in SIMULATION mode, demonstrating
    the concepts with standard vectorized operations.
        """)
    
    print("""
    AMX MODE INFO:
    ───────────────
    Even without AMX hardware, this lab demonstrates:
    ├─ Parallel scanning architecture
    ├─ Async non-blocking scanning
    ├─ MLBOM generation
    └─ Performance benefits of vectorized operations
    """)

def main():
    print_banner()
    
    is_intel, flags = check_cpu_info()
    has_amx = check_amx_flags(flags) if is_intel else False
    kernel_ok = check_kernel_support()
    check_amx_runtime()
    
    print_summary(has_amx, kernel_ok)
    
    # Save AMX status for other scripts
    amx_status = {
        "amx_available": has_amx and kernel_ok,
        "simulation_mode": not (has_amx and kernel_ok),
        "cpu_flags": list(flags) if flags else []
    }
    
    import json
    with open(".amx_status.json", "w") as f:
        json.dump(amx_status, f, indent=2)
    
    print("[i] AMX status saved to .amx_status.json")

if __name__ == "__main__":
    main()
