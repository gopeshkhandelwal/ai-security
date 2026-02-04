#!/usr/bin/env python3
"""
Step 0: Check Hardware Capabilities for Intel TDX/SGX

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import os
import platform
import subprocess
import sys

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║       Intel Confidential Computing Hardware Check                     ║
║                   TDX / SGX Capability Scanner                        ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def check_cpu_info():
    """Check CPU for Intel and relevant features."""
    print("\n[1/5] Checking CPU Information...")
    
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        
        brand = info.get('brand_raw', 'Unknown')
        vendor = info.get('vendor_id_raw', 'Unknown')
        
        print(f"      CPU: {brand}")
        print(f"      Vendor: {vendor}")
        
        is_intel = 'Intel' in vendor or 'Intel' in brand
        is_xeon = 'Xeon' in brand
        
        if is_intel:
            print("      [✓] Intel processor detected")
            if is_xeon:
                print("      [✓] Xeon processor (TDX/SGX capable family)")
            return True
        else:
            print("      [!] Non-Intel processor - TDX/SGX not available")
            return False
            
    except ImportError:
        print("      [!] py-cpuinfo not installed, using fallback...")
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'GenuineIntel' in cpuinfo:
                    print("      [✓] Intel processor detected")
                    return True
                else:
                    print("      [!] Non-Intel processor")
                    return False
        except FileNotFoundError:
            print("      [!] Cannot read CPU info (not Linux)")
            return None

def check_sgx_support():
    """Check for SGX support via CPUID."""
    print("\n[2/5] Checking Intel SGX Support...")
    
    # Check via /dev/sgx_enclave
    sgx_devices = ['/dev/sgx_enclave', '/dev/sgx/enclave', '/dev/isgx']
    
    for device in sgx_devices:
        if os.path.exists(device):
            print(f"      [✓] SGX device found: {device}")
            return True
    
    # Check via kernel module
    try:
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        if 'sgx' in result.stdout.lower() or 'isgx' in result.stdout.lower():
            print("      [✓] SGX kernel module loaded")
            return True
    except Exception:
        pass
    
    # Check CPUID flags
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'sgx' in cpuinfo.lower():
                print("      [✓] SGX CPU flag detected")
                print("      [!] SGX driver may not be loaded")
                return "partial"
    except Exception:
        pass
    
    print("      [!] SGX not detected (may need BIOS enable or driver)")
    print("      [i] To enable: Check BIOS for 'Software Guard Extensions'")
    return False

def check_tdx_support():
    """Check for TDX support."""
    print("\n[3/5] Checking Intel TDX Support...")
    
    # Check for TDX sysfs
    tdx_paths = [
        '/sys/firmware/tdx',
        '/sys/devices/system/cpu/tdx'
    ]
    
    for path in tdx_paths:
        if os.path.exists(path):
            print(f"      [✓] TDX interface found: {path}")
            return True
    
    # Check dmesg for TDX
    try:
        result = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=5)
        if 'tdx' in result.stdout.lower():
            print("      [✓] TDX references found in kernel log")
            return True
    except Exception:
        pass
    
    # Check kernel config
    try:
        result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
        kernel_version = result.stdout.strip()
        major, minor = map(int, kernel_version.split('.')[:2])
        if major >= 6 and minor >= 2:
            print(f"      [✓] Kernel {kernel_version} supports TDX")
            print("      [!] TDX may need to be enabled in BIOS/hypervisor")
        else:
            print(f"      [!] Kernel {kernel_version} - TDX requires kernel 6.2+")
    except Exception:
        pass
    
    print("      [!] TDX not detected (requires Xeon 4th/5th/6th Gen + BIOS enable)")
    return False

def check_tme_support():
    """Check for Total Memory Encryption."""
    print("\n[4/5] Checking Intel TME (Total Memory Encryption)...")
    
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'tme' in cpuinfo.lower():
                print("      [✓] TME CPU flag detected")
                return True
    except Exception:
        pass
    
    # Check via MSR (requires root)
    print("      [!] TME not detected in CPU flags")
    print("      [i] TME provides memory encryption foundation for TDX")
    return False

def check_virtualization():
    """Check virtualization status (for TDX)."""
    print("\n[5/5] Checking Virtualization Environment...")
    
    # Check if we're in a VM
    try:
        result = subprocess.run(['systemd-detect-virt'], capture_output=True, text=True)
        virt = result.stdout.strip()
        if virt and virt != 'none':
            print(f"      [i] Running in virtualized environment: {virt}")
            print("      [i] For TDX: Ensure this is a Confidential VM (TDX guest)")
        else:
            print("      [✓] Running on bare metal")
    except Exception:
        print("      [!] Could not detect virtualization status")

def print_summary(sgx, tdx, is_intel):
    """Print summary and recommendations."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if not is_intel:
        print("""
[!] Non-Intel CPU Detected

    TDX and SGX are Intel-specific technologies. For this lab:
    
    Option 1: Use SIMULATION MODE (default)
              The lab scripts will simulate TDX/SGX behavior
              
    Option 2: Deploy on Cloud with Confidential Computing
              - Azure: DCesv5/ECesv5 VMs (TDX)
              - GCP: C3 Confidential VMs
              
    Option 3: Use Intel DevCloud (free access to Intel hardware)
              https://devcloud.intel.com
        """)
        return
    
    if sgx:
        print("    [✓] SGX: AVAILABLE")
    else:
        print("    [!] SGX: Not enabled (check BIOS)")
    
    if tdx:
        print("    [✓] TDX: AVAILABLE")
    else:
        print("    [!] TDX: Not enabled (requires Xeon 4th Gen+ and BIOS)")
    
    if not sgx and not tdx:
        print("""
    RECOMMENDATIONS:
    
    1. Enable in BIOS:
       - Look for "Software Guard Extensions" (SGX)
       - Look for "Trust Domain Extensions" (TDX)
       - Ensure "Total Memory Encryption" (TME) is enabled
       
    2. For SGX: Install the Intel SGX driver
       https://github.com/intel/linux-sgx
       
    3. For TDX: Use kernel 6.2+ and enable TDX in BIOS
    
    4. Alternative: Run lab in SIMULATION MODE
       The scripts will simulate confidential computing behavior
        """)
    else:
        print("""
    [✓] Your system supports Intel Confidential Computing!
    
    Continue with the lab:
        python 1_train_proprietary_model.py
        """)

def main():
    print_banner()
    
    is_intel = check_cpu_info()
    sgx = check_sgx_support()
    tdx = check_tdx_support()
    tme = check_tme_support()
    check_virtualization()
    
    print_summary(sgx, tdx, is_intel)
    
    # Set environment variable for other scripts
    if sgx or tdx:
        print("\n[i] Set SIMULATION_MODE=false in .env for real hardware")
    else:
        print("\n[i] Lab will run in SIMULATION_MODE (set in .env)")

if __name__ == "__main__":
    main()
