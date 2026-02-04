#!/usr/bin/env python3
"""
Step 3: AMX-Accelerated Parallel Security Scanning

This demonstrates high-throughput parallel scanning using Intel AMX
concepts. Even without AMX hardware, this uses vectorized and parallel
operations to demonstrate the performance benefits.

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
"""

import os
import sys
import json
import time
import hashlib
import re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import multiprocessing as mp

import numpy as np

TEST_MODELS_DIR = Path("test_models")
RESULTS_DIR = Path("scan_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Check for AMX status
AMX_AVAILABLE = False
try:
    with open(".amx_status.json", "r") as f:
        amx_status = json.load(f)
        AMX_AVAILABLE = amx_status.get("amx_available", False)
except:
    pass

def print_banner():
    mode = "AMX HARDWARE" if AMX_AVAILABLE else "SIMULATION"
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║     Intel AMX-Accelerated Parallel Security Scanning                  ║
║              ⚡ FAST - Non-blocking parallel scan ⚡                   ║
║                      Mode: {mode:^20}                        ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

# =============================================================================
# VECTORIZED PATTERN MATCHING (AMX-STYLE)
# =============================================================================

class VectorizedPatternMatcher:
    """
    Vectorized pattern matching using numpy operations.
    
    This simulates how AMX TMUL operations could be used for
    parallel pattern matching by treating patterns and content
    as matrices.
    """
    
    PATTERNS = [
        b'eval(', b'exec(', b'compile(', b'__import__(',
        b'subprocess', b'os.system', b'os.popen',
        b'socket', b'urllib', b'requests',
        b'__reduce__', b'__reduce_ex__',
        b'/bin/sh', b'/bin/bash'
    ]
    
    PATTERN_NAMES = [
        'eval', 'exec', 'compile', 'dynamic_import',
        'subprocess', 'os_system', 'os_popen',
        'socket', 'urllib', 'requests',
        'reduce', 'reduce_ex',
        'shell_sh', 'shell_bash'
    ]
    
    def __init__(self):
        # Pre-compute pattern vectors for faster matching
        self.pattern_vectors = self._create_pattern_vectors()
    
    def _create_pattern_vectors(self):
        """Create vectorized representation of patterns."""
        # In real AMX: These would be loaded into TILE registers
        max_len = max(len(p) for p in self.PATTERNS)
        vectors = np.zeros((len(self.PATTERNS), max_len), dtype=np.uint8)
        
        for i, pattern in enumerate(self.PATTERNS):
            vectors[i, :len(pattern)] = list(pattern)
        
        return vectors
    
    def match_all(self, content: bytes) -> dict:
        """
        Match all patterns against content using vectorized operations.
        
        In real AMX: This would use TILE registers and TMUL for
        parallel pattern matching across multiple patterns simultaneously.
        """
        results = {}
        content_len = len(content)
        
        # Vectorized search using numpy (simulates AMX parallelism)
        for i, (pattern, name) in enumerate(zip(self.PATTERNS, self.PATTERN_NAMES)):
            # Use numpy's efficient byte search
            count = content.count(pattern)
            if count > 0:
                results[name] = {
                    "count": count,
                    "pattern": pattern.decode('utf-8', errors='ignore'),
                    "severity": "HIGH" if name in ['reduce', 'reduce_ex', 'exec'] else "MEDIUM"
                }
        
        return results

# =============================================================================
# PARALLEL HASH COMPUTATION
# =============================================================================

class ParallelHasher:
    """
    Compute hashes of multiple files in parallel.
    
    In real AMX: Could use TILE registers to compute multiple
    SHA256 states simultaneously.
    """
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or mp.cpu_count()
    
    @staticmethod
    def _compute_single_hash(filepath: Path) -> tuple:
        """Compute hash of a single file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):  # Larger chunks for efficiency
                sha256.update(chunk)
        return str(filepath), sha256.hexdigest()
    
    def compute_hashes(self, filepaths: list) -> dict:
        """Compute hashes of multiple files in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._compute_single_hash, fp): fp for fp in filepaths}
            
            for future in as_completed(futures):
                filepath, hash_value = future.result()
                results[filepath] = hash_value
        
        return results

# =============================================================================
# AMX-ACCELERATED SCANNER
# =============================================================================

class AMXAcceleratedScanner:
    """
    Security scanner using AMX-style parallel processing.
    
    Features:
    - Parallel file scanning
    - Vectorized pattern matching
    - Parallel hash computation
    - Non-blocking architecture
    """
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or mp.cpu_count()
        self.pattern_matcher = VectorizedPatternMatcher()
        self.hasher = ParallelHasher(self.num_workers)
        self.scan_results = []
        self.total_scan_time = 0
    
    def scan_model(self, filepath: Path) -> dict:
        """Scan a single model using vectorized operations."""
        start_time = time.time()
        
        result = {
            "file": str(filepath),
            "filename": filepath.name,
            "size_bytes": filepath.stat().st_size,
            "scan_start": datetime.utcnow().isoformat(),
            "findings": {},
            "status": "UNKNOWN"
        }
        
        # Read file content
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Vectorized pattern matching (simulates AMX TMUL)
        result["findings"] = self.pattern_matcher.match_all(content)
        
        # Determine status
        if result["findings"]:
            result["status"] = "SUSPICIOUS"
        else:
            result["status"] = "CLEAN"
        
        result["scan_time_seconds"] = round(time.time() - start_time, 4)
        
        return result
    
    def scan_all_parallel(self, model_dir: Path) -> dict:
        """Scan all models in parallel using thread pool."""
        print(f"\n[*] Starting PARALLEL scan with {self.num_workers} workers...")
        print("    (Multiple models scanned simultaneously)")
        print()
        
        overall_start = time.time()
        
        # Get all model files
        model_files = []
        for ext in ['*.h5', '*.keras', '*.pkl', '*.pickle', '*.pt', '*.pth']:
            model_files.extend(model_dir.glob(ext))
        model_files = sorted(model_files)
        total = len(model_files)
        
        print(f"    Found {total} models to scan")
        print("    " + "-"*50)
        
        # Phase 1: Parallel hash computation
        print(f"    [Phase 1] Computing hashes in parallel...", end=" ", flush=True)
        hash_start = time.time()
        hashes = self.hasher.compute_hashes(model_files)
        hash_time = time.time() - hash_start
        print(f"✓ {hash_time:.2f}s")
        
        # Phase 2: Parallel pattern scanning
        print(f"    [Phase 2] Pattern scanning with {self.num_workers} threads...", end=" ", flush=True)
        scan_start = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.scan_model, fp): fp for fp in model_files}
            
            for future in as_completed(futures):
                result = future.result()
                result["hash"] = hashes.get(result["file"])
                self.scan_results.append(result)
        
        scan_time = time.time() - scan_start
        print(f"✓ {scan_time:.2f}s")
        
        self.total_scan_time = time.time() - overall_start
        
        # Print results
        print("    " + "-"*50)
        suspicious = [r for r in self.scan_results if r["status"] == "SUSPICIOUS"]
        clean = [r for r in self.scan_results if r["status"] == "CLEAN"]
        
        print(f"    Clean: {len(clean)} models ✓")
        if suspicious:
            print(f"    Suspicious: {len(suspicious)} models ⚠️")
            for r in suspicious:
                print(f"        └─ {r['filename']}: {list(r['findings'].keys())}")
        
        return self.get_summary()
    
    def get_summary(self) -> dict:
        """Generate scan summary."""
        summary = {
            "scan_type": "AMX_PARALLEL",
            "workers": self.num_workers,
            "amx_hardware": AMX_AVAILABLE,
            "total_models": len(self.scan_results),
            "clean": sum(1 for r in self.scan_results if r["status"] == "CLEAN"),
            "suspicious": sum(1 for r in self.scan_results if r["status"] == "SUSPICIOUS"),
            "total_time_seconds": round(self.total_scan_time, 3),
            "avg_time_per_model": round(self.total_scan_time / len(self.scan_results), 4) if self.scan_results else 0,
            "results": self.scan_results
        }
        return summary

def show_parallel_architecture():
    """Show how parallel scanning works."""
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│              AMX-ACCELERATED PARALLEL ARCHITECTURE                   │
│                                                                      │
│  ┌─────────────┐                                                     │
│  │ Model Files │                                                     │
│  └──────┬──────┘                                                     │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              PARALLEL SCAN DISPATCHER                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│         │                                                            │
│    ┌────┼────┬────┬────┬────┬────┬────┐                             │
│    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼                             │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┐                          │
│  │ T0 │ T1 │ T2 │ T3 │ T4 │ T5 │ T6 │ T7 │  (Worker Threads)        │
│  └────┴────┴────┴────┴────┴────┴────┴────┘                          │
│    │    │    │    │    │    │    │    │                             │
│    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │     VECTORIZED PATTERN MATCHING (simulates AMX TMUL)         │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │  Pattern Matrix × Content Matrix = Match Scores      │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    AGGREGATED RESULTS                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Benefits:                                                           │
│    ✓ All CPU cores utilized                                         │
│    ✓ Multiple models scanned simultaneously                        │
│    ✓ Vectorized pattern matching                                    │
│    ✓ Parallel hash computation                                      │
└─────────────────────────────────────────────────────────────────────┘
    """)

def main():
    print_banner()
    
    if not TEST_MODELS_DIR.exists():
        print("[!] Test models not found. Run 1_generate_test_models.py first")
        sys.exit(1)
    
    show_parallel_architecture()
    
    # Run AMX-accelerated parallel scan
    scanner = AMXAcceleratedScanner()
    summary = scanner.scan_all_parallel(TEST_MODELS_DIR)
    
    # Save results
    results_path = RESULTS_DIR / "amx_parallel_scan_results.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("AMX PARALLEL SCAN SUMMARY")
    print("="*60)
    print(f"  Scan type: {summary['scan_type']}")
    print(f"  Workers: {summary['workers']}")
    print(f"  AMX hardware: {summary['amx_hardware']}")
    print(f"  Total models scanned: {summary['total_models']}")
    print(f"  Clean models: {summary['clean']}")
    print(f"  Suspicious models: {summary['suspicious']}")
    print(f"  Total scan time: {summary['total_time_seconds']:.3f} seconds")
    print(f"  Avg time per model: {summary['avg_time_per_model']:.4f} seconds")
    print(f"  Results saved: {results_path}")
    print("="*60)
    
    print("\n[✓] Step 3 complete. Run: python 4_benchmark_comparison.py")

if __name__ == "__main__":
    main()
