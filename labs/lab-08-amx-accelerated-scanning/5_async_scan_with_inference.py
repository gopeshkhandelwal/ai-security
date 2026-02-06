#!/usr/bin/env python3
"""
Step 5: Async Scanning with Concurrent Inference

Demonstrates running security scanning in parallel with model inference,
achieving true non-blocking security checks.

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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import time
import asyncio
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

TEST_MODELS_DIR = Path("test_models")
RESULTS_DIR = Path("scan_results")

# Check for simulation mode from .env (consistent with lab-07)
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "true").lower() == "true"

# Also check hardware status from amx_status.json for additional info
AMX_AVAILABLE = False
try:
    with open(".amx_status.json", "r") as f:
        amx_status = json.load(f)
        AMX_AVAILABLE = amx_status.get("amx_available", False)
except:
    pass

# In hardware mode, AMX must also be available
HARDWARE_MODE = (not SIMULATION_MODE) and AMX_AVAILABLE

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     Async Security Scanning with Concurrent Inference                 ║
║              🔒 Scan + 🧠 Infer = Parallel Execution                  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def print_amx_mode_indicator():
    """Print a clear indicator of AMX hardware status."""
    if HARDWARE_MODE:
        print("\n" + "="*70)
        print("🟢 " + " HARDWARE MODE - Intel AMX Active ".center(66, "=") + " 🟢")
        print("="*70)
        print("│ Async scanning with REAL Intel AMX acceleration                 │")
        print("│ Non-blocking security checks with hardware optimization         │")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("🔶 " + " SIMULATION MODE ".center(66, "=") + " 🔶")
        print("="*70)
        if not AMX_AVAILABLE:
            print("│ Async scanning WITHOUT Intel AMX hardware                       │")
        else:
            print("│ SIMULATION_MODE=true in .env (AMX hardware is available)         │")
        print("│ Demonstrates non-blocking architecture concepts                 │")
        print("│ Set SIMULATION_MODE=false in .env for hardware mode              │")
        print("="*70 + "\n")

# =============================================================================
# ASYNC SECURITY SCANNER
# =============================================================================

class AsyncSecurityScanner:
    """
    Asynchronous security scanner that runs in background
    while inference continues on the main thread.
    """
    
    def __init__(self):
        self.scan_complete = threading.Event()
        self.scan_results = {}
        self.scan_time = 0
    
    def _scan_worker(self, model_paths: list):
        """Worker function for background scanning."""
        start_time = time.time()
        
        for path in model_paths:
            # Simulate scanning
            with open(path, 'rb') as f:
                content = f.read()
            
            # Pattern matching
            suspicious = any(pattern in content for pattern in [
                b'eval(', b'exec(', b'subprocess', b'__reduce__'
            ])
            
            self.scan_results[str(path)] = {
                "status": "SUSPICIOUS" if suspicious else "CLEAN",
                "scanned_at": datetime.utcnow().isoformat()
            }
        
        self.scan_time = time.time() - start_time
        self.scan_complete.set()
    
    def start_async_scan(self, model_dir: Path):
        """Start scanning in background thread."""
        model_files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.h5"))
        
        # Start background scan
        scan_thread = threading.Thread(
            target=self._scan_worker,
            args=(model_files,),
            daemon=True
        )
        scan_thread.start()
        
        return scan_thread
    
    def is_complete(self) -> bool:
        """Check if scan is complete."""
        return self.scan_complete.is_set()
    
    def get_results(self) -> dict:
        """Get scan results (blocks until complete)."""
        self.scan_complete.wait()
        return self.scan_results

# =============================================================================
# MODEL INFERENCE SIMULATOR
# =============================================================================

class InferenceRunner:
    """Simulates model inference running concurrently with scanning."""
    
    def __init__(self):
        self.inference_count = 0
        self.total_inference_time = 0
    
    def run_inference(self, batch_size=10):
        """Run a batch of inferences."""
        start = time.time()
        
        # Simulate inference workload
        input_data = np.random.randn(batch_size, 100)
        
        # Simulate model forward pass
        weights = np.random.randn(100, 50)
        hidden = np.maximum(0, input_data @ weights)  # ReLU
        output = hidden @ np.random.randn(50, 10)
        
        inference_time = time.time() - start
        self.inference_count += batch_size
        self.total_inference_time += inference_time
        
        return output, inference_time

# =============================================================================
# DEMO: CONCURRENT SCAN + INFERENCE
# =============================================================================

def demo_blocking_approach():
    """Demonstrate blocking approach (scan THEN infer)."""
    print("\n" + "="*60)
    print("APPROACH 1: BLOCKING (Scan then Infer)")
    print("="*60)
    
    # Simulate scanning
    print("\n  [1] Scanning models...", flush=True)
    scan_start = time.time()
    time.sleep(2.0)  # Simulate 2s scan
    scan_time = time.time() - scan_start
    print(f"      Scan complete: {scan_time:.2f}s")
    
    # Then inference
    print("  [2] Running inference...", flush=True)
    inference_start = time.time()
    
    runner = InferenceRunner()
    for _ in range(10):
        runner.run_inference(100)
    
    inference_time = time.time() - inference_start
    print(f"      Inference complete: {inference_time:.3f}s ({runner.inference_count} samples)")
    
    total_time = scan_time + inference_time
    print(f"\n  Total time: {total_time:.2f}s")
    print("  ⚠️  Inference was BLOCKED during scanning")
    
    return total_time

def demo_async_approach():
    """Demonstrate non-blocking approach (scan WHILE inferring)."""
    print("\n" + "="*60)
    print("APPROACH 2: ASYNC (Scan while Inferring)")
    print("="*60)
    
    overall_start = time.time()
    
    # Start async scanning
    scanner = AsyncSecurityScanner()
    print("\n  [1] Starting background scan...", flush=True)
    scan_thread = scanner.start_async_scan(TEST_MODELS_DIR)
    
    # Run inference concurrently
    print("  [2] Running inference (concurrent with scan)...", flush=True)
    runner = InferenceRunner()
    
    inference_batches = 0
    while not scanner.is_complete() or inference_batches < 10:
        runner.run_inference(100)
        inference_batches += 1
        
        # Progress indicator
        if inference_batches % 3 == 0:
            scan_status = "🔍 scanning..." if not scanner.is_complete() else "✓ scan done"
            print(f"      Batch {inference_batches}: {runner.inference_count} samples inferred, {scan_status}")
    
    # Ensure scan completes
    scan_thread.join(timeout=5)
    
    total_time = time.time() - overall_start
    
    print(f"\n  Scan time: {scanner.scan_time:.3f}s (background)")
    print(f"  Inference: {runner.inference_count} samples in {runner.total_inference_time:.3f}s")
    print(f"  Total time: {total_time:.2f}s")
    print("  ✓ Inference continued DURING scanning!")
    
    return total_time

def show_architecture():
    """Show async scanning architecture."""
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│              ASYNC SCANNING ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────┘

BLOCKING APPROACH (Traditional):

  Time ────────────────────────────────────────────────────────────▶
  
  Thread 0: [████ SCAN ████████████████][████ INFERENCE ███████████]
                                        ↑
                                 Inference must wait


ASYNC APPROACH (AMX-Optimized):

  Time ────────────────────────────────────────────────────────────▶
  
  Thread 0: [════════════ INFERENCE (main) ════════════════════════]
            ↑ inference starts immediately
            
  Thread 1: [████ SCAN ███]
            │             │
            └─────────────┴──▶ Results ready (callback)
            
  Thread 2: [████ HASH ███]
            
  Thread 3: [██ MLBOM █]


KEY BENEFITS:
┌─────────────────────────────────────────────────────────────────────┐
│  ✓ Inference latency: Unaffected by scanning                        │
│  ✓ Throughput: Higher overall (parallel execution)                  │
│  ✓ Resource usage: Better CPU utilization                          │
│  ✓ Security: Same comprehensive scanning                            │
└─────────────────────────────────────────────────────────────────────┘
    """)

def main():
    print_banner()
    print_amx_mode_indicator()
    
    if not TEST_MODELS_DIR.exists():
        print("[!] Test models not found. Run 1_generate_test_models.py first")
        sys.exit(1)
    
    show_architecture()
    
    # Demo blocking approach
    blocking_time = demo_blocking_approach()
    
    # Demo async approach
    async_time = demo_async_approach()
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"""
    Blocking Approach:  {blocking_time:.2f}s
    Async Approach:     {async_time:.2f}s
    
    Improvement:        {((blocking_time - async_time) / blocking_time * 100):.1f}% faster
    
    Key Insight:
    ────────────
    With async scanning, inference starts IMMEDIATELY while
    security checks run in the background. This is especially
    valuable when:
    
    • Processing inference requests in production
    • Loading models on-demand
    • Running security scans on new model uploads
    • CI/CD pipeline integration
    """)
    
    print("\n[✓] Step 5 complete. Run: python 6_generate_mlbom.py")

if __name__ == "__main__":
    main()
