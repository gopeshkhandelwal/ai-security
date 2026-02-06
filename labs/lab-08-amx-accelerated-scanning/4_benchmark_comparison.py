#!/usr/bin/env python3
"""
Step 4: Benchmark Comparison - Sequential vs AMX Parallel

Compare the performance of sequential and parallel scanning approaches.

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

import sys
import json
from pathlib import Path

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
║         Performance Comparison: Sequential vs AMX Parallel            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def print_amx_mode_indicator():
    """Print a clear indicator of AMX hardware status."""
    if HARDWARE_MODE:
        print("\n" + "="*70)
        print("🟢 " + " HARDWARE MODE - Intel AMX Active ".center(66, "=") + " 🟢")
        print("="*70)
        print("│ Benchmark results show REAL Intel AMX acceleration               │")
        print("│ Performance numbers reflect actual hardware capabilities          │")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("🔶 " + " SIMULATION MODE ".center(66, "=") + " 🔶")
        print("="*70)
        if not AMX_AVAILABLE:
            print("│ Benchmark running WITHOUT Intel AMX hardware                     │")
        else:
            print("│ SIMULATION_MODE=true in .env (AMX hardware is available)         │")
        print("│ Results show vectorized parallel scanning (simulated AMX)        │")
        print("│ Set SIMULATION_MODE=false in .env for hardware mode              │")
        print("="*70 + "\n")

def load_results():
    """Load scan results from both methods."""
    sequential_path = RESULTS_DIR / "sequential_scan_results.json"
    parallel_path = RESULTS_DIR / "amx_parallel_scan_results.json"
    
    if not sequential_path.exists():
        print("[!] Sequential scan results not found. Run 2_sequential_scan.py first")
        sys.exit(1)
    
    if not parallel_path.exists():
        print("[!] Parallel scan results not found. Run 3_amx_parallel_scan.py first")
        sys.exit(1)
    
    with open(sequential_path, 'r') as f:
        sequential = json.load(f)
    
    with open(parallel_path, 'r') as f:
        parallel = json.load(f)
    
    return sequential, parallel

def calculate_speedup(sequential, parallel):
    """Calculate speedup metrics."""
    seq_time = sequential["total_time_seconds"]
    par_time = parallel["total_time_seconds"]
    
    speedup = seq_time / par_time if par_time > 0 else 0
    time_saved = seq_time - par_time
    percent_faster = ((seq_time - par_time) / seq_time) * 100 if seq_time > 0 else 0
    
    return {
        "sequential_time": seq_time,
        "parallel_time": par_time,
        "speedup": round(speedup, 2),
        "time_saved": round(time_saved, 2),
        "percent_faster": round(percent_faster, 1)
    }

def print_comparison(sequential, parallel, speedup):
    """Print detailed comparison."""
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE COMPARISON                            │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # Table header
    print(f"{'Metric':<30} {'Sequential':<20} {'AMX Parallel':<20}")
    print("-" * 70)
    
    # Data rows
    metrics = [
        ("Scan Type", sequential["scan_type"], parallel["scan_type"]),
        ("Total Models", str(sequential["total_models"]), str(parallel["total_models"])),
        ("Workers", "1", str(parallel.get("workers", "N/A"))),
        ("Total Time (seconds)", f"{sequential['total_time_seconds']:.2f}s", f"{parallel['total_time_seconds']:.3f}s"),
        ("Avg Time/Model", f"{sequential['avg_time_per_model']:.3f}s", f"{parallel['avg_time_per_model']:.4f}s"),
        ("Clean Models", str(sequential["clean"]), str(parallel["clean"])),
        ("Suspicious Models", str(sequential["suspicious"]), str(parallel["suspicious"])),
    ]
    
    for metric, seq_val, par_val in metrics:
        print(f"{metric:<30} {seq_val:<20} {par_val:<20}")
    
    print("-" * 70)
    
    # Speedup summary
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                       🚀 SPEEDUP SUMMARY 🚀                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   Sequential Time:    {speedup['sequential_time']:>8.2f} seconds                           ║
║   Parallel Time:      {speedup['parallel_time']:>8.3f} seconds                           ║
║                       ─────────────────                              ║
║   Speedup:            {speedup['speedup']:>8.2f}x faster                            ║
║   Time Saved:         {speedup['time_saved']:>8.2f} seconds                           ║
║   Improvement:        {speedup['percent_faster']:>8.1f}%                                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def print_visual_comparison(speedup):
    """Print visual bar chart comparison."""
    
    seq_time = speedup['sequential_time']
    par_time = speedup['parallel_time']
    
    # Normalize to 50 chars max
    max_time = max(seq_time, par_time)
    seq_bar = int((seq_time / max_time) * 50)
    par_bar = int((par_time / max_time) * 50)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    VISUAL COMPARISON                                 │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print(f"  Sequential: {'█' * seq_bar}{'░' * (50 - seq_bar)} {seq_time:.2f}s")
    print(f"  Parallel:   {'█' * par_bar}{'░' * (50 - par_bar)} {par_time:.3f}s")
    print()
    
    # Show speedup visually
    speedup_factor = speedup['speedup']
    if speedup_factor >= 1:
        print(f"  Speedup:    {'⚡' * min(int(speedup_factor), 20)} {speedup_factor:.1f}x faster")

def print_inference_impact():
    """Show impact on inference pipeline."""
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│              INFERENCE PIPELINE IMPACT                               │
└─────────────────────────────────────────────────────────────────────┘

SEQUENTIAL SCANNING (blocks inference):

  Time ────────────────────────────────────────────────────────────▶
  
  SCAN:      [████████████ Scan Model 1-15 █████████████████]
  INFERENCE: [╳ BLOCKED ╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳]
  
  → Inference WAITS until all scanning completes
  → High latency for model deployment


AMX PARALLEL SCANNING (non-blocking):

  Time ────────────────────────────────────────────────────────────▶
  
  SCAN:      [█████]  ← Completed quickly
  INFERENCE: [═══════════════════════════════════════════════]
              ↑
              Inference can start almost immediately
  
  → Minimal impact on inference latency
  → Models deployed faster
  → Better resource utilization

┌────────────────────────────────────────────────────────────────────┐
│  With Intel AMX hardware, additional benefits include:              │
│                                                                     │
│  • TILE registers for parallel matrix operations                   │
│  • TMUL unit for accelerated pattern matching                      │
│  • 8+ parallel operations per cycle                                │
│  • Hardware-level acceleration without software overhead           │
└────────────────────────────────────────────────────────────────────┘
    """)

def print_production_benefits():
    """Show production deployment benefits."""
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│              PRODUCTION DEPLOYMENT BENEFITS                          │
└─────────────────────────────────────────────────────────────────────┘

With AMX-Accelerated Parallel Scanning:

┌──────────────────────┬────────────────────┬─────────────────────────┐
│  Metric              │  Without AMX       │  With AMX Parallel      │
├──────────────────────┼────────────────────┼─────────────────────────┤
│  Model onboard time  │  Minutes           │  Seconds                │
│  Inference blocking  │  100% during scan  │  < 5%                   │
│  CPU utilization     │  ~12% (1 core)     │  ~100% (all cores)      │
│  Scan throughput     │  ~3 models/min     │  ~30+ models/min        │
│  Security coverage   │  Same              │  Same                   │
└──────────────────────┴────────────────────┴─────────────────────────┘

USE CASES:

  ✓ CI/CD Pipeline     - Fast pre-deployment security checks
  ✓ Model Registry     - Scan on upload without delays
  ✓ Edge Deployment    - Quick validation before loading
  ✓ Multi-tenant       - Scan multiple customer models in parallel
    """)

def main():
    print_banner()
    print_amx_mode_indicator()
    
    # Load results
    sequential, parallel = load_results()
    
    # Calculate speedup
    speedup = calculate_speedup(sequential, parallel)
    
    # Print comparisons
    print_comparison(sequential, parallel, speedup)
    print_visual_comparison(speedup)
    print_inference_impact()
    print_production_benefits()
    
    # Save comparison
    comparison = {
        "sequential": {
            "time": sequential["total_time_seconds"],
            "models": sequential["total_models"]
        },
        "parallel": {
            "time": parallel["total_time_seconds"],
            "workers": parallel.get("workers"),
            "models": parallel["total_models"]
        },
        "speedup": speedup
    }
    
    comparison_path = RESULTS_DIR / "benchmark_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n[✓] Comparison saved: {comparison_path}")
    print("\n[✓] Step 4 complete. Run: python 5_async_scan_with_inference.py")

if __name__ == "__main__":
    main()
