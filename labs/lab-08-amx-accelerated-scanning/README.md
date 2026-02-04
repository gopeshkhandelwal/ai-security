````markdown
# Lab 08: Intel AMX-Accelerated Model Security Scanning

[![Intel AMX](https://img.shields.io/badge/Intel-AMX-0071C5.svg)](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html)
[![Xeon 6](https://img.shields.io/badge/Intel-Xeon%206-0071C5.svg)](https://www.intel.com/content/www/us/en/products/details/processors/xeon.html)
[![MITRE ATLAS](https://img.shields.io/badge/ATLAS-AML.T0010-red.svg)](https://atlas.mitre.org/techniques/AML.T0010)

**Author:** GopeshK | [MIT License](../LICENSE)

> ⚠️ **Educational purposes only.** Do not use for malicious activities.

---

## Overview

This lab demonstrates how to use **Intel AMX (Advanced Matrix Extensions)** to accelerate security scanning of ML models without blocking inference. Intel AMX provides high-throughput matrix operations that can be leveraged for parallel security analysis.

### The Problem

Security scanning of ML models is critical but often creates bottlenecks:
- **ModelScan** needs to analyze model files for malicious code
- **PickleScan** checks for pickle deserialization attacks
- **SBOM/MLBOM** generation requires parsing dependencies
- Scanning blocks inference pipeline during model loading
- Large models (GB+) take significant time to scan

### The Solution: Intel AMX-Accelerated Parallel Scanning

| Feature | Without AMX | With AMX |
|---------|-------------|----------|
| Scanning Method | Sequential | Parallel batch processing |
| Pattern Matching | Single-threaded | Vectorized with TMUL |
| Hash Computation | Serial | Parallel multi-file |
| Impact on Inference | Blocking | Non-blocking (async) |

---

## Prerequisites

### Hardware Requirements

- **Intel Xeon 6** (or 4th/5th Gen Xeon Scalable) with AMX support
- AMX enabled in BIOS (usually default on supported CPUs)

### Software Requirements

- Python 3.9+
- Linux kernel 5.16+ (AMX support)

---

## Setup

```bash
# Create virtual environment
cd lab-08-amx-accelerated-scanning
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Check AMX capabilities
python 0_check_amx_support.py
```

---

## Lab Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL SCANNING (BLOCKING)                   │
│                                                                      │
│  Model Upload → [Wait] Scan Model → [Wait] Scan Deps → Load Model   │
│                    ↓                     ↓                           │
│              Inference BLOCKED      Inference BLOCKED                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│              AMX-ACCELERATED PARALLEL SCANNING                       │
│                                                                      │
│  Model Upload ──┬──→ AMX Parallel Scan ──→ Load Model               │
│                 │         ↑                                          │
│                 │    ┌────┴────┐                                     │
│                 │    │ TILE 0  │ Pattern Match (ModelScan)          │
│                 │    │ TILE 1  │ Pickle Analysis                    │
│                 │    │ TILE 2  │ Hash Verification                  │
│                 │    │ TILE 3  │ SBOM Generation                    │
│                 │    └─────────┘                                     │
│                 │         │                                          │
│                 └─────────┴──→ Results (Non-blocking)               │
│                                                                      │
│  Inference runs in PARALLEL on other CPU cores                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Lab Steps

### Part A: Problem Demo (Sequential Scanning)

```bash
# Step 1: Generate sample models for scanning
python 1_generate_test_models.py

# Step 2: Run traditional sequential scanning (slow)
python 2_sequential_scan.py
```

### Part B: Solution Demo (AMX-Accelerated Scanning)

```bash
# Step 3: Check Intel AMX hardware support
python 0_check_amx_support.py

# Step 4: Run AMX-accelerated parallel scanning (fast!)
python 3_amx_parallel_scan.py

# Step 5: Compare performance
python 4_benchmark_comparison.py

# Step 6: Non-blocking scan with inference
python 5_async_scan_with_inference.py
```

### Part C: MLBOM Generation

```bash
# Step 7: Generate ML Bill of Materials with AMX acceleration
python 6_generate_mlbom.py
```

### Clean Up
```bash
python reset.py
```

---

## Intel AMX Overview

### What is AMX?

Intel **Advanced Matrix Extensions (AMX)** introduces new matrix operations:

| Component | Description |
|-----------|-------------|
| **TILES** | 8 tile registers (1KB each) for matrix data |
| **TMUL** | Tile matrix multiply unit |
| **TILECFG** | Tile configuration instructions |

### AMX for Security Scanning

While AMX is designed for ML inference, we leverage it for security:

```
┌─────────────────────────────────────────────────────────────────┐
│                AMX TILE USAGE FOR SCANNING                       │
│                                                                  │
│  TILE 0-1: Pattern matching matrices                            │
│            (suspicious code patterns as vectors)                 │
│                                                                  │
│  TILE 2-3: File content chunks                                  │
│            (model file bytes as matrices)                        │
│                                                                  │
│  TILE 4-5: Hash state matrices                                  │
│            (parallel SHA256 computation)                         │
│                                                                  │
│  TILE 6-7: Results aggregation                                  │
│            (match scores, threat levels)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Security Scanning Layers

| Layer | Tool/Method | What It Detects |
|-------|-------------|-----------------|
| 1 | **ModelScan** | Malicious Lambda layers, unsafe ops |
| 2 | **PickleScan** | Pickle deserialization attacks |
| 3 | **Pattern Matching** | Suspicious code patterns (exec, eval, etc.) |
| 4 | **Hash Verification** | Model tampering |
| 5 | **SBOM/MLBOM** | Dependency vulnerabilities |

---

## Performance Benefits

### Benchmark Results (Simulated)

| Metric | Sequential | AMX Parallel | Speedup |
|--------|------------|--------------|---------|
| Single model scan | 2.5s | 0.4s | **6.25x** |
| Batch (10 models) | 25s | 2.1s | **11.9x** |
| MLBOM generation | 5.2s | 0.8s | **6.5x** |
| Inference impact | 100% blocked | <5% overhead | **20x** |

---

## What This Demonstrates

- **Problem:** Security scanning blocks inference and slows deployment
- **Solution:** Intel AMX enables parallel, vectorized scanning
- **Benefits:** 
  - Faster model onboarding
  - Non-blocking inference
  - Comprehensive security without performance penalty

---

## MLBOM (ML Bill of Materials)

This lab also demonstrates **MLBOM** generation:

```json
{
  "model_name": "my-model",
  "version": "1.0.0",
  "format": "tensorflow",
  "hash": "sha256:abc123...",
  "components": [
    {"name": "tensorflow", "version": "2.20.0"},
    {"name": "numpy", "version": "2.4.2"}
  ],
  "security_scan": {
    "modelscan": "PASS",
    "picklescan": "PASS",
    "pattern_match": "PASS"
  },
  "vulnerabilities": []
}
```

---

## Key Takeaways

> ⚡ **Intel AMX** accelerates matrix operations 8x+ compared to AVX-512

> 🔒 **Parallel scanning** enables security without sacrificing performance

> 📋 **MLBOM** provides supply chain transparency for ML models

> 🚀 **Non-blocking** scanning allows inference during security checks

---

## Additional Resources

- [Intel AMX Overview](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html)
- [ModelScan by Protect AI](https://github.com/protectai/modelscan)
- [Fickling (Pickle Analyzer)](https://github.com/trailofbits/fickling)
- [CycloneDX SBOM Standard](https://cyclonedx.org/)
- [OWASP ML Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)

---

## MITRE ATLAS Techniques Addressed

| Technique | Name | How AMX Scanning Helps |
|-----------|------|------------------------|
| [AML.T0010](https://atlas.mitre.org/techniques/AML.T0010) | ML Supply Chain Compromise | Fast pre-load scanning |
| [AML.T0011](https://atlas.mitre.org/techniques/AML.T0011) | Backdoor ML Model | Pattern detection at scale |
| [AML.T0047](https://atlas.mitre.org/techniques/AML.T0047) | ML Artifact Collection | MLBOM tracks provenance |

````
