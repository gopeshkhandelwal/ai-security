# Lab 10: Confidential AI with Intel TDX

Demonstrates how Intel TDX protects AI workloads by encrypting VM memory at the hardware level.

## What is TDX?

Intel Trust Domain Extensions (TDX) provides hardware-level memory encryption. A malicious cloud provider or hypervisor cannot read your VM's memory - they only see encrypted garbage.

## Scripts

| Script | Description |
|--------|-------------|
| `1_check_tdx.py` | Check if TDX is enabled on the VM |
| `2_tdx_demo.py` | **Main demo**: Memory attack + attestation |

## Quick Start

```bash
# Check TDX status
python3 1_check_tdx.py

# Run the demo
python3 2_tdx_demo.py
```

## What the Demo Shows

**Part 1: Memory Attack**
- Standard VM: Hypervisor extracts SSN and model weights (plaintext)
- TDX VM: Hypervisor sees encrypted garbage

**Part 2: Attestation**
- TDX can cryptographically prove protection to third parties
- Without TDX, you just have to trust the cloud provider

## Requirements

- Standard VM: Any GCP VM (e.g., e2-standard-4)
- TDX VM: C3 machine type with `--confidential-compute-type=TDX`
