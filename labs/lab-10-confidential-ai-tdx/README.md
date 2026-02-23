# Lab 10: Confidential AI with Intel TDX

Demonstrates how Intel TDX protects AI workloads by encrypting VM memory at the hardware level.

## What is TDX?

Intel Trust Domain Extensions (TDX) provides hardware-level memory encryption. A malicious cloud provider or hypervisor cannot read your VM's memory - they only see encrypted garbage.

## Scripts

| Script | Description |
|--------|-------------|
| `1_check_tdx.py` | Check if TDX is enabled on the VM |
| `2_memory_comparison_demo.py` | Show memory read attack (TDX vs Standard VM) |
| `3_tdx_attestation_demo.py` | Demonstrate TDX remote attestation |

## Quick Start

```bash
# Check TDX status
python3 1_check_tdx.py

# Run memory comparison demo
python3 2_memory_comparison_demo.py
```

## Requirements

- Standard VM: Any GCP VM (e.g., e2-standard-4)
- TDX VM: C3 machine type with `--confidential-compute-type=TDX`
