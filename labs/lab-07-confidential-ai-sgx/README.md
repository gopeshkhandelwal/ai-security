# Lab 07: Confidential AI with Intel SGX

[![Intel SGX](https://img.shields.io/badge/Intel-SGX-0071C5.svg)](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html)
[![Gramine](https://img.shields.io/badge/Gramine-LibOS-green.svg)](https://gramine.readthedocs.io/)
[![Xeon](https://img.shields.io/badge/Intel-Xeon-0071C5.svg)](https://www.intel.com/content/www/us/en/products/details/processors/xeon.html)

**Author:** GopeshK | [MIT License](../LICENSE)

> ⚠️ **Educational purposes only.** Do not use for malicious activities.

---

## Overview

This lab demonstrates how **Intel Software Guard Extensions (SGX)** protects AI model inference by running code inside hardware-encrypted enclaves. Unlike TDX (VM-level), SGX provides **application-level** confidential computing.

### The Problem

Traditional deployments expose AI models and inference data to:
- Privileged OS processes with memory access
- System administrators
- Memory scraping attacks
- Compromised kernels

### The Solution: Intel SGX Enclaves

| Feature | Protection |
|---------|------------|
| **Encrypted Memory** | Enclave memory encrypted with CPU-bound keys |
| **Isolation** | Protected from OS/kernel access |
| **Attestation** | Cryptographic proof of enclave integrity |
| **Fine-Grained** | Protect specific code paths (inference) |

---

## SGX vs TDX

| Feature | Intel SGX | Intel TDX |
|---------|-----------|-----------|
| **Isolation Level** | Application enclave | Full VM |
| **Code Changes** | Requires SDK or Gramine | None (lift-and-shift) |
| **Memory Limit** | Up to 512GB EPC | Full VM RAM |
| **Best For** | Fine-grained protection | Cloud AI workloads |
| **Lab** | **This lab (Lab 07)** | Lab 10 (TDX on GCP) |

---

## Lab Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STANDARD EXECUTION                               │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Application Memory                           │ │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │ │
│  │  │   Model     │   │  Inference  │   │   Output    │          │ │
│  │  │   Weights   │   │    Data     │   │   Results   │          │ │
│  │  └─────────────┘   └─────────────┘   └─────────────┘          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ▲                                       │
│                              │ OS/Admin CAN READ!                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Operating System                             │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     SGX ENCLAVE EXECUTION                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Application Memory                           │ │
│  │  ┌────────────────────────────────────────────────────────┐    │ │
│  │  │           🔒 SGX Enclave (Encrypted)                   │    │ │
│  │  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │    │ │
│  │  │  │   Model     │   │  Inference  │   │   Output    │  │    │ │
│  │  │  │   Weights   │   │    Data     │   │   Results   │  │    │ │
│  │  │  └─────────────┘   └─────────────┘   └─────────────┘  │    │ │
│  │  └────────────────────────────────────────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ▲                                       │
│                              │ OS SEES ENCRYPTED MEMORY!            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Operating System                             │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ▲                                       │
│                    ┌─────────┴─────────┐                            │
│                    │   Intel CPU       │                            │
│                    │   SGX Hardware    │                            │
│                    └────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware Requirements

- **Intel CPU with SGX support** (Xeon E3 v5+, Xeon Scalable, or client CPUs)
- SGX enabled in BIOS
- Linux kernel 5.11+ (in-kernel SGX driver)

### Software Requirements

- Python 3.9+
- Gramine LibOS (for running Python in SGX)
- User in `sgx` group

### Check SGX Availability

```bash
# Check for SGX device
ls -la /dev/sgx*

# Expected:
# /dev/sgx_enclave
# /dev/sgx_provision
```

---

## Lab Structure

```
lab-07-confidential-ai-sgx/
├── README.md                      # This file
├── 0_check_hardware.py            # Check SGX hardware
├── 1_train_proprietary_model.py   # Train model
├── 2_victim_inference_server.py   # Unprotected inference (attack demo)
├── 3_attacker_memory_reader.py    # Memory extraction attack
├── 4a_run_sgx_enclave.sh          # Run inference in SGX enclave
├── 4b_confidential_inference.py   # Inference code for enclave
├── gramine_manifest.template      # Gramine configuration
├── requirements.txt               # Python dependencies
└── reset.py                       # Cleanup script
```

---

## Quick Start

### Setup

```bash
cd lab-07-confidential-ai-sgx
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Check hardware
python 0_check_hardware.py
```

### Part A: Attack Demo (Without Protection)

```bash
# Step 1: Train model
python 1_train_proprietary_model.py

# Step 2: Terminal 1 - Run unprotected server
python 2_victim_inference_server.py

# Step 3: Terminal 2 - Attack
sudo .venv/bin/python 3_attacker_memory_reader.py
# Result: Attack SUCCEEDS - weights stolen!
```

### Part B: Defense Demo (With SGX)

```bash
# Step 4: Run inference in SGX enclave
./4a_run_sgx_enclave.sh

# Step 5: Terminal 2 - Try same attack
sudo .venv/bin/python 3_attacker_memory_reader.py
# Result: Attack FAILS - enclave memory is encrypted!
```

---

## Installing Gramine

Gramine is a Library OS that runs unmodified applications inside SGX enclaves.

### Automatic Installation

```bash
# The helper script installs Gramine if missing
./4a_run_sgx_enclave.sh
```

### Manual Installation (Ubuntu 22.04/24.04)

```bash
# Add Gramine repository
sudo curl -fsSLo /usr/share/keyrings/gramine-keyring.gpg \
  https://packages.gramineproject.io/gramine-keyring.gpg

DISTRO="jammy"  # or "noble" for 24.04

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/gramine-keyring.gpg] \
  https://packages.gramineproject.io/ $DISTRO main" \
  | sudo tee /etc/apt/sources.list.d/gramine.list

sudo apt-get update
sudo apt-get install -y gramine

# Verify
gramine-sgx --version
```

---

## Running Python in SGX Enclave

### Using Gramine

```bash
# Generate manifest
gramine-manifest \
  -Dlog_level=error \
  -Darch_libdir=/lib/x86_64-linux-gnu \
  -Dexecdir=/usr/bin \
  gramine_manifest.template > python.manifest

# Sign the enclave
gramine-sgx-sign --manifest python.manifest --output python.manifest.sgx

# Run in enclave
gramine-sgx python 4b_confidential_inference.py
```

### Key Manifest Settings

```toml
# Enclave size (must fit TensorFlow + model)
sgx.enclave_size = "8G"

# Thread limit
sgx.max_threads = 256

# Limit TensorFlow threading
loader.env.TF_NUM_INTRAOP_THREADS = "1"
loader.env.TF_NUM_INTEROP_THREADS = "1"
```

---

## What SGX Protects Against

| Attack Vector | Protection |
|---------------|------------|
| Privileged OS processes | ✅ Enclave memory encrypted |
| System administrator | ✅ Cannot read enclave |
| Memory scraping | ✅ Data encrypted in DRAM |
| Cold boot attacks | ✅ Keys protected in CPU |

### What SGX Does NOT Protect Against

| Threat | Why |
|--------|-----|
| Side-channel attacks | Some attacks possible (timing, cache) |
| Application vulnerabilities | SGX protects memory, not code bugs |
| Denial of service | OS can still terminate enclave |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `/dev/sgx_enclave` not found | Enable SGX in BIOS |
| Permission denied | `sudo usermod -aG sgx $USER` then re-login |
| "No available TCS pages" | Increase `sgx.max_threads` in manifest |
| Python module not found | Add venv to `loader.env.PYTHONPATH` |

---

## MITRE ATLAS Techniques Addressed

| Technique | Name | How SGX Helps |
|-----------|------|---------------|
| [AML.T0044](https://atlas.mitre.org/techniques/AML.T0044) | Full ML Model Access | Enclave encrypts model weights |
| [AML.T0024](https://atlas.mitre.org/techniques/AML.T0024) | Exfiltration via Inference | Inference data protected |
| [AML.T0010](https://atlas.mitre.org/techniques/AML.T0010) | ML Supply Chain | Attestation verifies integrity |

---

## Additional Resources

- [Intel SGX Developer Guide](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html)
- [Gramine Documentation](https://gramine.readthedocs.io/)
- [Lab 10: TDX on Google Cloud](../lab-10-confidential-ai-tdx/) (VM-level protection)

---

## Key Takeaways

> 🔐 **Intel SGX** provides application-level enclaves for fine-grained protection

> 🛡️ **Gramine LibOS** runs unmodified Python/TensorFlow in SGX

> ✅ **Memory extraction attacks** are blocked by hardware encryption

> 🔗 **Remote attestation** proves code runs in a trusted enclave
