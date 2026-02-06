````markdown
# Lab 07: Confidential AI with Intel TDX/SGX

[![Intel TDX](https://img.shields.io/badge/Intel-TDX-0071C5.svg)](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/overview.html)
[![Intel SGX](https://img.shields.io/badge/Intel-SGX-0071C5.svg)](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html)
[![Xeon 6](https://img.shields.io/badge/Intel-Xeon%206-0071C5.svg)](https://www.intel.com/content/www/us/en/products/details/processors/xeon.html)

**Author:** GopeshK | [MIT License](../LICENSE)

> ⚠️ **Educational purposes only.** Do not use for malicious activities.

---

## Overview

This lab demonstrates how Intel's hardware-based security features protect AI models and data from privileged attackers—including hypervisors, cloud operators, and compromised OS kernels.

### The Problem

Traditional cloud deployments expose AI models and inference data to:
- Cloud operators with physical access
- Hypervisor-level attackers
- Compromised host operating systems
- Memory scraping attacks

### The Solution: Intel Confidential Computing

| Technology | Protection Level | Use Case |
|------------|------------------|----------|
| **Intel TDX** (Trust Domain Extensions) | VM-level isolation | Confidential VMs for AI workloads |
| **Intel SGX** (Software Guard Extensions) | Application-level enclaves | Protect specific model inference code |

Both provide:
- ✅ **Encrypted memory** - Data encrypted in RAM
- ✅ **Attestation** - Cryptographic proof of secure execution
- ✅ **Isolation** - Protected from hypervisor/OS access

---

## Prerequisites

### Hardware Requirements

- **Intel Xeon 6** (or 4th/5th Gen Xeon Scalable) with TDX/SGX support
- BIOS with TDX/SGX enabled
- For TDX: Linux kernel 6.2+ with TDX support

### Software Requirements

- Python 3.9+
- Docker (for SGX simulation)
- Intel SGX SDK (optional, for real hardware)

### Cloud Options (TDX-enabled VMs)

| Provider | Service | TDX Support |
|----------|---------|-------------|
| Azure | DCesv5/ECesv5 VMs | ✅ |
| Google Cloud | C3 Confidential VMs | ✅ |
| AWS | (Coming) | 🔜 |

---

## Setup

```bash
# Create virtual environment
cd lab-07-confidential-ai-tdx-sgx
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Check hardware capabilities
python 0_check_hardware.py
```

---

## Lab Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ATTACK SCENARIO                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │   Attacker   │───▶│  Hypervisor  │───▶│  Memory Dump Attack  │   │
│  │ (Cloud Admin)│    │   Access     │    │  Extract Model/Data  │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     DEFENSE: Intel TDX/SGX                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Trust Domain (TDX)                         │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │              Encrypted Memory (AES-256)                 │  │   │
│  │  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │  │   │
│  │  │  │   Model     │   │  Inference  │   │   Output    │   │  │   │
│  │  │  │   Weights   │   │    Data     │   │   Results   │   │  │   │
│  │  │  └─────────────┘   └─────────────┘   └─────────────┘   │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ▲                                       │
│                              │ Attestation                           │
│                    ┌─────────┴─────────┐                            │
│                    │  Intel CPU (Xeon 6) │                           │
│                    │   Hardware Root     │                           │
│                    │     of Trust        │                           │
│                    └────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Lab Steps

### Part A: Attack Demo (Without Protection)

```bash
# Step 1: Train a simple model (simulates proprietary AI)
python 1_train_proprietary_model.py

# Step 2: Run inference normally (memory is exposed)
python 2_run_inference.py

# Step 3: Simulate hypervisor-level attack (extract model from memory)
python 3_memory_attack_demo.py
```

### Part B: Defense Demo (With Intel TDX/SGX)

```bash
# Step 4: Check hardware support for TDX/SGX
python 0_check_hardware.py

# Step 5: Run inference in confidential environment
python 4_confidential_inference.py

# Step 6: Verify attestation (prove secure execution)
python 5_verify_attestation.py

# Step 7: Demonstrate memory protection (attack fails!)
python 6_protected_memory_demo.py
```

### Clean Up
```bash
python reset.py
```

---

## Intel TDX vs SGX Comparison

| Feature | Intel TDX | Intel SGX |
|---------|-----------|-----------|
| **Isolation Level** | Full VM | Application enclave |
| **Memory Limit** | Full VM RAM | Up to 512GB EPC |
| **Guest OS Changes** | Minimal | Requires SDK integration |
| **Attestation** | TD-level | Enclave-level |
| **Best For** | Lift-and-shift AI workloads | Fine-grained protection |

---

## Defense Layers

| Layer | Defense | Technology |
|-------|---------|------------|
| 1 | Memory Encryption | Intel TME/MKTME (AES-XTS) |
| 2 | Execution Isolation | TDX Trust Domains / SGX Enclaves |
| 3 | Attestation | Intel TDX/SGX Remote Attestation |
| 4 | Secure Boot | Measured launch with TPM/TXT |

---

## Real-World Deployment Options

### Option 1: Azure Confidential Computing

```bash
# Deploy TDX-enabled VM
az vm create \
  --resource-group myResourceGroup \
  --name myConfidentialAIVM \
  --image Canonical:0001-com-ubuntu-confidential-vm-jammy:22_04-lts-cvm:latest \
  --size Standard_DC4es_v5 \
  --security-type ConfidentialVM \
  --os-disk-security-encryption-type VMGuestStateOnly
```

### Option 2: Google Cloud Confidential Computing

```bash
# Create Confidential VM
gcloud compute instances create confidential-ai-vm \
  --zone=us-central1-a \
  --machine-type=c3-standard-4 \
  --confidential-compute \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud
```

### Option 3: Gramine LibOS (SGX)

```bash
# Run PyTorch model in SGX enclave using Gramine
gramine-sgx python model_inference.py
```

---

## Running Python in a Real SGX Enclave (Gramine)

This lab includes scripts to run Python + TensorFlow inference inside a **real SGX enclave** using Gramine.

### Prerequisites

1. **Intel SGX hardware** - `/dev/sgx_enclave` must exist
2. **SGX driver** - Linux kernel 5.11+ has in-kernel SGX driver
3. **User in `sgx` group** - `sudo usermod -aG sgx $USER` (re-login required)

### Installing Gramine

Gramine is not a pip package—it's a system package that provides a Library OS for running unmodified applications inside SGX enclaves.

#### Automatic Installation (via run_sgx_enclave.sh)

The `run_sgx_enclave.sh` script will automatically install Gramine if not present:

```bash
./run_sgx_enclave.sh
```

#### Manual Installation (Ubuntu 22.04/24.04)

```bash
# 1. Download Gramine GPG key
sudo curl -fsSLo /usr/share/keyrings/gramine-keyring.gpg \
  https://packages.gramineproject.io/gramine-keyring.gpg

# 2. Add Gramine repository
# Use 'jammy' for 22.04, 'noble' for 24.04
# For Ubuntu 25.04 (plucky), use 'noble' packages
DISTRO="noble"  # or "jammy" for 22.04

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/gramine-keyring.gpg] \
  https://packages.gramineproject.io/ $DISTRO main" \
  | sudo tee /etc/apt/sources.list.d/gramine.list

# 3. Install Gramine
sudo apt-get update
sudo apt-get install -y gramine

# 4. Verify installation
gramine-sgx --version
```

#### Behind Corporate Proxy

If you're behind a proxy (e.g., Intel network), pass proxy to sudo:

```bash
# Set proxy for sudo commands
export SUDO_PROXY="http_proxy=$http_proxy https_proxy=$https_proxy no_proxy=$no_proxy"

# Download with proxy
sudo env $SUDO_PROXY curl -fsSLo /usr/share/keyrings/gramine-keyring.gpg \
  https://packages.gramineproject.io/gramine-keyring.gpg

# Install with proxy  
sudo env $SUDO_PROXY apt-get update
sudo env $SUDO_PROXY apt-get install -y gramine
```

### Running the SGX Enclave Demo

```bash
# Option 1: Use the helper script (recommended)
./run_sgx_enclave.sh

# Option 2: Run script directly (will prompt to launch enclave)
python 4_confidential_inference.py
# Select option [2] to launch inside SGX enclave

# Option 3: Manual Gramine commands
gramine-manifest \
  -Dlog_level=error \
  -Darch_libdir=/lib/x86_64-linux-gnu \
  -Dexecdir=/usr/bin \
  gramine_manifest.template > python.manifest

gramine-sgx-sign --manifest python.manifest --output python.manifest.sgx
gramine-sgx python 4_confidential_inference.py
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `/dev/sgx_enclave` not found | Enable SGX in BIOS, install SGX driver |
| Permission denied on SGX device | `sudo usermod -aG sgx $USER` then re-login |
| "No available TCS pages" | Increase `sgx.max_threads` in manifest (256+) |
| Thread creation failed | Add TensorFlow thread limits: `TF_NUM_INTRAOP_THREADS=1` |
| Python module not found | Add venv path to `loader.env.PYTHONPATH` in manifest |
| GPG key verification failed | Use `[trusted=yes]` in apt sources (dev/testing only) |
| Distro not supported | Use `noble` packages for Ubuntu 25.04+ |

### Gramine Manifest Configuration

Key settings in `gramine_manifest.template`:

```toml
# Enclave size (must fit TensorFlow + model)
sgx.enclave_size = "8G"

# Thread limit (TensorFlow needs many threads)
sgx.max_threads = 256

# Environment variables to limit TensorFlow threading
loader.env.TF_NUM_INTRAOP_THREADS = "1"
loader.env.TF_NUM_INTEROP_THREADS = "1"
loader.env.OMP_NUM_THREADS = "1"

# Python venv path
loader.env.PYTHONPATH = "/path/to/.venv/lib/python3.13/site-packages"
```

---

## What This Demonstrates

- **Attack:** Hypervisor/memory-level extraction of AI models
- **Impact:** Proprietary model weights and inference data exposed
- **Defense:** Hardware-enforced encryption and isolation (TDX/SGX)
- **Attestation:** Cryptographic proof that code runs in secure environment

---

## Key Takeaways

> 🔐 **Hardware-based security** provides protection that software alone cannot achieve.

> 🛡️ **Intel TDX** enables confidential VMs—run unmodified AI workloads with encrypted memory.

> 🔒 **Intel SGX** provides fine-grained enclaves for sensitive inference operations.

> ✅ **Remote Attestation** proves to clients that their data is processed in a trusted environment.

---

## Additional Resources

- [Intel TDX Documentation](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/documentation.html)
- [Intel SGX Developer Guide](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html)
- [Gramine LibOS for SGX](https://gramine.readthedocs.io/)
- [Azure Confidential Computing](https://azure.microsoft.com/en-us/solutions/confidential-compute/)
- [Google Cloud Confidential Computing](https://cloud.google.com/confidential-computing)

---

## MITRE ATLAS Techniques Addressed

| Technique | Name | How TDX/SGX Helps |
|-----------|------|-------------------|
| [AML.T0044](https://atlas.mitre.org/techniques/AML.T0044) | Full ML Model Access | Encrypted memory prevents extraction |
| [AML.T0024](https://atlas.mitre.org/techniques/AML.T0024) | Exfiltration via Inference | Isolated execution protects data |
| [AML.T0010](https://atlas.mitre.org/techniques/AML.T0010) | ML Supply Chain | Attestation verifies integrity |

````
