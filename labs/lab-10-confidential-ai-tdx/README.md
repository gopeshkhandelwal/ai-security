# Lab 10: Confidential AI with Intel TDX on Google Cloud

[![Intel TDX](https://img.shields.io/badge/Intel-TDX-0071C5.svg)](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/overview.html)
[![Google Cloud](https://img.shields.io/badge/GCP-Confidential%20VM-4285F4.svg)](https://cloud.google.com/confidential-computing)
[![Xeon 6](https://img.shields.io/badge/Intel-Xeon%206-0071C5.svg)](https://www.intel.com/content/www/us/en/products/details/processors/xeon.html)

**Author:** GopeshK | [MIT License](../LICENSE)

> ⚠️ **Educational purposes only.** Do not use for malicious activities.

---

## Overview

This lab demonstrates how **Intel Trust Domain Extensions (TDX)** protects AI models and inference data from privileged attackers—including hypervisors, cloud operators, and compromised OS kernels—using **Google Cloud C3 Confidential VMs**.

### The Problem

Traditional cloud deployments expose AI models and inference data to:
- Cloud operators with physical access
- Hypervisor-level attackers (malicious cloud admins)
- Compromised host operating systems
- Memory scraping attacks

### The Solution: Intel TDX

| Feature | Protection |
|---------|------------|
| **Encrypted Memory** | AES-256 encryption of all VM memory |
| **Isolation** | Hardware-enforced isolation from hypervisor |
| **Attestation** | Cryptographic proof of secure execution |
| **No Code Changes** | Run existing AI workloads unmodified |

---

## TDX vs SGX

| Feature | Intel TDX | Intel SGX |
|---------|-----------|-----------|
| **Isolation Level** | Full VM (Trust Domain) | Application enclave |
| **Memory Limit** | Full VM RAM | Up to 512GB EPC |
| **Code Changes** | None (lift-and-shift) | Requires SDK/Gramine |
| **Attestation** | TD-level quote | Enclave-level quote |
| **Best For** | Cloud AI workloads | Fine-grained protection |
| **Lab** | **This lab (Lab 10)** | Lab 07 (SGX) |

---

## Lab Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STANDARD VM (No Protection)                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Guest VM Memory                              │ │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │ │
│  │  │   Model     │   │  Inference  │   │   Output    │          │ │
│  │  │   Weights   │   │    Data     │   │   Results   │          │ │
│  │  └─────────────┘   └─────────────┘   └─────────────┘          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ▲                                       │
│                              │ ATTACKER CAN READ!                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              Hypervisor / Cloud Operator                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     TDX VM (Hardware Protection)                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              Trust Domain (Encrypted Memory)                    │ │
│  │  ┌────────────────────────────────────────────────────────┐    │ │
│  │  │             🔒 AES-256 Encrypted Memory                │    │ │
│  │  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │    │ │
│  │  │  │   Model     │   │  Inference  │   │   Output    │  │    │ │
│  │  │  │   Weights   │   │    Data     │   │   Results   │  │    │ │
│  │  │  └─────────────┘   └─────────────┘   └─────────────┘  │    │ │
│  │  └────────────────────────────────────────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ▲                                       │
│                              │ ATTACKER SEES ENCRYPTED GARBAGE!     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              Hypervisor / Cloud Operator                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ▲                                       │
│                    ┌─────────┴─────────┐                            │
│                    │  Intel Xeon 6 CPU  │                            │
│                    │   TDX Hardware     │                            │
│                    │   Root of Trust    │                            │
│                    └────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Google Cloud Account
- GCP account with billing enabled
- Quota for C3 Confidential VMs (~$0.25/hr)

### Local Tools
- `gcloud` CLI installed and authenticated

---

## Lab Structure

```
lab-10-confidential-ai-tdx/
├── README.md                     # This file
├── SETUP_GUIDE.md                # Detailed GCP setup instructions
├── 0_check_tdx.py                # Verify TDX is active
├── 1_train_proprietary_model.py  # Train model (same as lab-07)
├── 2_victim_inference_server.py  # Run inference server
├── 3_attacker_memory_reader.py   # Memory extraction attack
├── 4_verify_tdx_protection.py    # Verify TDX blocks attack
├── requirements.txt              # Python dependencies
└── reset.py                      # Cleanup script
```

---

## Quick Start

### Step 1: Create GCP TDX VM

```bash
# Create Confidential VM with TDX
gcloud compute instances create tdx-ai-demo \
  --project=YOUR_PROJECT \
  --zone=us-central1-a \
  --machine-type=c3-standard-4 \
  --confidential-compute-type=TDX \
  --min-cpu-platform="Intel Sapphire Rapids" \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB

# SSH into the VM
gcloud compute ssh tdx-ai-demo --zone=us-central1-a
```

### Step 2: Setup Environment (Inside VM)

```bash
# Install dependencies
sudo apt update && sudo apt install -y python3-pip python3-venv git

# Clone repo and setup
git clone <your-repo-url>
cd ai-security/labs/lab-10-confidential-ai-tdx
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify TDX is active
python 0_check_tdx.py
```

### Step 3: Run Attack Demo

```bash
# Train model
python 1_train_proprietary_model.py

# Terminal 1: Run inference server
python 2_victim_inference_server.py

# Terminal 2: Attempt memory attack
sudo .venv/bin/python 3_attacker_memory_reader.py
# Result: Attack FAILS - memory is hardware-encrypted!
```

### Step 4: Verify Protection

```bash
# Run verification script
python 4_verify_tdx_protection.py
```

---

## Comparison Demo

For maximum impact, run the attack on both:

| Environment | Command | Result |
|-------------|---------|--------|
| Standard VM (e2-standard-4) | `sudo python 3_attacker_memory_reader.py` | ❌ **ATTACK SUCCEEDS** - Weights extracted |
| TDX VM (c3-standard-4 + TDX) | `sudo python 3_attacker_memory_reader.py` | ✅ **ATTACK FAILS** - Memory encrypted |

---

## What TDX Protects Against

| Attack Vector | Protection |
|---------------|------------|
| Malicious cloud operator | ✅ Memory encrypted, can't read |
| Compromised hypervisor | ✅ Isolated by hardware |
| Cold boot attacks | ✅ Keys destroyed on reboot |
| DMA attacks | ✅ Memory encryption engine |
| Memory bus snooping | ✅ Encrypted on the bus |

### What TDX Does NOT Protect Against

| Threat | Why |
|--------|-----|
| Application vulnerabilities | TDX protects memory, not your code |
| Side-channel attacks | Some side channels still possible |
| Key management errors | You must handle keys properly |
| Denial of service | Cloud provider can still stop your VM |

---

## Intel-Google TDX Collaboration

This lab aligns with Intel and Google's joint security review of TDX:

> "Intel and Google collaborated on a 5-month security review of Intel Trust Domain Extensions (TDX), Intel's trusted Confidential Computing technology."
> — INT31 Blog, February 2026

The collaboration validated:
- TDX Live Migration security
- TD Partitioning security
- TDX Module 1.5 code

---

## MITRE ATLAS Techniques Addressed

| Technique | Name | How TDX Helps |
|-----------|------|---------------|
| [AML.T0044](https://atlas.mitre.org/techniques/AML.T0044) | Full ML Model Access | Encrypted memory prevents weight extraction |
| [AML.T0024](https://atlas.mitre.org/techniques/AML.T0024) | Exfiltration via Inference | Isolated execution protects inference data |
| [AML.T0010](https://atlas.mitre.org/techniques/AML.T0010) | ML Supply Chain | Attestation verifies execution environment |

---

## Cost Management

```bash
# Stop VM when not using
gcloud compute instances stop tdx-ai-demo --zone=us-central1-a

# Start when ready
gcloud compute instances start tdx-ai-demo --zone=us-central1-a

# Delete after demo
gcloud compute instances delete tdx-ai-demo --zone=us-central1-a --quiet
```

| Resource | Cost |
|----------|------|
| c3-standard-4 (TDX) | ~$0.25/hr |
| 50GB SSD | ~$0.01/hr |
| **Weekend total** | ~$15-20 |

---

## Additional Resources

- [Intel TDX Official Documentation](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/documentation.html)
- [Google Cloud Confidential Computing](https://cloud.google.com/confidential-computing)
- [INT31 TDX Security Blog](https://www.intel.com/content/www/us/en/security/security-practices/int31.html)
- [TDX-Google Collaboration Video](https://www.youtube.com/watch?v=_n6WDifszh8)
- [Lab 07: SGX for AI Security](../lab-07-confidential-ai-sgx/) (Application-level enclaves)

---

## Key Takeaways

> 🔐 **Intel TDX** provides VM-level confidential computing with zero code changes

> 🛡️ **Hardware encryption** protects AI models from hypervisor-level attackers

> ✅ **Google Cloud C3** offers production-ready TDX for AI workloads

> 🔗 **Attestation** proves to customers their data is processed securely
