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
├── deploy_vms.sh                 # Deploy TDX + Standard VM for comparison
├── 1_check_tdx.py                # Verify TDX is active
├── 2_memory_comparison_demo.py   # Main demo: TDX vs Standard VM
├── 3_tdx_attestation_demo.py     # Remote attestation demo (optional)
├── requirements.txt              # Python dependencies
└── reset.py                      # Cleanup script
```

---

## Quick Start

### Option A: Deploy Both VMs (Recommended for Demo)

```bash
# Set your GCP project and deploy both VMs
export GCP_PROJECT_ID=your-project-id
chmod +x deploy_vms.sh
./deploy_vms.sh
```

This creates:
- **tdx-vm**: C3 machine with TDX (memory encrypted)
- **standard-vm**: E2 machine without TDX (memory plaintext)

### Option B: Create Single TDX VM

```bash
# Create Confidential VM with TDX
gcloud compute instances create tdx-vm \
  --project=YOUR_PROJECT \
  --zone=us-central1-a \
  --machine-type=c3-standard-4 \
  --confidential-compute-type=TDX \
  --min-cpu-platform="Intel Sapphire Rapids" \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --maintenance-policy=TERMINATE

# SSH into the VM
gcloud compute ssh tdx-vm --zone=us-central1-a --tunnel-through-iap
```

### Setup Environment (Inside VM)

```bash
# Install dependencies
sudo apt update && sudo apt install -y python3-pip python3-venv git

# Clone repo and setup
git clone https://github.com/YOUR_ORG/ai-security.git
cd ai-security/labs/lab-10-confidential-ai-tdx
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Demo

```bash
# 1. Verify TDX is active
python3 1_check_tdx.py

# 2. Run main demo: TDX vs Standard VM comparison
python3 2_memory_comparison_demo.py

# 3. Optional: Remote attestation demo
python3 3_tdx_attestation_demo.py
```

> **Note:** TDX protects against HYPERVISOR-level attacks (malicious cloud provider).
> For intra-VM process isolation, use SGX (Lab 07).

---

## Understanding TDX Protection Scope

### What TDX PROTECTS Against:
| Threat | Protection |
|--------|------------|
| Malicious cloud operator | ✅ VM memory encrypted |
| Hypervisor memory dump | ✅ Returns encrypted garbage |
| Cold boot attack on host | ✅ Keys in CPU, not memory |
| Other VMs on same host | ✅ Hardware isolation |
| Host OS kernel exploit | ✅ Trust Domain isolation |

### What TDX DOES NOT Protect Against:
| Threat | Solution |
|--------|----------|
| Process-to-process attack (same VM) | Use SGX enclaves (Lab 07) |
| Software bugs in your application | Secure coding practices |
| Network-based attacks | Network security |
| Authorized user with VM access | Access control |

---

## Comparison Demo: TDX vs Standard VM

To see the difference between encrypted and plaintext memory:

### Option 1: Deploy Both VMs (Recommended)

```bash
# Set project ID and deploy
export GCP_PROJECT_ID=your-project-id
chmod +x deploy_vms.sh
./deploy_vms.sh
```

This creates:
- **tdx-vm**: Memory encrypted by Intel TDX
- **standard-vm**: Memory in plaintext (vulnerable)

### Option 2: Run Demo on Current VM

```bash
python3 2_memory_comparison_demo.py
```

This shows what a hypervisor would see when reading memory:
- **TDX VM**: Encrypted garbage (PROTECTED)
- **Standard VM**: Clear-text model weights (VULNERABLE)

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
# Stop VM when not in use
gcloud compute instances stop tdx-ai-security --zone=us-central1-a

# Start when ready
gcloud compute instances start tdx-ai-security --zone=us-central1-a

# Delete when finished
gcloud compute instances delete tdx-ai-security --zone=us-central1-a --quiet
```

| Resource | Hourly Cost |
|----------|-------------|
| c3-standard-4 (TDX) | ~$0.25 |
| 50GB SSD | ~$0.01 |

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
