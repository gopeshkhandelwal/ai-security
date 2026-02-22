# TDX on GCP: Setup Guide

**Purpose:** Deploy Intel TDX Confidential VMs on Google Cloud Platform to demonstrate memory encryption vs plaintext.

**Estimated Time:** 30-45 minutes

---

## Lab Files (Simplified)

| File | Purpose |
|------|---------|
| `deploy_vms.sh` | Deploy both TDX + Standard VM automatically |
| `1_check_tdx.py` | Verify TDX is active on the VM |
| `2_memory_comparison_demo.py` | **Main demo**: TDX (encrypted) vs Standard (plaintext) |
| `3_tdx_attestation_demo.py` | Remote attestation (optional) |

---

## Prerequisites

- [ ] Google Cloud account with billing enabled
- [ ] `gcloud` CLI installed locally
- [ ] Basic familiarity with Linux command line

---

## Part 1: Local Setup

### 1.1 Install Google Cloud CLI

```bash
# Linux (Debian/Ubuntu)
sudo apt-get update && sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install -y google-cloud-cli

# Verify installation
gcloud --version
```

### 1.2 Authenticate with GCP

```bash
gcloud auth login
gcloud auth list
```

---

## Part 2: GCP Project Setup

### 2.1 Create Project

```bash
# Create new project (replace YOUR_PROJECT_ID with your project ID)
gcloud projects create YOUR_PROJECT_ID --name="Confidential AI TDX"

# Set as active project
gcloud config set project YOUR_PROJECT_ID

# Verify
gcloud config get-value project
```

### 2.2 Enable Billing

1. Go to: https://console.cloud.google.com/billing
2. Link a billing account to your project

### 2.3 Enable Required APIs

```bash
gcloud services enable compute.googleapis.com
gcloud services enable iap.googleapis.com

# Verify
gcloud services list --enabled | grep -E "compute|iap"
```

---

## Part 3: Create TDX Confidential VM

### 3.1 Understanding TDX Machine Requirements

**Which machines support TDX?**
- **C3 series** (Intel Sapphire Rapids / 4th Gen Xeon) → **Supports TDX**
- N2D series (AMD EPYC) → Supports AMD SEV-SNP (not TDX)
- C2 series (Intel Cascade Lake) → No confidential computing

**TDX-enabled zones** (not all zones have TDX hardware):
- `us-central1-a`, `us-central1-b`, `us-central1-c`
- `us-east4-a`, `us-east4-b`, `us-east4-c`  
- `europe-west1-b`, `europe-west1-c`, `europe-west1-d`

```bash
# List zones where C3 machines are available
gcloud compute machine-types list --filter="name=c3-standard-4" --format="value(zone)" | head -10
```

**Verify TDX support:** GCP will reject the VM creation if TDX is not supported in the chosen zone/machine combination. The `--confidential-compute-type=TDX` flag is validated at creation time.

### 3.2 Create the TDX VM

```bash
gcloud compute instances create tdx-ai-security \
  --project=YOUR_PROJECT_ID \
  --zone=us-central1-a \
  --machine-type=c3-standard-4 \
  --confidential-compute-type=TDX \
  --min-cpu-platform="Intel Sapphire Rapids" \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE
```

### 3.3 Configure Firewall for SSH

```bash
# Allow SSH via IAP (recommended for corporate networks)
gcloud compute firewall-rules create allow-ssh-iap \
  --project=YOUR_PROJECT_ID \
  --direction=INGRESS \
  --action=allow \
  --rules=tcp:22 \
  --source-ranges=35.235.240.0/20

# Allow direct SSH (optional)
gcloud compute firewall-rules create allow-ssh \
  --project=YOUR_PROJECT_ID \
  --allow=tcp:22 \
  --source-ranges=0.0.0.0/0
```

### 3.4 SSH into the VM

```bash
# Via IAP tunnel (recommended)
gcloud compute ssh tdx-ai-security --zone=us-central1-a --tunnel-through-iap

# Direct SSH (if firewall allows)
gcloud compute ssh tdx-ai-security --zone=us-central1-a
```

---

## Part 4: Verify TDX is Active

Run these commands inside the VM:

### 4.1 Check Kernel Messages

```bash
sudo dmesg | grep -i tdx
```

**Expected output:**
```
[    0.000000] tdx: Guest detected
[    1.452862] process: using TDX aware idle routine
[    1.452862] Memory Encryption Features active: Intel TDX
```

### 4.2 Check CPU Flags

```bash
grep -o 'tdx[^ ]*' /proc/cpuinfo | head -1
# Expected: tdx_guest
```

### 4.3 Check Attestation Device

```bash
ls -la /dev/tdx_guest
# Should show the TDX attestation device
```

### 4.4 Save Verification Output

```bash
sudo dmesg | grep -i tdx > ~/tdx_verification.txt
cat /proc/cpuinfo | head -30 >> ~/tdx_verification.txt
echo "=== TDX VERIFIED ===" >> ~/tdx_verification.txt
```

---

## Part 5: Set Up Lab Environment

### 5.1 Install Dependencies

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git build-essential
```

### 5.2 Clone Repository

```bash
cd ~
git clone https://github.com/YOUR_ORG/ai-security.git
cd ai-security/labs/lab-10-confidential-ai-tdx
```

### 5.3 Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Part 6: Run the Lab

### 6.1 Verify TDX Status

```bash
python3 1_check_tdx.py
```

### 6.2 Run Memory Comparison Demo

```bash
python3 2_memory_comparison_demo.py
```

### 6.3 Optional: Remote Attestation Demo

```bash
python3 3_tdx_attestation_demo.py
```

> **Note:** TDX protects against hypervisor-level attacks (malicious cloud provider), not process-to-process attacks within the same VM. For intra-VM protection, see Lab 07 (SGX).

---

## Part 7: Compare TDX VM vs Standard VM (Encrypted vs Plaintext)

This section demonstrates the security difference between a TDX-protected VM (encrypted memory) and a standard VM (plaintext memory).

### 7.1 Quick Deployment (Automated)

Use the deployment script to create both VMs automatically:

```bash
# Set your GCP project ID
export GCP_PROJECT_ID=your-project-id
export GCP_ZONE=us-central1-a

# Make script executable
chmod +x deploy_vms.sh

# Deploy both VMs
./deploy_vms.sh
```

This creates:
- **tdx-vm**: C3 machine with Intel TDX (memory encrypted)
- **standard-vm**: E2 machine without TDX (memory plaintext)

### 7.2 Manual Deployment

#### Create TDX VM (Encrypted Memory)

```bash
gcloud compute instances create tdx-vm \
  --project=YOUR_PROJECT_ID \
  --zone=us-central1-a \
  --machine-type=c3-standard-4 \
  --confidential-compute-type=TDX \
  --min-cpu-platform="Intel Sapphire Rapids" \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE
```

#### Create Standard VM (Plaintext Memory)

```bash
gcloud compute instances create standard-vm \
  --project=YOUR_PROJECT_ID \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB
```

### 7.3 Run the Memory Encryption Demo

**Step 1: SSH into TDX VM and run demo**

```bash
# Connect to TDX VM
gcloud compute ssh tdx-vm --zone=us-central1-a --tunnel-through-iap

# Inside the VM:
cd ~
git clone <your-repo-url> ai-security
cd ai-security/labs/lab-10-confidential-ai-tdx
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the memory comparison demo
python3 2_memory_comparison_demo.py
```

**Expected Output (TDX VM):**
```
✅ TDX Guest Mode: ACTIVE
✅ Memory: ENCRYPTED (AES-256)
✅ Hypervisor Access: BLOCKED

[HYPERVISOR] Attempting to read VM memory...
MEMORY CONTENTS: a7 f3 2b 9c 4e 8d 1a b0... (encrypted garbage)

🔒 ATTACK FAILED - Data is protected by hardware encryption
```

**Step 2: SSH into Standard VM and run the same demo**

```bash
# Connect to Standard VM
gcloud compute ssh standard-vm --zone=us-central1-a --tunnel-through-iap

# Run the same demo
python3 2_memory_comparison_demo.py
```

**Expected Output (Standard VM):**
```
❌ TDX Guest Mode: NOT ACTIVE
❌ Memory: PLAINTEXT
❌ Hypervisor Access: FULL ACCESS

[HYPERVISOR] Attempting to read VM memory...
MEMORY CONTENTS:
  [STOLEN] Model Name: ProprietaryLLM-v3.7
  [STOLEN] API Key: sk-abc123...
  [STOLEN] Customer SSN: 123-45-6789...

⚠️ ATTACK SUCCESSFUL - All data exposed in plaintext!
```

### 7.4 Side-by-Side Comparison

| Aspect | TDX VM (Encrypted) | Standard VM (Plaintext) |
|--------|-------------------|------------------------|
| Memory State | AES-256 encrypted | Plaintext |
| Hypervisor Access | BLOCKED | FULL ACCESS |
| Model Weights | Protected | Exposed |
| API Keys | Protected | Exposed |
| Customer Data | Protected | Exposed |
| Attestation | Available | Not available |
| Cost (c3/e2-standard-4) | ~$0.25/hr | ~$0.13/hr |

### 7.5 What This Demonstrates

1. **TDX VM**: The hypervisor sees only encrypted bytes. Even with root access to the host, the cloud provider cannot read your AI model or data.

2. **Standard VM**: The hypervisor can read all memory contents in plaintext, including model weights, API keys, and customer data.

> **Key Insight**: TDX provides hardware-enforced protection that cannot be bypassed by software - the encryption keys never leave the CPU.

---

## Part 8: Resource Management

### 8.1 Stop VMs (Cost Savings)

```bash
gcloud compute instances stop tdx-ai-security --zone=us-central1-a
gcloud compute instances stop standard-vm --zone=us-central1-a
```

### 8.2 Start VMs

```bash
gcloud compute instances start tdx-ai-security --zone=us-central1-a
```

### 8.3 Cost Estimates

| Resource | Hourly Cost |
|----------|-------------|
| c3-standard-4 (TDX) | ~$0.25 |
| e2-standard-4 (standard) | ~$0.13 |
| 50GB SSD | ~$0.01 |

### 8.4 Cleanup

```bash
# Delete VMs
gcloud compute instances delete tdx-ai-security --zone=us-central1-a --quiet
gcloud compute instances delete standard-vm --zone=us-central1-a --quiet

# Delete firewall rules
gcloud compute firewall-rules delete allow-ssh-iap --quiet
gcloud compute firewall-rules delete allow-ssh --quiet

# Delete project (removes all resources) - optional
# gcloud projects delete YOUR_PROJECT_ID
```

---

## Troubleshooting

### SSH Connection Timeout

```bash
# Use IAP tunnel instead of direct SSH
gcloud compute ssh tdx-ai-security --zone=us-central1-a --tunnel-through-iap
```

### Quota Exceeded Error

1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. Filter: "C3 CPUs"
3. Request quota increase

### TDX Not Detected

```bash
# Verify VM configuration
gcloud compute instances describe tdx-ai-security \
  --zone=us-central1-a \
  --format="yaml(confidentialInstanceConfig)"

# Should show: confidentialInstanceType: TDX
```

### Confidential Computing Unavailable

Try a different zone:
```bash
# Available zones: us-central1-a, us-east4-a, europe-west1-b
gcloud compute instances create tdx-ai-security \
  --zone=us-east4-a \
  # ... rest of options
```

---

## Additional Resources

- [Intel TDX Documentation](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/overview.html)
- [GCP Confidential Computing](https://cloud.google.com/confidential-computing)
- [Lab 07: SGX Protection](../lab-07-confidential-ai-sgx/README.md) - For intra-VM protection
