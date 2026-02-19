# TDX on GCP: Setup Guide

**Purpose:** Deploy Intel TDX Confidential VMs on Google Cloud Platform for AI model protection.

**Estimated Time:** 30-45 minutes

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
# Create new project (use your own project name)
gcloud projects create YOUR_PROJECT_ID --name="AI Security TDX"

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

### 3.1 Check Available Zones

```bash
# List zones that support C3 machines with TDX
gcloud compute machine-types list --filter="name=c3-standard-4" --format="value(zone)" | head -10
```

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
python 0_check_tdx.py
```

### 6.2 Train the Model

```bash
python 1_train_proprietary_model.py
```

### 6.3 Run Inference Server

**Terminal 1:**
```bash
python 2_victim_inference_server.py
```

### 6.4 Attempt Memory Attack

**Terminal 2 (new SSH session):**
```bash
cd ~/ai-security/labs/lab-10-confidential-ai-tdx
source .venv/bin/activate
sudo .venv/bin/python 3_attacker_memory_reader.py
```

> **Note:** This intra-VM attack will succeed because TDX protects against hypervisor-level attacks, not process-to-process attacks within the same VM. For intra-VM protection, see Lab 07 (SGX).

### 6.5 Verify TDX Protection Scope

```bash
python 4_verify_tdx_protection.py
```

---

## Part 7: Compare with Standard VM (Optional)

To demonstrate that TDX provides protection against hypervisor-level attacks:

### 7.1 Create Standard VM

```bash
gcloud compute instances create standard-vm \
  --project=YOUR_PROJECT_ID \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB
```

### 7.2 Run Same Lab on Standard VM

```bash
gcloud compute ssh standard-vm --zone=us-central1-a

# Setup environment (same steps as Part 5)
# Run lab (same steps as Part 6)
# Compare results
```

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

# Delete project (removes all resources)
gcloud projects delete YOUR_PROJECT_ID
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
