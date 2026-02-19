# TDX on GCP: Step-by-Step Setup Guide

**Purpose:** Set up Intel TDX Confidential VM on Google Cloud Platform to demonstrate AI model protection.

**Timeline:** Thursday PM → Tuesday meeting

---

## Prerequisites Checklist

- [ ] Google Cloud account (create at https://cloud.google.com if needed)
- [ ] Billing enabled (credit card required)
- [ ] gcloud CLI installed locally

---

## Part 1: Local Setup (Your Machine)

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
# This opens a browser for authentication
gcloud auth login

# Verify you're logged in
gcloud auth list
```

---

## Part 2: GCP Project Setup

### 2.1 Create Project

```bash
# Create new project
gcloud projects create ai-security-tdx-demo --name="AI Security TDX Demo"

# Set as active project
gcloud config set project ai-security-tdx-demo

# Verify
gcloud config get-value project
# Expected output: ai-security-tdx-demo
```

### 2.2 Enable Billing

**MANUAL STEP - Do in browser:**

1. Go to: https://console.cloud.google.com/billing
2. Click "Link a billing account"
3. Select your billing account (or create one with credit card)
4. Link to project "ai-security-tdx-demo"

### 2.3 Enable Required APIs

```bash
# Enable Compute Engine API
gcloud services enable compute.googleapis.com

# Verify it's enabled
gcloud services list --enabled | grep compute
# Expected: compute.googleapis.com
```

---

## Part 3: Create TDX Confidential VM

### 3.1 Check Available Zones

```bash
# List zones that support C3 machines
gcloud compute machine-types list --filter="name=c3-standard-4" --format="value(zone)" | head -10
```

### 3.2 Create the TDX VM

```bash
# Create Confidential VM with TDX
gcloud compute instances create tdx-ai-demo \
  --project=ai-security-tdx-demo \
  --zone=us-central1-a \
  --machine-type=c3-standard-4 \
  --confidential-compute-type=TDX \
  --min-cpu-platform="Intel Sapphire Rapids" \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd \
  --tags=tdx-demo \
  --metadata=enable-oslogin=true

# Expected output:
# Created [https://www.googleapis.com/compute/v1/projects/ai-security-tdx-demo/zones/us-central1-a/instances/tdx-ai-demo].
# NAME          ZONE           MACHINE_TYPE   PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP    STATUS
# tdx-ai-demo   us-central1-a  c3-standard-4               10.x.x.x     34.x.x.x       RUNNING
```

### 3.3 SSH into the VM

```bash
# SSH into the VM
gcloud compute ssh tdx-ai-demo --zone=us-central1-a

# You should now be inside the VM
# Prompt changes to: username@tdx-ai-demo:~$
```

---

## Part 4: Verify TDX is Active (Inside VM)

### 4.1 Check TDX Status

```bash
# Check kernel messages for TDX
sudo dmesg | grep -i tdx

# Expected output (one of these):
# [    0.000000] tdx: Guest detected
# OR
# [    0.000000] x86/tdx: Guest detected

# Check CPU flags
grep -o 'tdx[^ ]*' /proc/cpuinfo | head -1
# May show: tdx_guest

# Check /sys for TDX
ls /sys/firmware/ | grep -i tdx
```

### 4.2 Capture Screenshot/Output

**IMPORTANT:** Save this output for your demo!

```bash
# Save TDX verification to file
sudo dmesg | grep -i tdx > ~/tdx_verification.txt
cat /proc/cpuinfo | head -30 >> ~/tdx_verification.txt
echo "---TDX CONFIRMED---" >> ~/tdx_verification.txt
cat ~/tdx_verification.txt
```

---

## Part 5: Set Up Python Environment (Inside VM)

### 5.1 Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and git
sudo apt install -y python3-pip python3-venv git build-essential

# Verify Python
python3 --version
# Expected: Python 3.10.x or higher
```

### 5.2 Clone Your Repository

```bash
# Clone your repo (replace with your actual repo URL)
cd ~
git clone https://github.com/YOUR_USERNAME/ai-security.git
# OR if private repo, use HTTPS with token

# Navigate to lab
cd ai-security/labs/lab-07-confidential-ai-tdx-sgx
```

### 5.3 Create Virtual Environment

```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Verify venv is active
which python
# Expected: /home/username/ai-security/labs/lab-07-confidential-ai-tdx-sgx/.venv/bin/python

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Part 6: Run the Demo

### 6.1 Train the Model

```bash
# Make sure venv is active
source .venv/bin/activate

# Train proprietary model
python 1_train_proprietary_model.py

# Expected output:
# Training proprietary model...
# Model saved to: proprietary_model.h5
```

### 6.2 Run Attack Demo (Two Terminals)

**Terminal 1 - Inference Server:**
```bash
cd ~/ai-security/labs/lab-07-confidential-ai-tdx-sgx
source .venv/bin/activate
python 2_victim_inference_server.py

# Keep this running - shows inference server is active
```

**Terminal 2 - Attack (new SSH session):**
```bash
# Open new terminal, SSH again
gcloud compute ssh tdx-ai-demo --zone=us-central1-a

cd ~/ai-security/labs/lab-07-confidential-ai-tdx-sgx
source .venv/bin/activate

# Run the memory attack
sudo .venv/bin/python 3_attacker_memory_reader.py

# EXPECTED RESULT ON TDX:
# Attack should FAIL or return encrypted/unreadable data
# Memory is hardware-encrypted by TDX
```

### 6.3 Capture Demo Output

```bash
# Save attack output to file for demo
sudo .venv/bin/python 3_attacker_memory_reader.py 2>&1 | tee ~/tdx_attack_result.txt

# Download files to your local machine (from your local terminal)
gcloud compute scp tdx-ai-demo:~/tdx_verification.txt . --zone=us-central1-a
gcloud compute scp tdx-ai-demo:~/tdx_attack_result.txt . --zone=us-central1-a
```

---

## Part 7: Create Comparison (Non-TDX VM)

To show the attack WORKS without TDX protection:

### 7.1 Create Standard VM (No TDX)

```bash
# From your local machine
gcloud compute instances create standard-vm-demo \
  --project=ai-security-tdx-demo \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB
```

### 7.2 Run Same Attack on Standard VM

```bash
# SSH into standard VM
gcloud compute ssh standard-vm-demo --zone=us-central1-a

# Set up environment (same steps as above)
sudo apt update && sudo apt install -y python3-pip python3-venv git
git clone <your-repo>
cd ai-security/labs/lab-07-confidential-ai-tdx-sgx
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run attack
python 1_train_proprietary_model.py
python 2_victim_inference_server.py &
sudo .venv/bin/python 3_attacker_memory_reader.py

# EXPECTED: Attack SUCCEEDS - model weights extracted!
```

---

## Part 8: Cost Management

### 8.1 Stop VMs When Not Using

```bash
# Stop VMs to save money
gcloud compute instances stop tdx-ai-demo --zone=us-central1-a
gcloud compute instances stop standard-vm-demo --zone=us-central1-a

# Start when ready to work
gcloud compute instances start tdx-ai-demo --zone=us-central1-a
```

### 8.2 Cost Estimates

| Resource | Hourly Cost | 60hr Weekend |
|----------|-------------|--------------|
| c3-standard-4 (TDX) | ~$0.25 | ~$15 |
| e2-standard-4 (standard) | ~$0.13 | ~$8 |
| 100GB SSD total | ~$0.02 | ~$1 |
| **Total** | | **~$25** |

### 8.3 Delete Everything After Meeting

```bash
# Delete VMs
gcloud compute instances delete tdx-ai-demo --zone=us-central1-a --quiet
gcloud compute instances delete standard-vm-demo --zone=us-central1-a --quiet

# Delete project (removes all resources)
gcloud projects delete ai-security-tdx-demo
```

---

## Part 9: Demo Script for Meeting

### What to Show Dhinesh

1. **Open with TDX verification:**
   ```
   "Running on GCP C3 Confidential VM with TDX enabled"
   Show: dmesg | grep tdx output
   ```

2. **Show the attack on standard VM:**
   ```
   "First, let me show the attack on a standard VM"
   Run: 3_attacker_memory_reader.py
   Result: Model weights extracted - attack succeeds
   ```

3. **Show the same attack on TDX VM:**
   ```
   "Now the same attack inside a TDX Trust Domain"
   Run: 3_attacker_memory_reader.py
   Result: Memory encrypted - attack fails
   ```

4. **The message:**
   ```
   "This is why Intel's TDX is essential for protecting 
   AI models in cloud environments"
   ```

---

## Troubleshooting

### Issue: "Quota exceeded" error
```bash
# Request quota increase for C3 CPUs
# Go to: https://console.cloud.google.com/iam-admin/quotas
# Filter: "C3 CPUs"
# Request increase to 8
```

### Issue: "Confidential computing not available"
```bash
# Try different zone
gcloud compute instances create tdx-ai-demo \
  --zone=us-east4-a \  # Different zone
  ... (rest of command)
```

### Issue: SSH connection refused
```bash
# Wait 1-2 minutes for VM to boot fully
# Check VM status
gcloud compute instances describe tdx-ai-demo --zone=us-central1-a --format="value(status)"
```

### Issue: TDX not showing in dmesg
```bash
# Verify VM was created with TDX
gcloud compute instances describe tdx-ai-demo --zone=us-central1-a --format="value(confidentialInstanceConfig)"
# Should show: enableConfidentialCompute: true
```

---

## Friday Cutoff Decision

**By Friday 6pm, you should have:**
- [ ] TDX VM running
- [ ] `dmesg | grep tdx` shows "Guest detected"
- [ ] Attack script runs (even if you're still debugging output)

**If any of these are missing by Friday 6pm → Abandon TDX, focus on SGX demo**

---

## Files to Have Ready for Demo

1. `~/tdx_verification.txt` - Proof TDX is active
2. `~/tdx_attack_result.txt` - Attack output on TDX (fails)
3. `~/standard_attack_result.txt` - Attack output on standard VM (succeeds)
4. Live SSH sessions to both VMs (as backup)

---

**Good luck! Let me know if you hit any issues during setup.**
