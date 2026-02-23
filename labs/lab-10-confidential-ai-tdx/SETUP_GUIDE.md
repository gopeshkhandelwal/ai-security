# Setup Guide

## Create VMs on GCP

### Standard VM (no protection)

```bash
gcloud compute instances create standard-vm \
  --project=YOUR_PROJECT \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud
```

### TDX VM (memory encrypted)

```bash
gcloud compute instances create tdx-vm \
  --project=YOUR_PROJECT \
  --zone=us-central1-a \
  --machine-type=c3-standard-4 \
  --min-cpu-platform="Intel Sapphire Rapids" \
  --confidential-compute-type=TDX \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud
```

## SSH into VMs

```bash
gcloud compute ssh standard-vm --zone=us-central1-a
gcloud compute ssh tdx-vm --zone=us-central1-a
```

## Run Demo

```bash
# Clone repo and run
git clone https://github.com/YOUR_REPO/ai-security.git
cd ai-security/labs/lab-10-confidential-ai-tdx
python3 2_memory_comparison_demo.py
```
