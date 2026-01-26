# Pathfinder Secure Deployment

This directory contains scripts for secure model deployment on Intel accelerators.

## Quick Start

```bash
# Set your Hugging Face token
export HF_TOKEN="hf_..."

# Deploy a model with security scanning
./secure_vllm_deploy.sh meta-llama/Llama-3.1-8B

# Deploy with benchmark
./secure_vllm_deploy.sh meta-llama/Llama-3.1-8B --benchmark
```

## Security Pipeline

The deployment script implements a 6-stage security pipeline:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Download   │───▶│  Quarantine  │───▶│    Scan      │
│   Model      │    │   Storage    │    │  (ModelScan) │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                                               ▼
                    ┌──────────────┐    ┌──────────────┐
                    │    Serve     │◀───│   Promote    │
                    │   (vLLM)     │    │ to Verified  │
                    └──────────────┘    └──────────────┘
```

## Directory Structure

After deployment:

```
/home/compat/models/vLLM/
├── quarantine/          # Models being scanned (untrusted)
│   └── <model-name>/
├── verified/            # Approved models (scanned & safe)
│   └── <model-name>/
│       ├── model files...
│       └── mlbom.json   # ML Bill of Materials
├── scan-results/        # Security scan reports
│   └── <model>_scan_<timestamp>.json
└── pathfinder/          # Security scanning tools
    └── security/
```

## Command Reference

### Git Commands (Pathfinder Setup)

```bash
# Clone Pathfinder repository
git clone https://github.com/your-org/ai-security.git pathfinder

# Switch to pathfinder branch
cd pathfinder
git checkout pathfinder

# Update to latest
git fetch origin
git pull origin pathfinder
```

### Docker Commands

```bash
# Pull Intel vLLM image
docker pull intel/llm-scaler-vllm:1.2

# Start container
docker run -td \
    --privileged \
    --net=host \
    --device=/dev/dri \
    --name=lsv-container \
    -v /home/compat/models/vLLM:/llm/models/ \
    -e PYTHONPATH=/llm/pathfinder \
    --shm-size="32g" \
    --entrypoint /bin/bash \
    intel/llm-scaler-vllm:1.2

# Enter container
docker exec -it lsv-container bash

# Check GPUs
docker exec lsv-container xpu-smi discovery
```

### Model Download (Inside Container)

```bash
# Using huggingface-cli
HF_TOKEN="hf_..." HF_HOME="/llm/models" \
huggingface-cli download meta-llama/Llama-3.1-8B \
    --local-dir /llm/models/quarantine/Llama-3.1-8B

# Using Python
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'meta-llama/Llama-3.1-8B',
    local_dir='/llm/models/quarantine/Llama-3.1-8B',
    token='hf_...'
)
"
```

### Security Scanning (Inside Container)

```bash
# Install scanning tools
pip install modelscan picklescan

# Run Pathfinder scanner
python3 /llm/pathfinder/security/pathfinder_scanner.py \
    /llm/models/quarantine/Llama-3.1-8B

# Check results
cat /llm/models/scan-results/*.json | jq .
```

### Model Serving (After Scan Passes)

```bash
# Start vLLM (SECURE - no trust_remote_code)
HF_TOKEN="hf_" \
HF_HOME="/llm/models" \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve /llm/models/verified/Llama-3.1-8B \
    --served-model-name Llama-3.1-8B \
    --dtype=float16 \
    --enforce-eager \
    --port 8001 \
    --host 0.0.0.0 \
    --disable-sliding-window \
    --gpu-memory-util=0.9 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=8192 \
    --disable-log-requests \
    --max-model-len=8192 \
    --block-size 64 \
    --quantization fp8 \
    -tp=1 \
    2>&1 | tee /llm/vllm.log
```

### Benchmarking

```bash
vllm bench serve \
    --model /llm/models/verified/Llama-3.1-8B \
    --dataset-name random \
    --served-model-name Llama-3.1-8B \
    --random-input-len=1024 \
    --random-output-len=512 \
    --ignore-eos \
    --num-prompt 10 \
    --request-rate inf \
    --backend vllm \
    --port=8001
```

## Command-Line Options

```bash
./secure_vllm_deploy.sh <model-id> [options]
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--serve-only` | Skip download/scan, just serve an already-verified model | Disabled |
| `--skip-scan` | Skip security scanning (NOT RECOMMENDED) | Disabled |
| `--force` | Re-download even if model exists in quarantine | Disabled |
| `--trust-remote-code` | Enable trust_remote_code (DANGEROUS - allows arbitrary code execution) | Disabled |
| `--benchmark` | Run performance benchmark after deployment | Disabled |

### Examples

```bash
# Standard deployment with full security pipeline
./secure_vllm_deploy.sh meta-llama/Llama-3.1-8B

# Re-download a model (overwrite existing)
./secure_vllm_deploy.sh meta-llama/Llama-3.1-8B --force

# Serve an already-verified model (skip download/scan)
./secure_vllm_deploy.sh meta-llama/Llama-3.1-8B --serve-only

# Deploy and run benchmark
./secure_vllm_deploy.sh meta-llama/Llama-3.1-8B --benchmark

# Deploy model requiring custom code (use with caution)
./secure_vllm_deploy.sh helpful-ai/custom-model --trust-remote-code

# Skip scan (NOT RECOMMENDED - for testing only)
./secure_vllm_deploy.sh some-model --skip-scan
```

### Flag Details

#### `--serve-only`
Use this when the model has already been downloaded, scanned, and promoted to the verified directory. This skips:
- Pathfinder git setup
- Model download
- Security scanning
- Promotion step

Useful for:
- Restarting the vLLM server after a reboot
- Switching between already-verified models
- Quick server restarts without re-scanning

#### `--skip-scan`
⚠️ **NOT RECOMMENDED** - Bypasses security scanning entirely. The model will still be promoted to verified without validation. Only use for:
- Testing/development environments
- Models you've manually verified

#### `--force`
Forces re-download of the model even if it already exists in quarantine. Useful when:
- Model has been updated upstream
- Previous download was corrupted
- You want a fresh copy

#### `--trust-remote-code`
⚠️ **DANGEROUS** - Enables `trust_remote_code=True` in vLLM, allowing the model to execute arbitrary Python code during loading. Required for some models with custom architectures. Only use if:
- You've manually reviewed the model's Python files
- The model is from a trusted source
- You understand the security implications

#### `--benchmark`
Runs a performance benchmark after deployment using `vllm bench serve` with:
- 10 prompts
- 1024 token input length
- 512 token output length

## Scan Results

Example scan output:

```json
{
  "model_path": "/llm/models/quarantine/Llama-3.1-8B",
  "timestamp": "2026-01-26T12:00:00Z",
  "passed": true,
  "findings": []
}
```

Failed scan (blocked model):

```json
{
  "model_path": "/llm/models/quarantine/malicious-model",
  "timestamp": "2026-01-26T12:00:00Z",
  "passed": false,
  "findings": [
    {
      "severity": "CRITICAL",
      "category": "PTY_SPAWN",
      "message": "PTY shell spawning detected",
      "file": "modeling_evil.py"
    }
  ]
}
```

## Troubleshooting

### Model scan failed

```bash
# View detailed findings
cat /llm/models/scan-results/*_scan_*.json | jq '.findings[]'

# Options:
# 1. Choose a different model
# 2. Request security exception
# 3. Skip scan (NOT RECOMMENDED):
./secure_vllm_deploy.sh model-id --skip-scan
```

### Container won't start

```bash
# Check if container exists
docker ps -a | grep lsv-container

# Remove and recreate
docker rm -f lsv-container
./secure_vllm_deploy.sh meta-llama/Llama-3.1-8B
```

### vLLM won't start

```bash
# Check logs
docker exec lsv-container tail -100 /llm/vllm.log

# Check GPU availability
docker exec lsv-container xpu-smi discovery

# Check memory
docker exec lsv-container free -h
```
