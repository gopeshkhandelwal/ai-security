# Platform Security

A reusable security framework for AI/ML model verification, scanning, and secure deployment. Provides defense-in-depth protection against supply chain attacks, malicious payloads, and unsafe model formats.

## Docker Image

```bash
docker pull amr-registry.caas.intel.com/intelcloud/ai-security:1.0
```

### Quick Start with Docker

```bash
# Show help
docker run amr-registry.caas.intel.com/intelcloud/ai-security:1.0

# Run full pipeline (download → scan → promote)
# Note: Include proxy vars if behind corporate firewall
docker run -v /srv/models:/srv/models/vLLM \
  -e HF_TOKEN=$HF_TOKEN \
  -e http_proxy="$http_proxy" -e https_proxy="$https_proxy" \
  -e HTTP_PROXY="$HTTP_PROXY" -e HTTPS_PROXY="$HTTPS_PROXY" \
  -e no_proxy="$no_proxy" \
  amr-registry.caas.intel.com/intelcloud/ai-security:1.0 \
  pipeline meta-llama/Llama-3.1-8B-Instruct

# Scan a local model
docker run -v /path/to/models:/models \
  amr-registry.caas.intel.com/intelcloud/ai-security:1.0 \
  scan /models/my-model

# Generate MLBOM
docker run -v /path/to/models:/models \
  amr-registry.caas.intel.com/intelcloud/ai-security:1.0 \
  mlbom /models/my-model

# Interactive shell
docker run -it amr-registry.caas.intel.com/intelcloud/ai-security:1.0 shell
```

### Build from Source

```bash
cd platform-security

# Build with proxy (required for corporate networks)
docker build \
  --build-arg http_proxy="$http_proxy" \
  --build-arg https_proxy="$https_proxy" \
  --build-arg no_proxy="$no_proxy" \
  -t amr-registry.caas.intel.com/intelcloud/ai-security:1.1 .

# Push to registry
docker push amr-registry.caas.intel.com/intelcloud/ai-security:1.1
```

### Proxy Configuration

When running behind a corporate proxy:

| Variable | Purpose | Example |
|----------|---------|--------|
| `http_proxy` | HTTP proxy URL | `http://proxy.example.com:911` |
| `https_proxy` | HTTPS proxy URL | `http://proxy.example.com:912` |
| `no_proxy` | Bypass proxy for these hosts | `localhost,127.0.0.1,.internal.com` |

**Build time**: Pass as `--build-arg` for apt-get and pip installs.  
**Run time**: Pass as `-e` environment variables for HuggingFace downloads.

## Structure

```
platform-security/
├── README.md
├── __init__.py
└── model-security/
    ├── __init__.py
    ├── requirements.txt
    ├── scanner.py          # ModelSecurityScanner class
    ├── loader.py           # SecureModelLoader class
    ├── mlbom.py            # MLBOMGenerator class
    ├── downloader.py       # ModelDownloader class
    └── scripts/
        ├── secure_pipeline.sh
        ├── git_commands.sh
        └── secure_vllm_deploy.sh
```

## Features

- **ModelSecurityScanner**: Multi-tool security scanning (ModelScan, PickleScan, AST patterns)
- **SecureModelLoader**: Verified model loading with signature verification
- **MLBOMGenerator**: Machine Learning Bill of Materials generation
- **ModelDownloader**: Secure download to quarantine with safetensors filtering
- **Format Enforcement**: Blocks pickle-based formats, requires SafeTensors
- **Cryptographic Verification**: ECDSA signature verification for model integrity

## Installation

```bash
cd platform-security/model-security
pip install -r requirements.txt
```

## Quick Start

### Security Scanning

```python
from model_security import ModelSecurityScanner

scanner = ModelSecurityScanner()
result = scanner.scan_model("/path/to/model")

if result.passed:
    print("✓ Model passed security scan")
else:
    for finding in result.findings:
        print(f"[{finding.severity.value}] {finding.message}")
```

### Secure Model Loading

```python
from model_security import SecureModelLoader

loader = SecureModelLoader(
    verified_models_path="/verified-models",
    public_key_path="/keys/signing.pub",
    device="cuda"
)

result = loader.load("meta-llama/Llama-3.2-1B")
if result.success:
    model, tokenizer = result.model, result.tokenizer
```

### Download to Quarantine

```python
from model_security import ModelDownloader

downloader = ModelDownloader(quarantine_dir="/models/quarantine")
result = downloader.download_safetensors_only("meta-llama/Llama-3.2-1B")
```

### Generate MLBOM

```python
from model_security import MLBOMGenerator

generator = MLBOMGenerator()
mlbom = generator.generate("/path/to/model", "meta-llama/Llama-3.2-1B")
mlbom.save()
```

## CLI

```bash
# Scan a model
python model-security/scanner.py /path/to/model --output scan.json

# Download to quarantine
python model-security/downloader.py meta-llama/Llama-3.2-1B --safetensors-only

# Generate MLBOM
python model-security/mlbom.py /path/to/model --model-id meta-llama/Llama-3.2-1B

# Run pipeline
./model-security/scripts/secure_pipeline.sh meta-llama/Llama-3.2-1B
```

## Severity Levels

| Severity | Action |
|----------|--------|
| CRITICAL | Block immediately |
| HIGH | Block, requires approval |
| MEDIUM | Warning, log for audit |
| LOW/INFO | Informational |

## License

MIT
