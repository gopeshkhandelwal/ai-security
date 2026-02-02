# AI Security Platform - Model Security Scanner
# 
# Provides security scanning, MLBOM generation, and secure model pipeline
#
# Commands:
#   pipeline <model-id>    Full workflow: Download → Scan → Promote → MLBOM
#                          Input: HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)
#                          Output: Verified model + MLBOM + scan results
#
#   scan <model-path>      Scan only: Check existing model for security issues
#                          Input: Local path to model directory (must be mounted)
#                          Output: Pass/Fail + findings JSON
#
# Usage:
#   # Full pipeline - download and verify a new model
#   docker run -v /srv/models/vLLM:/srv/models/vLLM \
#     -e HF_TOKEN="$HF_TOKEN" \
#     -e http_proxy="$http_proxy" -e https_proxy="$https_proxy" \
#     amr-registry.caas.intel.com/intelcloud/ai-security:1.1 \
#     pipeline meta-llama/Llama-3.1-8B-Instruct
#
#   # Scan only - check a model already in quarantine
#   docker run -v /srv/models/vLLM:/srv/models/vLLM \
#     amr-registry.caas.intel.com/intelcloud/ai-security:1.1 \
#     scan /srv/models/vLLM/quarantine/gpt2
#
# Build (with proxy for corporate networks):
#   docker build \
#     --build-arg http_proxy="$http_proxy" \
#     --build-arg https_proxy="$https_proxy" \
#     --build-arg no_proxy="$no_proxy" \
#     -t amr-registry.caas.intel.com/intelcloud/ai-security:1.1 .

FROM python:3.11-slim

LABEL maintainer="Intel AI Security Team"
LABEL description="AI Model Security Scanner and MLBOM Generator"
LABEL version="1.4.0"

# Proxy args for corporate network
ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY model-security/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy platform-security code
COPY . /app/platform-security/

# Add scripts to PATH
ENV PATH="/app/platform-security/model-security/scripts:${PATH}"
ENV PYTHONPATH="/app/platform-security:/app/platform-security/model-security"

# Create default model directories
RUN mkdir -p /srv/models/vLLM/quarantine \
             /srv/models/vLLM/verified \
             /srv/models/vLLM/scan-results

# Make scripts executable
RUN chmod +x /app/platform-security/model-security/scripts/*.sh

# Default entrypoint
COPY <<EOF /app/entrypoint.sh
#!/bin/bash
set -e

case "\$1" in
    scan)
        shift
        python /app/platform-security/model-security/scanner.py "\$@"
        ;;
    pipeline)
        shift
        /app/platform-security/model-security/scripts/secure_pipeline.sh "\$@" --in-container
        ;;
    mlbom)
        shift
        python /app/platform-security/model-security/mlbom.py "\$@"
        ;;
    shell)
        exec /bin/bash
        ;;
    *)
        echo "AI Security Platform - Model Security"
        echo ""
        echo "Commands:"
        echo "  scan <model-path>      Run security scan on a model"
        echo "  pipeline <model-id>    Download, scan, and promote model"
        echo "  mlbom <model-path>     Generate MLBOM for a model"
        echo "  shell                  Start interactive shell"
        echo ""
        echo "Examples:"
        echo "  docker run -v /srv/models/vLLM:/srv/models/vLLM ai-security pipeline meta-llama/Llama-3.1-8B-Instruct"
        echo "  docker run -v /srv/models/vLLM:/srv/models/vLLM ai-security scan /srv/models/vLLM/quarantine/gpt2"
        ;;
esac
EOF

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
