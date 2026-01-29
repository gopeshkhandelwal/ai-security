# AI Security Platform - Model Security Scanner
# 
# Provides security scanning, MLBOM generation, and secure model pipeline
#
# Usage:
#   docker run ai-security scan <model-id>
#   docker run ai-security pipeline <model-id>
#   docker run -v /models:/srv/models/vLLM ai-security pipeline <model-id>
#
# Build with proxy:
#   docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t ai-security:1.0.0 .

FROM python:3.11-slim

LABEL maintainer="Intel AI Security Team"
LABEL description="AI Model Security Scanner and MLBOM Generator"
LABEL version="1.0.0"

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
ENV PYTHONPATH="/app/platform-security:/app/platform-security/model-security:${PYTHONPATH:-}"

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
        python -m model_security.scanner "\$@"
        ;;
    pipeline)
        shift
        /app/platform-security/model-security/scripts/secure_pipeline.sh "\$@" --in-container
        ;;
    mlbom)
        shift
        python -m model_security.mlbom "\$@"
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
        echo "  docker run ai-security pipeline meta-llama/Llama-3.1-8B-Instruct"
        echo "  docker run -v ./models:/srv/models/vLLM ai-security pipeline Qwen/Qwen2-7B"
        ;;
esac
EOF

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
