#!/bin/bash
#
# Platform Security - Secure Model Deployment for Intel vLLM
# Lightweight orchestration script - logic in Python files
#
# This script is part of ai-security/platform-security and should be
# invoked via bootstrap scripts from developer repos.
#
# Usage:
#   ./secure_vllm_deploy.sh <model-id> [options]
#
# Options:
#   --force              Re-download existing model
#   --benchmark          Run benchmark after deployment
#   --serve-only         Skip download/scan, just serve verified model

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID="${1:-meta-llama/Llama-3.1-8B}"
MODEL_NAME="${MODEL_ID##*/}"

# Use /srv for storage (more space available)
MODELS_DIR="${MODELS_DIR:-/srv/models/vLLM}"
QUARANTINE_DIR="${MODELS_DIR}/quarantine"
VERIFIED_DIR="${MODELS_DIR}/verified"
SCAN_RESULTS_DIR="${MODELS_DIR}/scan-results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Security tools directory (this repo, cloned by bootstrap script)
SECURITY_DIR="${SECURITY_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

CONTAINER_NAME="lsv-container"
CONTAINER_IMAGE="intel/llm-scaler-vllm:1.2"
VLLM_PORT=8001

# Flags
FORCE_DEPLOY=false
RUN_BENCHMARK=false
SERVE_ONLY=false

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_header()  { echo -e "\n============ $1 ============\n"; }

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

for arg in "$@"; do
    case $arg in
        --force)            FORCE_DEPLOY=true ;;
        --benchmark)        RUN_BENCHMARK=true ;;
        --serve-only)       SERVE_ONLY=true ;;
    esac
done

# =============================================================================
# SETUP SECURITY TOOLS
# =============================================================================

setup_security_tools() {
    log_header "Setup Platform Security Tools"
    
    # Verify security tools are available (should be cloned by bootstrap script)
    if [ ! -d "$SECURITY_DIR/platform-security" ]; then
        log_error "Security tools not found at: $SECURITY_DIR/platform-security"
        log_error "This script should be invoked via a bootstrap script that clones ai-security repo."
        log_error "Example: ./deploy.sh from your project's scripts directory"
        exit 1
    fi
    
    # Create symlink in models directory for container access
    SECURITY_MOUNT="${MODELS_DIR}/platform-security"
    if [ ! -L "$SECURITY_MOUNT" ] || [ "$(readlink -f "$SECURITY_MOUNT")" != "$(readlink -f "$SECURITY_DIR/platform-security")" ]; then
        sudo rm -rf "$SECURITY_MOUNT" 2>/dev/null || true
        sudo ln -sf "$SECURITY_DIR/platform-security" "$SECURITY_MOUNT"
    fi
    
    log_success "Platform security tools ready: $SECURITY_DIR/platform-security"
}

# =============================================================================
# DOCKER SETUP
# =============================================================================

setup_container() {
    log_header "Setup Container"
    
    # Pull image
    log_info "Pulling $CONTAINER_IMAGE..."
    docker pull "$CONTAINER_IMAGE"
    
    # Create directories
    sudo mkdir -p "$QUARANTINE_DIR" "$VERIFIED_DIR" "$SCAN_RESULTS_DIR"
    sudo chmod -R 777 "$MODELS_DIR"
    
    # Start container if not running
    if ! docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        log_info "Starting container..."
        
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
        
        sudo docker run -td \
            --privileged --net=host --device=/dev/dri \
            --name="$CONTAINER_NAME" \
            -v "$MODELS_DIR:/llm/models/" \
            -v "$SECURITY_DIR:/llm/security/" \
            -e no_proxy=localhost,127.0.0.1 \
            -e http_proxy="${http_proxy:-}" \
            -e https_proxy="${https_proxy:-}" \
            -e PYTHONPATH=/llm/security/platform-security/deploy \
            --shm-size="32g" \
            --entrypoint /bin/bash \
            "$CONTAINER_IMAGE"
        
        sleep 2
    fi
    
    # Install Python dependencies
    docker exec "$CONTAINER_NAME" pip install -q modelscan picklescan huggingface_hub 2>/dev/null || true
    
    log_success "Container ready"
    docker exec "$CONTAINER_NAME" xpu-smi discovery 2>/dev/null || true
}

# =============================================================================
# DOWNLOAD MODEL
# =============================================================================

download_model() {
    log_header "Download Model to Quarantine"
    
    if [ -d "$QUARANTINE_DIR/$MODEL_NAME" ] && [ "$FORCE_DEPLOY" != "true" ]; then
        log_info "Model exists in quarantine. Use --force to re-download"
        return 0
    fi
    
    rm -rf "$QUARANTINE_DIR/$MODEL_NAME"
    
    docker exec -e HF_TOKEN="${HF_TOKEN:-}" "$CONTAINER_NAME" \
        python3 /llm/security/platform-security/deploy/download_model.py \
            "$MODEL_ID" \
            --output-dir /llm/models/quarantine
    
    log_success "Download complete"
}

# =============================================================================
# SECURITY SCAN
# =============================================================================

security_scan() {
    log_header "Security Scan"
    
    SCAN_FILE="$SCAN_RESULTS_DIR/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).json"
    
    if docker exec "$CONTAINER_NAME" \
        python3 /llm/security/platform-security/deploy/scan_model.py \
            "/llm/models/quarantine/$MODEL_NAME" \
            --output "/llm/models/scan-results/$(basename "$SCAN_FILE")"; then
        log_success "Security scan PASSED"
        return 0
    else
        log_error "Security scan FAILED"
        log_info "Results: $SCAN_FILE"
        return 1
    fi
}

# =============================================================================
# PROMOTE MODEL
# =============================================================================

promote_model() {
    log_header "Promote to Verified"
    
    sudo rm -rf "$VERIFIED_DIR/$MODEL_NAME"
    sudo mv "$QUARANTINE_DIR/$MODEL_NAME" "$VERIFIED_DIR/$MODEL_NAME"
    sudo chmod -R 777 "$VERIFIED_DIR/$MODEL_NAME"
    
    # Generate MLBOM (security attestation)
    docker exec "$CONTAINER_NAME" \
        python3 /llm/security/platform-security/deploy/generate_mlbom.py \
            "/llm/models/verified/$MODEL_NAME" \
            --model-id "$MODEL_ID" \
            --scan-passed
    
    log_success "Model promoted: $VERIFIED_DIR/$MODEL_NAME"
}

# =============================================================================
# SERVE MODEL
# =============================================================================

serve_model() {
    log_header "Serve Model"
    
    log_info "Starting vLLM on port $VLLM_PORT..."
    
    docker exec -d "$CONTAINER_NAME" bash -c "
HF_TOKEN='${HF_TOKEN:-}' \
HF_HOME='/llm/models' \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve '/llm/models/verified/$MODEL_NAME' \
    --served-model-name '$MODEL_NAME' \
    --dtype=float16 \
    --enforce-eager \
    --port $VLLM_PORT \
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
    2>&1 | tee /llm/vllm.log &
"
    
    sleep 5
    log_success "vLLM starting..."
    log_info "Logs: docker exec $CONTAINER_NAME tail -f /llm/vllm.log"
    log_info "Test: curl http://localhost:$VLLM_PORT/v1/models"
}

# =============================================================================
# BENCHMARK
# =============================================================================

run_benchmark() {
    log_header "Benchmark"
    
    docker exec "$CONTAINER_NAME" bash -c "
vllm bench serve \
    --model '/llm/models/verified/$MODEL_NAME' \
    --dataset-name random \
    --served-model-name '$MODEL_NAME' \
    --random-input-len=1024 \
    --random-output-len=512 \
    --ignore-eos \
    --num-prompt 10 \
    --request-rate inf \
    --backend vllm \
    --port=$VLLM_PORT
"
}

# =============================================================================
# MAIN
# =============================================================================

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Platform Security - Secure vLLM Deployment                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL_ID"
echo ""

if [ "$SERVE_ONLY" == "true" ]; then
    setup_security_tools
    setup_container
    serve_model
else
    setup_security_tools
    setup_container
    download_model
    
    if security_scan; then
        promote_model
        serve_model
        [ "$RUN_BENCHMARK" == "true" ] && run_benchmark
        
        log_header "Deployment Complete"
        echo "✓ Model: $MODEL_NAME"
        echo "✓ Endpoint: http://localhost:$VLLM_PORT/v1"
        
        # Wait for server and test
        echo ""
        log_info "Waiting for vLLM server to be ready..."
        for i in {1..30}; do
            if curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
                log_success "Server is ready!"
                echo ""
                echo "Testing endpoint:"
                curl -s "http://localhost:$VLLM_PORT/v1/models" | python3 -m json.tool 2>/dev/null || curl -s "http://localhost:$VLLM_PORT/v1/models"
                echo ""
                break
            fi
            echo -n "."
            sleep 2
        done
        
        if ! curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
            log_warning "Server not responding yet. Check logs:"
            log_info "docker exec $CONTAINER_NAME tail -f /llm/vllm.log"
        fi
    else
        log_header "Deployment Blocked"
        log_error "Model failed security scan"
        echo "Model remains in: $QUARANTINE_DIR/$MODEL_NAME"
        echo ""
        echo "Options:"
        echo "  1. Choose a different model with safe format (safetensors)"
        echo "  2. Contact security team for review"
        echo ""
        echo "Scan results: $SCAN_RESULTS_DIR/"
        exit 1
    fi
fi
