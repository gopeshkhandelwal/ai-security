#!/bin/bash
#
# Pathfinder Secure Model Deployment for Intel vLLM
# Lightweight orchestration script - logic in Python files
#
# Usage:
#   ./secure_vllm_deploy.sh <model-id> [options]
#
# Options:
#   --skip-scan          Skip security scanning (NOT RECOMMENDED)
#   --force              Re-download existing model
#   --trust-remote-code  Enable trust_remote_code (DANGEROUS)
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

CONTAINER_NAME="lsv-container"
CONTAINER_IMAGE="intel/llm-scaler-vllm:1.2"
VLLM_PORT=8001

# Flags
SKIP_SCAN=false
FORCE_DEPLOY=false
TRUST_REMOTE_CODE=false
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
        --skip-scan)        SKIP_SCAN=true ;;
        --force)            FORCE_DEPLOY=true ;;
        --trust-remote-code) TRUST_REMOTE_CODE=true ;;
        --benchmark)        RUN_BENCHMARK=true ;;
        --serve-only)       SERVE_ONLY=true ;;
    esac
done

# =============================================================================
# GIT SETUP
# =============================================================================

setup_pathfinder() {
    log_header "Setup Pathfinder"
    
    PATHFINDER_DIR="${MODELS_DIR}/pathfinder"
    
    if [ -d "$PATHFINDER_DIR/.git" ]; then
        log_info "Updating Pathfinder..."
        cd "$PATHFINDER_DIR"
        git fetch origin
        git checkout pathfinder 2>/dev/null || git checkout main
        git pull
    else
        log_info "Cloning Pathfinder..."
        sudo mkdir -p "$MODELS_DIR"
        cd "$MODELS_DIR"
        git clone https://github.com/intel/ai-security.git pathfinder 2>/dev/null || {
            log_warning "Clone failed, copying from local..."
            sudo cp -r "$SCRIPT_DIR/.." "$PATHFINDER_DIR"
        }
        cd "$PATHFINDER_DIR" && git checkout pathfinder 2>/dev/null || true
    fi
    
    log_success "Pathfinder ready"
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
            -e no_proxy=localhost,127.0.0.1 \
            -e http_proxy="${http_proxy:-}" \
            -e https_proxy="${https_proxy:-}" \
            -e PYTHONPATH=/llm/models/pathfinder/deploy \
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
        python3 /llm/models/pathfinder/deploy/download_model.py \
            "$MODEL_ID" \
            --output-dir /llm/models/quarantine
    
    log_success "Download complete"
}

# =============================================================================
# SECURITY SCAN
# =============================================================================

security_scan() {
    log_header "Security Scan"
    
    if [ "$SKIP_SCAN" == "true" ]; then
        log_warning "SCAN SKIPPED (--skip-scan)"
        return 0
    fi
    
    SCAN_FILE="$SCAN_RESULTS_DIR/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).json"
    
    if docker exec "$CONTAINER_NAME" \
        python3 /llm/models/pathfinder/deploy/scan_model.py \
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
    
    # Generate MLBOM
    TRUST_FLAG=""
    [ "$TRUST_REMOTE_CODE" == "true" ] && TRUST_FLAG="--trust-remote-code"
    
    docker exec "$CONTAINER_NAME" \
        python3 /llm/models/pathfinder/deploy/generate_mlbom.py \
            "/llm/models/verified/$MODEL_NAME" \
            --model-id "$MODEL_ID" \
            --scan-passed \
            $TRUST_FLAG
    
    log_success "Model promoted: $VERIFIED_DIR/$MODEL_NAME"
}

# =============================================================================
# SERVE MODEL
# =============================================================================

serve_model() {
    log_header "Serve Model"
    
    TRUST_FLAG=""
    [ "$TRUST_REMOTE_CODE" == "true" ] && TRUST_FLAG="--trust-remote-code"
    
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
    $TRUST_FLAG \
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
echo "║     AI Model Pathfinder - Secure vLLM Deployment              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL_ID"
echo ""

if [ "$SERVE_ONLY" == "true" ]; then
    setup_container
    serve_model
else
    setup_pathfinder
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
        echo "Options:"
        echo "  1. Choose a different model"
        echo "  2. Skip scan (dangerous): $0 $MODEL_ID --skip-scan"
        exit 1
    fi
fi
