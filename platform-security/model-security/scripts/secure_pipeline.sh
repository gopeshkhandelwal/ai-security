#!/bin/bash
#
# Platform Security - Secure Model Pipeline API
# 
# This is the PUBLIC API for developers. It handles:
#   - Model download to quarantine
#   - Security scanning (ModelScan, PickleScan, AST analysis)
#   - Promotion to verified directory
#   - MLBOM generation
#
# Developers call this script and then handle serving themselves.
#
# Usage:
#   ./secure_pipeline.sh <model-id> [options]
#
# Returns:
#   Exit 0: Model passed security scan, available at VERIFIED_DIR
#   Exit 1: Model failed security scan, remains in quarantine
#
# Output Variables (printed to stdout):
#   VERIFIED_MODEL_PATH=/path/to/verified/model
#

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID="${1:-}"
if [ -z "$MODEL_ID" ]; then
    echo "Usage: $0 <model-id> [--force]" >&2
    exit 1
fi

MODEL_NAME="${MODEL_ID##*/}"

# Directories
MODELS_DIR="${MODELS_DIR:-/srv/models/vLLM}"
QUARANTINE_DIR="${MODELS_DIR}/quarantine"
VERIFIED_DIR="${MODELS_DIR}/verified"
SCAN_RESULTS_DIR="${MODELS_DIR}/scan-results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-lsv-container}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-intel/llm-scaler-vllm:1.2}"

# Flags
FORCE_DEPLOY=false
for arg in "$@"; do
    case $arg in
        --force) FORCE_DEPLOY=true ;;
    esac
done

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1" >&2; }
log_success() { echo -e "${GREEN}[✓]${NC} $1" >&2; }
log_error()   { echo -e "${RED}[✗]${NC} $1" >&2; }
log_header()  { echo -e "\n============ $1 ============\n" >&2; }

# =============================================================================
# CONTAINER SETUP
# =============================================================================

setup_container() {
    log_header "Setup Container"
    
    docker pull "$CONTAINER_IMAGE" >&2
    
    sudo mkdir -p "$QUARANTINE_DIR" "$VERIFIED_DIR" "$SCAN_RESULTS_DIR"
    sudo chmod -R 777 "$MODELS_DIR"
    
    if ! docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        log_info "Starting container..."
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
        
        sudo docker run -td \
            --privileged --net=host --device=/dev/dri \
            --name="$CONTAINER_NAME" \
            -v "$MODELS_DIR:/llm/models/" \
            -v "$SCRIPT_DIR:/llm/security/" \
            -e no_proxy=localhost,127.0.0.1 \
            -e http_proxy="${http_proxy:-}" \
            -e https_proxy="${https_proxy:-}" \
            --shm-size="32g" \
            --entrypoint /bin/bash \
            "$CONTAINER_IMAGE" >&2
        
        sleep 2
    fi
    
    docker exec "$CONTAINER_NAME" pip install -q modelscan picklescan huggingface_hub 2>/dev/null || true
    log_success "Container ready"
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
        python3 /llm/security/download_model.py \
            "$MODEL_ID" \
            --output-dir /llm/models/quarantine >&2
    
    log_success "Download complete"
}

# =============================================================================
# SECURITY SCAN
# =============================================================================

security_scan() {
    log_header "Security Scan"
    
    SCAN_FILE="$SCAN_RESULTS_DIR/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).json"
    
    if docker exec "$CONTAINER_NAME" \
        python3 /llm/security/scan_model.py \
            "/llm/models/quarantine/$MODEL_NAME" \
            --output "/llm/models/scan-results/$(basename "$SCAN_FILE")" >&2; then
        log_success "Security scan PASSED"
        return 0
    else
        log_error "Security scan FAILED"
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
    docker exec "$CONTAINER_NAME" \
        python3 /llm/security/generate_mlbom.py \
            "/llm/models/verified/$MODEL_NAME" \
            --model-id "$MODEL_ID" \
            --scan-passed >&2
    
    log_success "Model promoted: $VERIFIED_DIR/$MODEL_NAME"
}

# =============================================================================
# MAIN
# =============================================================================

echo "" >&2
echo "╔═══════════════════════════════════════════════════════════════╗" >&2
echo "║     Platform Security - Model Security Pipeline               ║" >&2
echo "╚═══════════════════════════════════════════════════════════════╝" >&2
echo "" >&2
echo "Model: $MODEL_ID" >&2
echo "" >&2

setup_container
download_model

if security_scan; then
    promote_model
    
    log_header "Security Pipeline Complete"
    echo "Model verified and ready for serving" >&2
    
    # Output for calling script (stdout only)
    echo "VERIFIED_MODEL_PATH=$VERIFIED_DIR/$MODEL_NAME"
    echo "MODEL_NAME=$MODEL_NAME"
    echo "CONTAINER_NAME=$CONTAINER_NAME"
    
    exit 0
else
    log_header "Security Pipeline Failed"
    log_error "Model failed security scan - blocked from deployment"
    echo "Model remains in: $QUARANTINE_DIR/$MODEL_NAME" >&2
    echo "" >&2
    echo "Options:" >&2
    echo "  1. Choose a different model with safe format (safetensors)" >&2
    echo "  2. Contact security team for review" >&2
    exit 1
fi
