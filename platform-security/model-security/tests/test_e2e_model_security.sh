#!/bin/bash
#
# Model Security E2E Integration Test
#
# This test validates the complete model security pipeline:
#   1. Setup directories and container (optional)
#   2. Download model to quarantine
#   3. Run security scan
#   4. Promote to verified directory
#   5. Generate MLBOM
#   6. Serve with vLLM (optional)
#   7. Run benchmark (optional)
#
# Usage:
#   ./test_e2e_model_security.sh [model-id] [options]
#
# Options:
#   --mock                      Use mock model (no real download)
#   --docker-image=<image>      Test using ai-security Docker image
#   --container                 Use Intel vLLM container mode (for serving)
#   --serve                     Start vLLM server after security scan
#   --benchmark                 Run vLLM benchmark after serving
#   --force                     Force redeploy even if model exists
#
# Examples:
#   ./test_e2e_model_security.sh openai-community/gpt2
#   ./test_e2e_model_security.sh --mock                           # Test without real download
#   ./test_e2e_model_security.sh --docker-image=amr-registry.caas.intel.com/intelcloud/ai-security:1.7.0
#   ./test_e2e_model_security.sh meta-llama/Llama-3.1-8B --container --serve
#   ./test_e2e_model_security.sh meta-llama/Llama-3.1-8B --container --serve --benchmark

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID="${1:-openai-community/gpt2}"
MODEL_NAME="${MODEL_ID##*/}"

# Model storage paths
MODELS_DIR="${MODELS_DIR:-/tmp/model-security-test/models}"
VERIFIED_DIR="${MODELS_DIR}/verified"
QUARANTINE_DIR="${MODELS_DIR}/quarantine"
SCAN_RESULTS_DIR="${MODELS_DIR}/scan-results"

# Security framework path (this repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECURITY_MODULE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Container settings
USE_CONTAINER="${USE_CONTAINER:-false}"  # Set true for container mode
CONTAINER_NAME="${CONTAINER_NAME:-model-security-test}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-intel/llm-scaler-vllm:1.2}"

# AI Security Docker image (optional - use instead of local Python files)
AI_SECURITY_IMAGE=""

# vLLM settings
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-fp8}"

# Flags
MOCK_MODE=false
FORCE_DEPLOY=false
DO_SERVE=false
DO_BENCHMARK=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
        --mock)       MOCK_MODE=true ;;
        --force)      FORCE_DEPLOY=true ;;
        --container)  USE_CONTAINER=true ;;
        --serve)      DO_SERVE=true ;;
        --benchmark)  DO_BENCHMARK=true ;;
        --docker-image=*) AI_SECURITY_IMAGE="${arg#*=}" ;;
        -*)           ;; 
        *)            
            if [ "$arg" != "--mock" ] && [ "$arg" != "--force" ] && [ "$arg" != "--container" ] && [ "$arg" != "--serve" ] && [ "$arg" != "--benchmark" ]; then
                MODEL_ID="$arg"
                MODEL_NAME="${MODEL_ID##*/}"
            fi
            ;;
    esac
done

# =============================================================================
# SETUP
# =============================================================================

setup() {
    log_header "Setup"
    
    # Create directories
    mkdir -p "$QUARANTINE_DIR" "$VERIFIED_DIR" "$SCAN_RESULTS_DIR"
    
    log_success "Directories created:"
    log_info "  Quarantine: $QUARANTINE_DIR"
    log_info "  Verified: $VERIFIED_DIR"
    log_info "  Scan Results: $SCAN_RESULTS_DIR"
    
    # Verify security module is available
    if [ ! -f "$SECURITY_MODULE_DIR/scanner.py" ]; then
        log_error "Security module not found at: $SECURITY_MODULE_DIR"
        exit 1
    fi
    log_success "Security module: $SECURITY_MODULE_DIR"
}

# =============================================================================
# SETUP CONTAINER (Optional)
# =============================================================================

setup_container() {
    if [ "$USE_CONTAINER" != "true" ]; then
        log_info "Running in local mode (set --container for Docker mode)"
        return 0
    fi
    
    log_header "Setup Container (Intel vLLM)"
    
    # Pull the Intel vLLM image
    log_info "Pulling $CONTAINER_IMAGE..."
    docker pull "$CONTAINER_IMAGE"
    
    # Create model directories
    sudo mkdir -p "$QUARANTINE_DIR" "$VERIFIED_DIR" "$SCAN_RESULTS_DIR"
    sudo chown -R "$(id -u):$(id -g)" "$MODELS_DIR"
    chmod -R 755 "$MODELS_DIR"
    
    # Always recreate container to ensure correct volume mounts
    log_info "Starting container..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    
    sudo docker run -td \
        --net=host \
        --device=/dev/dri \
        --name="$CONTAINER_NAME" \
        -v "$MODELS_DIR:/llm/models/" \
        -v "$SECURITY_MODULE_DIR:/llm/security/" \
        -e no_proxy=localhost,127.0.0.1 \
        -e http_proxy="${http_proxy:-}" \
        -e https_proxy="${https_proxy:-}" \
        --shm-size="32g" \
        --entrypoint /bin/bash \
        "$CONTAINER_IMAGE"
    
    sleep 2
    
    # Check XPU availability
    log_info "Checking Intel XPU..."
    docker exec "$CONTAINER_NAME" xpu-smi discovery 2>/dev/null || log_warning "No Intel XPU detected"
    
    # Install security tools if not present
    log_info "Installing security dependencies..."
    docker exec "$CONTAINER_NAME" pip install -q modelscan picklescan 2>/dev/null || true
    
    log_success "Container ready: $CONTAINER_NAME"
}

# =============================================================================
# CREATE MOCK MODEL (for testing without download)
# =============================================================================

create_mock_model() {
    log_header "Create Mock Model"
    
    MODEL_NAME="mock-gpt2"
    local model_path="$QUARANTINE_DIR/$MODEL_NAME"
    mkdir -p "$model_path"
    
    # Create a minimal valid model structure
    cat > "$model_path/config.json" << 'EOF'
{
    "model_type": "gpt2",
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12
}
EOF
    
    cat > "$model_path/tokenizer_config.json" << 'EOF'
{
    "model_max_length": 1024
}
EOF
    
    # Create empty safetensors file (safe format)
    touch "$model_path/model.safetensors"
    
    log_success "Mock model created: $model_path"
}

# =============================================================================
# RUN DOCKER IMAGE PIPELINE (when --docker-image is specified)
# =============================================================================

run_docker_pipeline() {
    log_header "Run Docker Image Pipeline"
    
    log_info "Using Docker image: $AI_SECURITY_IMAGE"
    log_info "Running full pipeline: download → scan → promote → mlbom"
    
    if docker run --rm \
        -v "$MODELS_DIR:/srv/models/vLLM" \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e http_proxy="${http_proxy:-}" \
        -e https_proxy="${https_proxy:-}" \
        "$AI_SECURITY_IMAGE" \
        pipeline "$MODEL_ID"; then
        log_success "Docker pipeline completed successfully"
        return 0
    else
        log_error "Docker pipeline failed"
        return 1
    fi
}

# =============================================================================
# DOWNLOAD MODEL TO QUARANTINE
# =============================================================================

download_model() {
    log_header "Download Model to Quarantine"
    
    if [ -d "$QUARANTINE_DIR/$MODEL_NAME" ] && [ "$FORCE_DEPLOY" != "true" ]; then
        log_info "Model exists in quarantine. Use --force to re-download"
        return 0
    fi
    
    sudo rm -rf "$QUARANTINE_DIR/$MODEL_NAME" 2>/dev/null || rm -rf "$QUARANTINE_DIR/$MODEL_NAME"
    
    log_info "Downloading $MODEL_ID..."
    
    # Build token arg only if HF_TOKEN is set and non-empty
    local token_arg=""
    if [ -n "${HF_TOKEN:-}" ]; then
        token_arg="--token $HF_TOKEN"
    fi
    
    if [ -n "$AI_SECURITY_IMAGE" ]; then
        # Use ai-security Docker image
        log_info "Using Docker image: $AI_SECURITY_IMAGE"
        docker run --rm \
            -v "$MODELS_DIR:/srv/models/vLLM" \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            -e http_proxy="${http_proxy:-}" \
            -e https_proxy="${https_proxy:-}" \
            "$AI_SECURITY_IMAGE" \
            pipeline "$MODEL_ID" --download-only
    elif [ "$USE_CONTAINER" = "true" ]; then
        # Container mode - paths inside container
        docker exec -e HF_TOKEN="${HF_TOKEN:-}" "$CONTAINER_NAME" \
            python3 /llm/security/downloader.py \
                "$MODEL_ID" \
                --output-dir /llm/models/quarantine \
                --safetensors-only \
                $token_arg
    else
        # Local mode
        python3 "$SECURITY_MODULE_DIR/downloader.py" \
            "$MODEL_ID" \
            --output-dir "$QUARANTINE_DIR" \
            --safetensors-only \
            $token_arg
    fi
    
    log_success "Download complete: $QUARANTINE_DIR/$MODEL_NAME"
}

# =============================================================================
# SECURITY SCAN
# =============================================================================

security_scan() {
    log_header "Security Scan"
    
    local model_path="$QUARANTINE_DIR/$MODEL_NAME"
    local scan_file="$SCAN_RESULTS_DIR/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).json"
    
    log_info "Scanning: $model_path"
    
    if [ -n "$AI_SECURITY_IMAGE" ]; then
        # Use ai-security Docker image
        log_info "Using Docker image: $AI_SECURITY_IMAGE"
        if docker run --rm \
            -v "$MODELS_DIR:/srv/models/vLLM" \
            "$AI_SECURITY_IMAGE" \
            scan "/srv/models/vLLM/quarantine/$MODEL_NAME" \
            --output "/srv/models/vLLM/scan-results/$(basename "$scan_file")"; then
            log_success "Security scan PASSED"
            return 0
        else
            log_error "Security scan FAILED"
            return 1
        fi
    elif [ "$USE_CONTAINER" = "true" ]; then
        # Container mode - paths inside container
        if docker exec "$CONTAINER_NAME" \
            python3 /llm/security/scanner.py \
                "/llm/models/quarantine/$MODEL_NAME" \
                --output "/llm/models/scan-results/$(basename "$scan_file")"; then
            log_success "Security scan PASSED"
            return 0
        else
            log_error "Security scan FAILED"
            return 1
        fi
    else
        # Local mode
        if python3 "$SECURITY_MODULE_DIR/scanner.py" \
            "$model_path" \
            --output "$scan_file"; then
            log_success "Security scan PASSED"
            return 0
        else
            log_error "Security scan FAILED"
            log_info "Results: $scan_file"
            return 1
        fi
    fi
}

# =============================================================================
# PROMOTE TO VERIFIED
# =============================================================================

promote_model() {
    log_header "Promote to Verified"
    
    if [ -n "$AI_SECURITY_IMAGE" ] || [ "$USE_CONTAINER" = "true" ]; then
        # Docker mode - files owned by root, use sudo
        sudo rm -rf "$VERIFIED_DIR/$MODEL_NAME"
        sudo mv "$QUARANTINE_DIR/$MODEL_NAME" "$VERIFIED_DIR/$MODEL_NAME"
        sudo chown -R "$(id -u):$(id -g)" "$VERIFIED_DIR/$MODEL_NAME"
        chmod -R 755 "$VERIFIED_DIR/$MODEL_NAME"
    else
        rm -rf "$VERIFIED_DIR/$MODEL_NAME"
        mv "$QUARANTINE_DIR/$MODEL_NAME" "$VERIFIED_DIR/$MODEL_NAME"
    fi
    
    log_success "Model promoted: $VERIFIED_DIR/$MODEL_NAME"
}

# =============================================================================
# GENERATE MLBOM
# =============================================================================

generate_mlbom() {
    log_header "Generate MLBOM"
    
    local mlbom_model_id="$MODEL_ID"
    [ "$MOCK_MODE" = "true" ] && mlbom_model_id="mock/mock-gpt2"
    
    if [ -n "$AI_SECURITY_IMAGE" ]; then
        # Use ai-security Docker image
        docker run --rm \
            -v "$MODELS_DIR:/srv/models/vLLM" \
            "$AI_SECURITY_IMAGE" \
            mlbom "/srv/models/vLLM/verified/$MODEL_NAME" \
            --model-id "$mlbom_model_id" \
            --scan-passed
    elif [ "$USE_CONTAINER" = "true" ]; then
        docker exec "$CONTAINER_NAME" \
            python3 /llm/security/mlbom.py \
                "/llm/models/verified/$MODEL_NAME" \
                --model-id "$mlbom_model_id" \
                --scan-passed
    else
        python3 "$SECURITY_MODULE_DIR/mlbom.py" \
            "$VERIFIED_DIR/$MODEL_NAME" \
            --model-id "$mlbom_model_id" \
            --scan-passed
    fi
    
    log_success "MLBOM generated: $VERIFIED_DIR/$MODEL_NAME/mlbom.json"
}

# =============================================================================
# SERVE MODEL - Intel vLLM
# =============================================================================

serve_model() {
    log_header "Serve Model with vLLM"
    
    if [ "$USE_CONTAINER" != "true" ]; then
        log_error "Serving requires container mode (--container)"
        return 1
    fi
    
    local model_path="/llm/models/verified/$MODEL_NAME"
    
    log_info "Starting vLLM server..."
    log_info "Model: $model_path"
    log_info "Port: $VLLM_PORT"
    log_info "Dtype: $VLLM_DTYPE"
    
    # Serve model with vLLM (runs in background within container)
    docker exec -d "$CONTAINER_NAME" \
        vllm serve "$model_path" \
            --served-model-name "$MODEL_NAME" \
            --port "$VLLM_PORT" \
            --dtype="$VLLM_DTYPE" \
            --quantization "$VLLM_QUANTIZATION" \
            --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTIL" \
            --max-model-len "$VLLM_MAX_MODEL_LEN"
    
    # Wait for server to be ready
    log_info "Waiting for vLLM server to be ready..."
    local max_wait=120
    local waited=0
    
    while [ $waited -lt $max_wait ]; do
        if docker exec "$CONTAINER_NAME" curl -s "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
            log_success "vLLM server is ready!"
            log_info "Endpoint: http://localhost:$VLLM_PORT/v1"
            
            # Verify with curl requests
            verify_vllm_endpoint
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        echo -n "."
    done
    echo ""
    
    log_error "vLLM server did not start within ${max_wait}s"
    return 1
}

# =============================================================================
# VERIFY VLLM ENDPOINT - Curl verification tests
# =============================================================================

verify_vllm_endpoint() {
    log_header "Verify vLLM Endpoint"
    
    # Test 1: Health check
    log_info "Testing health endpoint..."
    if docker exec "$CONTAINER_NAME" curl -s "http://localhost:$VLLM_PORT/health" | grep -q "ok\|healthy"; then
        log_success "Health check passed"
    else
        log_warning "Health check returned unexpected response"
    fi
    
    # Test 2: List models
    log_info "Testing /v1/models endpoint..."
    local models_response
    models_response=$(docker exec "$CONTAINER_NAME" curl -s "http://localhost:$VLLM_PORT/v1/models")
    if echo "$models_response" | grep -q "$MODEL_NAME"; then
        log_success "Model '$MODEL_NAME' is available"
        echo "$models_response" | python3 -m json.tool 2>/dev/null | head -20 || echo "$models_response"
    else
        log_warning "Model not found in response"
        echo "$models_response"
    fi
    
    # Test 3: Simple completion request
    log_info "Testing /v1/completions endpoint..."
    local completion_response
    completion_response=$(docker exec "$CONTAINER_NAME" curl -s -X POST "http://localhost:$VLLM_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_NAME\",
            \"prompt\": \"Hello, world!\",
            \"max_tokens\": 10,
            \"temperature\": 0.7
        }")
    
    if echo "$completion_response" | grep -q "choices"; then
        log_success "Completion request successful"
        echo "$completion_response" | python3 -m json.tool 2>/dev/null | head -30 || echo "$completion_response"
    else
        log_warning "Completion request failed or unexpected response"
        echo "$completion_response"
    fi
    
    # Test 4: Chat completion (if supported)
    log_info "Testing /v1/chat/completions endpoint..."
    local chat_response
    chat_response=$(docker exec "$CONTAINER_NAME" curl -s -X POST "http://localhost:$VLLM_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_NAME\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}],
            \"max_tokens\": 10
        }")
    
    if echo "$chat_response" | grep -q "choices"; then
        log_success "Chat completion request successful"
        echo "$chat_response" | python3 -m json.tool 2>/dev/null | head -30 || echo "$chat_response"
    else
        log_warning "Chat completion not supported or failed (expected for some models)"
    fi
    
    log_success "Endpoint verification complete"
}

# =============================================================================
# RUN BENCHMARK - vLLM Benchmark
# =============================================================================

run_benchmark() {
    log_header "Run vLLM Benchmark"
    
    if [ "$USE_CONTAINER" != "true" ]; then
        log_error "Benchmark requires container mode (--container)"
        return 1
    fi
    
    local model_path="/llm/models/verified/$MODEL_NAME"
    
    log_info "Running vLLM benchmark..."
    log_info "Input length: 128, Output length: 128"
    log_info "Num prompts: 10"
    
    docker exec "$CONTAINER_NAME" \
        python3 /llm/vllm/benchmarks/benchmark_throughput.py \
            --model "$model_path" \
            --input-len 128 \
            --output-len 128 \
            --num-prompts 10 \
            --dtype "$VLLM_DTYPE" \
            --quantization "$VLLM_QUANTIZATION"
    
    log_success "Benchmark complete"
}

# =============================================================================
# CLEANUP
# =============================================================================

cleanup() {
    log_info "Cleaning up..."
    
    if [ -n "$AI_SECURITY_IMAGE" ] || [ "$USE_CONTAINER" = "true" ]; then
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
        # Files created by container/docker are owned by root
        sudo rm -rf "$MODELS_DIR" 2>/dev/null || true
    else
        rm -rf "$MODELS_DIR"
    fi
}

# =============================================================================
# MAIN
# =============================================================================

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          Model Security - E2E Integration Test                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL_ID"
echo "Mode: $([ "$MOCK_MODE" = "true" ] && echo "Mock" || echo "Real")"
echo "Docker Image: $([ -n "$AI_SECURITY_IMAGE" ] && echo "$AI_SECURITY_IMAGE" || echo "(local)")"
echo "Container: $([ "$USE_CONTAINER" = "true" ] && echo "Yes" || echo "No")"
echo "Serve: $([ "$DO_SERVE" = "true" ] && echo "Yes" || echo "No")"
echo "Benchmark: $([ "$DO_BENCHMARK" = "true" ] && echo "Yes" || echo "No")"
echo ""

# Trap cleanup
trap cleanup EXIT

# Docker image mode: run full pipeline via Docker
if [ -n "$AI_SECURITY_IMAGE" ]; then
    setup
    if run_docker_pipeline; then
        log_header "Test PASSED ✓"
        echo ""
        echo "Docker image pipeline completed successfully:"
        echo "  Image: $AI_SECURITY_IMAGE"
        echo "  1. ✓ Download to quarantine"
        echo "  2. ✓ Security scan"
        echo "  3. ✓ Promote to verified"
        echo "  4. ✓ Generate MLBOM"
        echo ""
        echo "Verified model: $VERIFIED_DIR/$MODEL_NAME"
        exit 0
    else
        log_header "Test FAILED"
        log_error "Docker pipeline failed"
        exit 1
    fi
fi

# Step 1: Setup directories and container
setup
[ "$USE_CONTAINER" = "true" ] && setup_container

# Step 2: Download or create mock model
if [ "$MOCK_MODE" = "true" ]; then
    create_mock_model
else
    download_model
fi

# Step 3-5: Security Pipeline
if security_scan; then
    promote_model
    generate_mlbom
    
    # Step 6: Serve model (optional)
    if [ "$DO_SERVE" = "true" ]; then
        serve_model
    fi
    
    # Step 7: Run benchmark (optional)
    if [ "$DO_BENCHMARK" = "true" ]; then
        run_benchmark
    fi
    
    log_header "Test PASSED ✓"
    echo ""
    echo "Security pipeline completed successfully:"
    echo "  1. ✓ Setup directories"
    echo "  2. ✓ Download to quarantine"
    echo "  3. ✓ Security scan"
    echo "  4. ✓ Promote to verified"
    echo "  5. ✓ Generate MLBOM"
    [ "$DO_SERVE" = "true" ] && echo "  6. ✓ vLLM serving started"
    [ "$DO_BENCHMARK" = "true" ] && echo "  7. ✓ Benchmark completed"
    echo ""
    echo "Verified model: $VERIFIED_DIR/$MODEL_NAME"
    echo "MLBOM: $VERIFIED_DIR/$MODEL_NAME/mlbom.json"
    [ "$DO_SERVE" = "true" ] && echo "vLLM endpoint: http://localhost:$VLLM_PORT/v1"
    
    EXIT_CODE=0
else
    log_header "Test: Security Blocked (Expected for malicious models)"
    log_error "Model failed security scan"
    echo "Model remains in: $QUARANTINE_DIR/$MODEL_NAME"
    EXIT_CODE=1
fi

exit $EXIT_CODE
