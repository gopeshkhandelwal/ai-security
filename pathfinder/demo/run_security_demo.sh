#!/bin/bash
#
# Pathfinder Security Demo Runner
#
# This script demonstrates the security scanning pipeline
# inside an Intel Gaudi Docker container.
#
# Usage:
#   ./run_security_demo.sh
#
# Or run manually:
#   docker run -it --runtime=habana \
#     -v $(pwd):/workspace \
#     vault.habana.ai/gaudi-docker/1.23.0/ubuntu22.04/habanalabs/pytorch-installer-2.9.0:latest \
#     /workspace/pathfinder/demo/run_demo_inside_container.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "============================================================"
echo "  Pathfinder Security Demo - Intel Gaudi Container"
echo "============================================================"
echo ""
echo "This demo will:"
echo "  1. Start a Gaudi Docker container"
echo "  2. Install security scanning tools"
echo "  3. Scan a malicious model (from Lab 01)"
echo "  4. Show the scanner catching the attack"
echo "  5. Demonstrate secure vs insecure loading"
echo ""
echo "Workspace: $WORKSPACE_DIR"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if habana runtime is available (optional)
RUNTIME_FLAG=""
if docker info 2>/dev/null | grep -q habana; then
    RUNTIME_FLAG="--runtime=habana"
    echo "✓ Habana runtime detected"
else
    echo "⚠️  Habana runtime not detected - running in CPU mode"
    echo "   (Scanning demo will still work, inference will use CPU)"
fi

echo ""
echo "Starting container..."
echo ""

# Run the demo inside the container
docker run -it --rm \
    $RUNTIME_FLAG \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    -e PYTHONPATH=/workspace \
    --cap-add=sys_nice \
    --ipc=host \
    -v "$WORKSPACE_DIR:/workspace:rw" \
    -w /workspace \
    vault.habana.ai/gaudi-docker/1.23.0/ubuntu22.04/habanalabs/pytorch-installer-2.9.0:latest \
    /bin/bash -c "
        echo '============================================================'
        echo '  Inside Gaudi Container'
        echo '============================================================'
        echo ''
        
        # Install required packages
        echo '[1/4] Installing security scanning tools...'
        pip install --quiet modelscan picklescan cryptography transformers safetensors
        
        echo '[2/4] Installing optimum-habana...'
        pip install --quiet optimum-habana 2>/dev/null || echo '  (optimum-habana install skipped - may not have Gaudi HW)'
        
        echo '[3/4] Running security scan demo...'
        echo ''
        cd /workspace
        python pathfinder/demo/security_scan_demo.py
        
        echo ''
        echo '[4/4] Demo complete!'
        echo ''
    "

echo ""
echo "============================================================"
echo "  Demo Complete"
echo "============================================================"
