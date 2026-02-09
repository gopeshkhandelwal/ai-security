#!/bin/bash
# Run Confidential AI Inference inside SGX Enclave using Gramine
#
# Prerequisites:
#   1. Intel SGX driver installed (/dev/sgx_enclave exists)
#   2. Gramine installed: sudo apt install gramine
#   3. SGX signing key generated
#
# Author: GopeshK
# License: MIT

# Ensure script is run with bash, not sh
if [ -z "$BASH_VERSION" ]; then
    echo "Error: This script must be run with bash, not sh."
    echo "Usage: bash $0 or ./$0"
    exit 1
fi

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Preserve proxy settings for sudo commands
SUDO_PROXY=""
if [ -n "$http_proxy" ] || [ -n "$https_proxy" ]; then
    SUDO_PROXY="http_proxy=$http_proxy https_proxy=$https_proxy no_proxy=$no_proxy"
    echo "[*] Proxy detected: $http_proxy"
fi

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║      🔒 SGX Enclave Setup for Confidential AI Inference 🔒            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check for SGX device
if [ ! -e /dev/sgx_enclave ] && [ ! -e /dev/sgx/enclave ]; then
    echo "❌ ERROR: SGX enclave device not found!"
    echo "   Make sure SGX is enabled in BIOS and driver is installed."
    echo ""
    echo "   To install SGX driver:"
    echo "   sudo apt install linux-image-generic  # Kernel 5.11+ has in-kernel driver"
    exit 1
fi

echo "✓ SGX enclave device found"

# Check for Gramine
if ! command -v gramine-sgx &> /dev/null; then
    echo ""
    echo "❌ Gramine not installed."
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│  Gramine is required to run Python inside an SGX enclave.       │"
    echo "│                                                                  │"
    echo "│  Installation options:                                           │"
    echo "│                                                                  │"
    echo "│  Option 1: From Ubuntu packages (if available):                  │"
    echo "│    sudo apt update && sudo apt install gramine                   │"
    echo "│                                                                  │"
    echo "│  Option 2: From Gramine repo (requires internet):                │"
    echo "│    sudo curl -fsSLo /usr/share/keyrings/gramine-keyring.gpg \\   │"
    echo "│      https://packages.gramineproject.io/gramine-keyring.gpg      │"
    echo "│    echo 'deb [arch=amd64 signed-by=...] ...' | sudo tee ...      │"
    echo "│    sudo apt update && sudo apt install gramine                   │"
    echo "│                                                                  │"
    echo "│  Option 3: Build from source:                                    │"
    echo "│    git clone https://github.com/gramineproject/gramine.git       │"
    echo "│    cd gramine && meson setup build/ && ninja -C build/           │"
    echo "│                                                                  │"
    echo "│  See: https://gramine.readthedocs.io/en/latest/installation.html │"
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo ""
    
    # Try to install from Gramine repo (with proxy support)
    echo "[*] Attempting to install Gramine from official repo..."
    
    # Download and import keyring properly (handle both armored and binary formats)
    echo "[*] Downloading and importing Gramine GPG key..."
    KEYRING_FILE="/usr/share/keyrings/gramine-keyring.gpg"
    if [ -n "$SUDO_PROXY" ]; then
        sudo env $SUDO_PROXY curl -fsSL https://packages.gramineproject.io/gramine-keyring.gpg \
            | sudo gpg --dearmor -o "$KEYRING_FILE" 2>/dev/null \
            || sudo env $SUDO_PROXY curl -fsSLo "$KEYRING_FILE" https://packages.gramineproject.io/gramine-keyring.gpg
    else
        sudo curl -fsSL https://packages.gramineproject.io/gramine-keyring.gpg \
            | sudo gpg --dearmor -o "$KEYRING_FILE" 2>/dev/null \
            || sudo curl -fsSLo "$KEYRING_FILE" https://packages.gramineproject.io/gramine-keyring.gpg
    fi
    
    # Verify key was imported
    if [ ! -s "$KEYRING_FILE" ]; then
        echo "❌ Failed to download Gramine GPG key"
        exit 1
    fi
    
    # Add repo - Use noble (24.04) if current distro not supported
    DISTRO=$(lsb_release -sc 2>/dev/null || echo "noble")
    # Gramine only supports up to noble (24.04), fallback for newer Ubuntu
    case "$DISTRO" in
        plucky|oracular|mantic)
            echo "[*] Ubuntu $DISTRO not yet supported by Gramine, using noble (24.04) packages"
            DISTRO="noble"
            ;;
    esac
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/gramine-keyring.gpg] https://packages.gramineproject.io/ $DISTRO main" \
        | sudo tee /etc/apt/sources.list.d/gramine.list
    
    # Update and install with proxy
    echo "[*] Installing Gramine..."
    if [ -n "$SUDO_PROXY" ]; then
        sudo env $SUDO_PROXY apt-get update
        sudo env $SUDO_PROXY apt-get install -y gramine
    else
        sudo apt-get update
        sudo apt-get install -y gramine
    fi
    
    if ! command -v gramine-sgx &> /dev/null; then
        echo ""
        echo "❌ Gramine installation failed."
        echo ""
        echo "Please install Gramine manually using one of the options above,"
        echo "then re-run this script."
        exit 1
    fi
    
    echo "✓ Gramine installed successfully"
fi

echo "✓ Gramine available"

# Generate signing key if not exists
SGX_KEY="$HOME/.config/gramine/enclave-key.pem"
if [ ! -f "$SGX_KEY" ]; then
    echo ""
    echo "[*] Generating SGX signing key..."
    mkdir -p "$(dirname "$SGX_KEY")"
    gramine-sgx-gen-private-key -f
    echo "✓ SGX signing key generated: $SGX_KEY"
fi

# Generate manifest
echo ""
echo "[*] Generating Gramine manifest..."

ARCH_LIBDIR=/lib/x86_64-linux-gnu
PYTHON_PATH=$(which python3)
EXECDIR=$(dirname "$PYTHON_PATH")

gramine-manifest \
    -Dlog_level=error \
    -Darch_libdir="$ARCH_LIBDIR" \
    -Dexecdir="$EXECDIR" \
    gramine_manifest.template > python.manifest

echo "✓ Manifest generated: python.manifest"

# Sign the manifest
echo ""
echo "[*] Signing enclave..."
gramine-sgx-sign --manifest python.manifest --output python.manifest.sgx

echo "✓ Enclave signed: python.manifest.sgx"

# Run inside SGX enclave
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║      🟢 Launching Python inside SGX Enclave...                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Set environment to indicate real SGX mode
export SGX_ENCLAVE=true

gramine-sgx python 4b_confidential_inference.py

echo ""
echo "✓ Enclave execution complete"
