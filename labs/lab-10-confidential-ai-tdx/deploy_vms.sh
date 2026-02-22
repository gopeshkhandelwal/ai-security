#!/bin/bash
# ============================================================================
# TDX vs Standard VM Deployment Script
# 
# This script creates TWO VMs on Google Cloud Platform:
# 1. TDX VM (tdx-vm) - Memory encrypted by Intel TDX hardware
# 2. Standard VM (standard-vm) - Memory in plaintext, no encryption
#
# Purpose: Demonstrate the security difference between encrypted and
#          unencrypted memory in cloud AI workloads.
#
# Author: GopeshK
# License: MIT
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration - UPDATE THESE VALUES
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"
REGION="${ZONE%-*}"

# VM Names
TDX_VM_NAME="tdx-vm"
STANDARD_VM_NAME="standard-vm"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} ${BOLD}$1${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
}

print_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found. Install from: https://cloud.google.com/sdk"
        exit 1
    fi
    print_success "gcloud CLI found"
}

check_project() {
    if [[ "$PROJECT_ID" == "your-project-id" ]]; then
        print_error "Please set your GCP project ID:"
        echo "  export GCP_PROJECT_ID=your-actual-project-id"
        echo "  Or edit this script and set PROJECT_ID"
        exit 1
    fi
    
    gcloud config set project "$PROJECT_ID" 2>/dev/null
    print_success "Project: $PROJECT_ID"
}

# ============================================================================
# VM Creation Functions
# ============================================================================

create_tdx_vm() {
    print_step "Creating TDX-enabled VM: $TDX_VM_NAME"
    
    # Check if VM already exists
    if gcloud compute instances describe "$TDX_VM_NAME" --zone="$ZONE" &>/dev/null; then
        print_warning "TDX VM '$TDX_VM_NAME' already exists. Skipping creation."
        return 0
    fi
    
    echo "Creating TDX Confidential VM with hardware memory encryption..."
    
    gcloud compute instances create "$TDX_VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type=c3-standard-4 \
        --confidential-compute-type=TDX \
        --min-cpu-platform="Intel Sapphire Rapids" \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=50GB \
        --boot-disk-type=pd-ssd \
        --maintenance-policy=TERMINATE \
        --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y python3-pip python3-venv git
echo "TDX VM setup complete" > /tmp/setup_done
'
    
    print_success "TDX VM created: $TDX_VM_NAME"
    echo -e "  ${GREEN}• Memory: AES-256 Hardware Encrypted${NC}"
    echo -e "  ${GREEN}• Hypervisor Access: BLOCKED${NC}"
    echo -e "  ${GREEN}• Attestation: Available${NC}"
}

create_standard_vm() {
    print_step "Creating Standard VM: $STANDARD_VM_NAME"
    
    # Check if VM already exists
    if gcloud compute instances describe "$STANDARD_VM_NAME" --zone="$ZONE" &>/dev/null; then
        print_warning "Standard VM '$STANDARD_VM_NAME' already exists. Skipping creation."
        return 0
    fi
    
    echo "Creating standard VM WITHOUT memory encryption..."
    
    gcloud compute instances create "$STANDARD_VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type=e2-standard-4 \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=50GB \
        --boot-disk-type=pd-ssd \
        --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y python3-pip python3-venv git
echo "Standard VM setup complete" > /tmp/setup_done
'
    
    print_success "Standard VM created: $STANDARD_VM_NAME"
    echo -e "  ${RED}• Memory: PLAINTEXT (No Encryption)${NC}"
    echo -e "  ${RED}• Hypervisor Access: FULL ACCESS${NC}"
    echo -e "  ${RED}• Attestation: Not Available${NC}"
}

create_firewall_rules() {
    print_step "Configuring firewall rules"
    
    # IAP SSH rule
    if ! gcloud compute firewall-rules describe allow-ssh-iap --project="$PROJECT_ID" &>/dev/null; then
        gcloud compute firewall-rules create allow-ssh-iap \
            --project="$PROJECT_ID" \
            --direction=INGRESS \
            --action=allow \
            --rules=tcp:22 \
            --source-ranges=35.235.240.0/20 \
            --description="Allow SSH via IAP tunnel"
        print_success "Created firewall rule: allow-ssh-iap"
    else
        print_warning "Firewall rule 'allow-ssh-iap' already exists"
    fi
}

# ============================================================================
# Display Functions
# ============================================================================

show_comparison() {
    print_header "VM Comparison: Memory Encryption"
    
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────────────┐"
    echo "│                       VM SECURITY COMPARISON                            │"
    echo "├───────────────────────────────┬───────────────────────────────────────┤"
    echo "│  TDX VM (tdx-vm)              │  Standard VM (standard-vm)            │"
    echo "├───────────────────────────────┼───────────────────────────────────────┤"
    echo "│  ✅ Memory: ENCRYPTED         │  ❌ Memory: PLAINTEXT                 │"
    echo "│  ✅ Hypervisor: BLOCKED       │  ❌ Hypervisor: FULL ACCESS           │"
    echo "│  ✅ Cloud Admin: NO ACCESS    │  ❌ Cloud Admin: CAN READ DATA        │"
    echo "│  ✅ Attestation: AVAILABLE    │  ❌ Attestation: NOT AVAILABLE        │"
    echo "│  ✅ Hardware Root of Trust    │  ❌ No Hardware Protection            │"
    echo "├───────────────────────────────┴───────────────────────────────────────┤"
    echo "│  Cost: c3-standard-4 ~\$0.25/hr │  Cost: e2-standard-4 ~\$0.13/hr        │"
    echo "└─────────────────────────────────────────────────────────────────────────┘"
    echo ""
}

show_next_steps() {
    print_header "Next Steps: Run the Demo"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 1: SSH into the TDX VM (encrypted memory)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  gcloud compute ssh $TDX_VM_NAME --zone=$ZONE --tunnel-through-iap"
    echo ""
    echo "  # Inside the VM:"
    echo "  cd ~ && git clone <your-repo> && cd ai-security/labs/lab-10-confidential-ai-tdx"
    echo "  python3 -m venv .venv && source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo "  python3 2_memory_comparison_demo.py"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 2: SSH into the Standard VM (plaintext memory)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  gcloud compute ssh $STANDARD_VM_NAME --zone=$ZONE --tunnel-through-iap"
    echo ""
    echo "  # Run the same demo - observe the DIFFERENCE in protection"
    echo "  python3 2_memory_comparison_demo.py"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  CLEANUP: Delete VMs when done"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  gcloud compute instances delete $TDX_VM_NAME --zone=$ZONE --quiet"
    echo "  gcloud compute instances delete $STANDARD_VM_NAME --zone=$ZONE --quiet"
    echo ""
}

# ============================================================================
# Main Script
# ============================================================================

main() {
    print_header "TDX vs Standard VM Deployment"
    
    echo "This script will create two VMs to demonstrate memory encryption:"
    echo "  1. TDX VM     - Intel TDX hardware-encrypted memory"
    echo "  2. Standard VM - No memory encryption (vulnerable)"
    echo ""
    
    # Pre-flight checks
    print_step "Checking prerequisites"
    check_gcloud
    check_project
    
    # Enable required APIs
    print_step "Enabling required GCP APIs"
    gcloud services enable compute.googleapis.com --quiet
    print_success "Compute API enabled"
    
    # Create firewall rules
    create_firewall_rules
    
    # Create VMs
    create_tdx_vm
    create_standard_vm
    
    # Wait for VMs to be ready
    print_step "Waiting for VMs to initialize..."
    sleep 30
    
    # Show comparison
    show_comparison
    
    # Show next steps
    show_next_steps
    
    print_success "Deployment complete!"
}

# ============================================================================
# Script Entry Point
# ============================================================================

case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Deploy TDX and Standard VMs for memory encryption comparison."
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --cleanup     Delete both VMs"
        echo "  --tdx-only    Create only the TDX VM"
        echo "  --std-only    Create only the Standard VM"
        echo ""
        echo "Environment Variables:"
        echo "  GCP_PROJECT_ID  Your GCP project ID (required)"
        echo "  GCP_ZONE        GCP zone (default: us-central1-a)"
        ;;
    --cleanup)
        print_header "Cleaning up VMs"
        gcloud compute instances delete "$TDX_VM_NAME" --zone="$ZONE" --quiet 2>/dev/null || true
        gcloud compute instances delete "$STANDARD_VM_NAME" --zone="$ZONE" --quiet 2>/dev/null || true
        print_success "VMs deleted"
        ;;
    --tdx-only)
        check_gcloud
        check_project
        create_firewall_rules
        create_tdx_vm
        ;;
    --std-only)
        check_gcloud
        check_project
        create_firewall_rules
        create_standard_vm
        ;;
    *)
        main
        ;;
esac
