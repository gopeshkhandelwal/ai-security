# Lab 08: TPM Attestation for AI Models

## Overview

This lab demonstrates hardware-rooted attestation for AI models using:
- **Intel TPM 2.0**: Hardware security chip for cryptographic measurements
- **Intel Trust Authority**: Cloud-based attestation verification service

## Why TPM Attestation for AI?

Traditional software-based security can be bypassed by attackers with system access. TPM attestation provides:

1. **Hardware Root of Trust**: TPM is a tamper-resistant chip - measurements cannot be faked
2. **Model Integrity Binding**: Model hash is cryptographically bound to platform boot state
3. **Remote Verification**: Third parties can verify model integrity without accessing the model
4. **Non-repudiation**: Attestation tokens prove model state at a specific time

## Attack Scenario

Without attestation, an attacker could:
1. Replace a production model with a trojaned version
2. The system would happily run the malicious model
3. No way to detect the tampering

With TPM attestation:
1. Model hash is measured into TPM PCR
2. Any model change = different PCR value
3. Attestation fails = inference blocked

## Prerequisites

**IMPORTANT:** This lab uses REAL Intel TPM 2.0 hardware for all cryptographic operations.

### Hardware Requirements
- Intel platform with TPM 2.0 (most modern Intel systems) - **REQUIRED**

### Software Requirements
- Linux with `tpm2-tools` installed - **REQUIRED**
- Intel Trust Authority API key - *Optional* (ITA response simulated if not provided)
- Python 3.8+
- TensorFlow (for model creation/inference)

### What's Real vs Simulated

| Component | Real Hardware | Notes |
|-----------|--------------|-------|
| TPM Device | ✅ Yes | `/dev/tpm0` |
| PCR Extension | ✅ Yes | Real `tpm2_pcrextend` |
| Quote Generation | ✅ Yes | Real `tpm2_quote` with AIK |
| Intel Trust Authority | Optional | Simulated if no API key |

### System Setup

```bash
# 1. Check if TPM device exists
ls -la /dev/tpm*

# 2. Check TPM version (must be 2)
cat /sys/class/tpm/tpm0/tpm_version_major

# 3. Install tpm2-tools
sudo apt install tpm2-tools    # Debian/Ubuntu
# or: sudo dnf install tpm2-tools  # Fedora/RHEL

# 4. Add user to tss group (required for TPM access)
sudo usermod -aG tss $USER

# 5. Apply group change (or log out and back in)
newgrp tss

# 6. Set Intel Trust Authority API key
export INTEL_TRUST_AUTHORITY_API_KEY="your-key-here"
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Verify Setup

```bash
python 0_check_tpm.py
```

All checks must pass before proceeding.

## Lab Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TPM Model Attestation Flow                       │
└─────────────────────────────────────────────────────────────────────┘

  1. CHECK TPM            2. MEASURE MODEL         3. GENERATE QUOTE
  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
  │ TPM 2.0     │         │ model.h5    │         │ TPM Quote   │
  │ Available?  │────────▶│     │       │────────▶│ + Signature │
  │ Version?    │         │     ▼       │         │ + Nonce     │
  └─────────────┘         │ PCR[14]     │         └─────────────┘
                          │ extend(hash)│                 │
                          └─────────────┘                 │
                                                          ▼
  4. ATTEST (Intel Trust Authority)      5. SECURE INFERENCE
  ┌─────────────────────────────────┐   ┌──────────────────────────┐
  │ POST quote to ITA               │   │ Verify token             │
  │         │                       │   │ Check expiration         │
  │         ▼                       │──▶│ Verify model hash        │
  │ Receive JWT attestation token   │   │         │                │
  │ (signed by Intel)               │   │         ▼                │
  └─────────────────────────────────┘   │ Run inference (if valid) │
                                        └──────────────────────────┘
```

## Running the Lab

### Step 0: Check TPM Hardware

```bash
python 0_check_tpm.py
```

Checks:
- TPM device exists (`/dev/tpm0`, `/dev/tpmrm0`)
- TPM version (must be 2.0)
- User permissions
- tpm2-tools availability
- Basic TPM functionality

### Step 1: Measure Model into TPM

```bash
python 1_measure_model.py [model_path]
```

This script:
1. Computes SHA256 hash of the model file
2. Extends TPM PCR[14] with the hash
3. Saves measurement record

**Technical Detail**: PCR extension is a one-way operation:
```
PCR_new = SHA256(PCR_old || model_hash)
```

### Step 2: Generate TPM Quote

```bash
python 2_generate_quote.py
```

Creates a TPM quote containing:
- Selected PCR values (including model measurement)
- Nonce (prevents replay attacks)
- TPM signature (using Attestation Identity Key)

### Step 3: Send to Intel Trust Authority

```bash
python 3_attest_model.py
```

Options:
1. **With API Key**: Sends to real Intel Trust Authority
2. **Without API Key**: Simulates the response (for demo)

To use real attestation:
```bash
export INTEL_TRUST_AUTHORITY_API_KEY="your-key-here"
# or
echo "your-key-here" > ~/.intel_trust_authority_key
```

Returns a signed JWT token containing:
- Trust evaluation result
- Model hash binding
- Expiration time

### Step 4: Secure Inference

```bash
python 4_secure_inference.py
```

Before running inference:
1. Verifies attestation token is valid
2. Checks token hasn't expired
3. Verifies current model hash matches attested hash
4. Only runs inference if all checks pass

Also demonstrates tampering detection.

## Security Guarantees

| Threat | Protection |
|--------|-----------|
| Model replacement | Hash mismatch detected |
| Model tampering | PCR value changes, attestation fails |
| Replay attacks | Nonce ensures freshness |
| Fake attestation | JWT signed by Intel, verifiable |
| Token reuse | Expiration + model hash binding |

## Technical Details

### Platform Configuration Registers (PCRs)

PCRs are special TPM registers that can only be extended, never directly written:
- PCR[0-7]: Platform firmware
- PCR[8-13]: OS boot
- **PCR[14-15]**: Application use (we use PCR[14] for model)

### Intel Trust Authority

Intel Trust Authority is a cloud service that verifies attestation evidence:
- Validates TPM quotes
- Checks TCB (Trusted Computing Base) status
- Issues signed attestation tokens
- Supports TDX, SGX, and TPM attestation

API Documentation: https://www.intel.com/trustauthority

## Files

| File | Description |
|------|-------------|
| `0_check_tpm.py` | TPM hardware availability check |
| `1_measure_model.py` | Extend PCR with model hash |
| `2_generate_quote.py` | Generate signed TPM quote |
| `3_attest_model.py` | Send to Intel Trust Authority |
| `4_secure_inference.py` | Verified inference |
| `requirements.txt` | Python dependencies |
| `reset.py` | Clean up generated files |

## Generated Files

After running the lab:
- `.tpm_status.json` - TPM availability status
- `test_model.h5` - Sample model (if created)
- `model_measurement.json` - Measurement record
- `attestation_package.json` - Quote package
- `attestation_result.json` - Attestation result
- `attestation_token.jwt` - JWT token
- `aik.pub`, `aik.priv`, `aik.ctx` - Attestation keys

## Troubleshooting

### "Permission denied" on /dev/tpm0

```bash
# Add user to tss group
sudo usermod -aG tss $USER
# Re-login or
newgrp tss
```

### TPM resource manager busy

```bash
# Use /dev/tpmrm0 instead of /dev/tpm0
export TPM2TOOLS_TCTI="device:/dev/tpmrm0"
```

### Missing tpm2-tools

```bash
sudo apt install tpm2-tools    # Debian/Ubuntu
sudo dnf install tpm2-tools    # Fedora/RHEL
```

## Further Reading

- [Intel Trust Authority Documentation](https://www.intel.com/trustauthority)
- [TPM 2.0 Specification](https://trustedcomputinggroup.org/resource/tpm-library-specification/)
- [tpm2-tools Documentation](https://github.com/tpm2-software/tpm2-tools)
- [NIST Special Publication 800-155: BIOS Integrity Measurement](https://csrc.nist.gov/publications/detail/sp/800-155/archive/2011-12-01)

## Author

GopeshK

## License

MIT License - Educational purposes only.

## Disclaimer

This is a demonstration lab. For production use:
- Obtain proper Intel Trust Authority credentials
- Implement proper key management
- Add additional security layers (secure boot, etc.)
- Follow your organization's security policies
