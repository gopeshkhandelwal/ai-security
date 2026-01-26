# AI Model Pathfinder - Integration Design Document

**Version:** 1.0  
**Author:** Security Architecture Team  
**Date:** January 2026  
**Status:** Draft for Review

---

## Executive Summary

This document describes how AI Model Pathfinder integrates security controls into the existing Intel Gaudi model enablement workflow. The design ensures **zero-trust model handling** while maintaining developer productivity and Intel hardware optimization benefits.

---

## 1. Current Workflow (As-Is)

### Developer Experience Today

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CURRENT WORKFLOW (No Security Controls)                  │
└─────────────────────────────────────────────────────────────────────────────┘

  Developer                    Docker Host                   Hugging Face
     │                              │                              │
     │  1. docker run --runtime=habana vault.habana.ai/...        │
     │─────────────────────────────▶│                              │
     │                              │                              │
     │  2. pip install deepspeed    │                              │
     │─────────────────────────────▶│                              │
     │                              │                              │
     │  3. git clone Model-References                              │
     │─────────────────────────────▶│                              │
     │                              │                              │
     │  4. model.from_pretrained("org/model", trust_remote_code=True)
     │─────────────────────────────▶│─────────────────────────────▶│
     │                              │◀─────────────────────────────│
     │                              │  (Downloads model + code)    │
     │                              │                              │
     │  5. deepspeed cifar10.py     │  ⚠️ ARBITRARY CODE EXECUTES  │
     │─────────────────────────────▶│                              │
     │                              │                              │
```

### Current Command Sequence

```bash
# Step 1: Run Intel Gaudi Docker (privileged, host network)
docker run -it --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --cap-add=sys_nice --net=host --ipc=host \
  vault.habana.ai/gaudi-docker/1.23.0/ubuntu22.04/habanalabs/pytorch-installer-2.9.0:latest

# Step 2: Install DeepSpeed
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.23.0

# Step 3: Clone Model References
git clone https://github.com/HabanaAI/Model-References.git
cd Model-References/PyTorch/examples/DeepSpeed/cifar_example/
pip install -r requirements.txt

# Step 4: Set environment
export PYTHONPATH=$PYTHONPATH:Model-References
export PYTHON=/usr/bin/python3.10

# Step 5: Execute (NO SECURITY CHECKS!)
deepspeed --num_nodes=1 --num_gpus=8 cifar10_deepspeed.py \
  --deepspeed --deepspeed_config ds_config.json
```

### Security Gaps in Current Workflow

| Gap | Risk | MITRE ATLAS |
|-----|------|-------------|
| No model verification | Malicious model weights/code | AML.T0010 |
| `trust_remote_code=True` | Arbitrary code execution | AML.T0011 |
| `--net=host` exposure | Network exfiltration | AML.T0024 |
| No provenance tracking | Supply chain compromise | AML.T0010 |
| Privileged container | Container escape | - |
| No output sanitization | Secret leakage | - |

---

## 2. Proposed Workflow (To-Be with Pathfinder)

### Secure Developer Experience

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               PATHFINDER-SECURED WORKFLOW (Defense in Depth)                 │
└─────────────────────────────────────────────────────────────────────────────┘

  Developer          Pathfinder CLI         Secure Container        Hugging Face
     │                     │                       │                      │
     │  1. pathfinder enable "meta-llama/Llama-3"  │                      │
     │────────────────────▶│                       │                      │
     │                     │                       │                      │
     │              ┌──────┴──────┐                │                      │
     │              │ PREFLIGHT   │                │                      │
     │              │ • Allowlist │                │                      │
     │              │ • GPG verify│                │                      │
     │              │ • Format ok │                │                      │
     │              └──────┬──────┘                │                      │
     │                     │                       │                      │
     │              ┌──────┴──────┐                │                      │
     │              │  DOWNLOAD   │────────────────│─────────────────────▶│
     │              │  (isolated) │◀───────────────│◀─────────────────────│
     │              └──────┬──────┘                │                      │
     │                     │                       │                      │
     │              ┌──────┴──────┐                │                      │
     │              │   SCAN      │                │                      │
     │              │ • ModelScan │                │                      │
     │              │ • AST scan  │                │                      │
     │              │ • Bandit    │                │                      │
     │              │ • Presidio  │                │                      │
     │              └──────┬──────┘                │                      │
     │                     │                       │                      │
     │              ┌──────┴──────┐                │                      │
     │              │   SIGN      │                │                      │
     │              │ • ECDSA     │                │                      │
     │              │ • MLBOM gen │                │                      │
     │              └──────┬──────┘                │                      │
     │                     │                       │                      │
     │                     │  2. Launch secure container                  │
     │                     │──────────────────────▶│                      │
     │                     │                       │                      │
     │                     │  3. Verify signature  │                      │
     │                     │──────────────────────▶│                      │
     │                     │                       │                      │
     │  4. Interactive session (sandboxed)         │                      │
     │────────────────────────────────────────────▶│                      │
     │                     │                       │                      │
     │  5. result.json + enablement_report.html    │                      │
     │◀────────────────────│◀──────────────────────│                      │
```

---

## 3. Integration Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI MODEL PATHFINDER ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           PATHFINDER CLI                               │ │
│  │  $ pathfinder enable <model> [--approve-remote-code] [--skip-scan]    │ │
│  └───────────────────────────────────┬────────────────────────────────────┘ │
│                                      │                                      │
│  ┌───────────────────────────────────▼────────────────────────────────────┐ │
│  │                        ORCHESTRATION LAYER                             │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│  │  │ plan.json   │ │ Workflow    │ │ Approval    │ │ Audit       │      │ │
│  │  │ Generator   │ │ Engine      │ │ Manager     │ │ Logger      │      │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘      │ │
│  └───────────────────────────────────┬────────────────────────────────────┘ │
│                                      │                                      │
│  ┌───────────────────────────────────▼────────────────────────────────────┐ │
│  │                         SECURITY PIPELINE                              │ │
│  │                                                                        │ │
│  │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐ │ │
│  │  │  PREFLIGHT       │    │  ANALYSIS        │    │  VERIFICATION    │ │ │
│  │  │  ──────────────  │    │  ──────────────  │    │  ──────────────  │ │ │
│  │  │  • Allowlist     │───▶│  • ModelScan     │───▶│  • ECDSA Sign    │ │ │
│  │  │  • GPG Verify    │    │  • AST Scanner   │    │  • Hash Registry │ │ │
│  │  │  • Format Check  │    │  • Bandit        │    │  • MLBOM Gen     │ │ │
│  │  │  • Repo Origin   │    │  • Presidio      │    │  • Dependency    │ │ │
│  │  │                  │    │  • PickleScan    │    │    Lock          │ │ │
│  │  │                  │    │                  │    │                  │ │ │
│  │  └──────────────────┘    └──────────────────┘    └──────────────────┘ │ │
│  │                                                                        │ │
│  └───────────────────────────────────┬────────────────────────────────────┘ │
│                                      │                                      │
│  ┌───────────────────────────────────▼────────────────────────────────────┐ │
│  │                      CONTAINER ISOLATION LAYER                         │ │
│  │                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │                 PATHFINDER SECURE CONTAINER                      │ │ │
│  │  │  ┌────────────────────────────────────────────────────────────┐  │ │ │
│  │  │  │  Base: vault.habana.ai/gaudi-docker/1.23.0/...             │  │ │ │
│  │  │  │  + Pathfinder Security Agent                                │  │ │ │
│  │  │  │  + Pre-verified Dependencies (version-locked)               │  │ │ │
│  │  │  └────────────────────────────────────────────────────────────┘  │ │ │
│  │  │                                                                  │ │ │
│  │  │  Security Constraints:                                           │ │ │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐    │ │ │
│  │  │  │ No --net=  │ │ Read-only  │ │ CPU/GPU    │ │ Seccomp    │    │ │ │
│  │  │  │ host       │ │ rootfs     │ │ Limits     │ │ Profile    │    │ │ │
│  │  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘    │ │ │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐    │ │ │
│  │  │  │ No cap-add │ │ Timeout    │ │ Egress     │ │ Secret     │    │ │ │
│  │  │  │ dangerous  │ │ Watchdog   │ │ Allowlist  │ │ Injection  │    │ │ │
│  │  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘    │ │ │
│  │  │                                                                  │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         OUTPUT ARTIFACTS                               │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│  │  │ plan.json   │ │ result.json │ │ mlbom.json  │ │ report.html │      │ │
│  │  │ (pre-run)   │ │ (post-run)  │ │ (provenance)│ │ (summary)   │      │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Workflow Stages Detail

### Stage 1: Preflight Validation

**Trigger:** `pathfinder enable <model-id>`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STAGE 1: PREFLIGHT                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: model_id (e.g., "meta-llama/Llama-3-8B")                            │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CHECK 1: Allowlist Verification                                        │ │
│  │ ──────────────────────────────────                                     │ │
│  │ • Is model in approved allowlist?                                      │ │
│  │ • Is organization verified? (TRUSTED_PUBLISHERS list)                  │ │
│  │                                                                        │ │
│  │ if model in ALLOWLIST:                                                 │ │
│  │     status = "pre-approved"                                            │ │
│  │ elif org in TRUSTED_PUBLISHERS:                                        │ │
│  │     status = "trusted-org"                                             │ │
│  │ else:                                                                  │ │
│  │     status = "requires-approval"                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CHECK 2: Repository Verification                                       │ │
│  │ ───────────────────────────────────                                    │ │
│  │ • Fetch repo metadata from Hugging Face API                            │ │
│  │ • Verify GPG-signed commits (if required)                              │ │
│  │ • Check for security advisories                                        │ │
│  │                                                                        │ │
│  │ GET https://huggingface.co/api/models/{model_id}                       │ │
│  │ → Verify: lastModified, author, gated, private, library_name           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CHECK 3: Format Validation                                             │ │
│  │ ────────────────────────────                                           │ │
│  │ • List files in repository                                             │ │
│  │ • Reject pickle-based formats (.pkl, .bin with pickle)                 │ │
│  │ • Require .safetensors for v1                                          │ │
│  │                                                                        │ │
│  │ ALLOWED_FORMATS = [".safetensors", ".json", ".txt", ".md"]             │ │
│  │ BLOCKED_FORMATS = [".pkl", ".pickle", ".pt", ".pth", ".bin"]           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CHECK 4: trust_remote_code Detection                                   │ │
│  │ ──────────────────────────────────────                                 │ │
│  │ • Parse config.json for auto_map entries                               │ │
│  │ • Flag if custom code required                                         │ │
│  │ • Require two-person approval if detected                              │ │
│  │                                                                        │ │
│  │ if config.get("auto_map"):                                              │ │
│  │     plan["trust_remote_code_required"] = True                          │ │
│  │     plan["approval_level"] = "two-person"                              │ │
│  │     plan["risk_warnings"] = [...]                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  Output: plan.json                                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ {                                                                      │ │
│  │   "model_id": "meta-llama/Llama-3-8B",                                 │ │
│  │   "preflight_status": "passed",                                        │ │
│  │   "trust_remote_code_required": false,                                 │ │
│  │   "format_check": "safetensors-only",                                  │ │
│  │   "org_verified": true,                                                │ │
│  │   "approval_level": "auto",                                            │ │
│  │   "estimated_vram_gb": 16,                                             │ │
│  │   "recommended_gaudi_count": 1,                                        │ │
│  │   "next_stage": "download"                                             │ │
│  │ }                                                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 2: Secure Download & Scan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: DOWNLOAD & SCAN                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ DOWNLOAD (Isolated Network Context)                                    │ │
│  │ ─────────────────────────────────────                                  │ │
│  │                                                                        │ │
│  │ • Download to quarantine directory (not model cache)                   │ │
│  │ • Verify checksums against Hugging Face API                            │ │
│  │ • HF_TOKEN injected via secret mount (never in env/logs)               │ │
│  │                                                                        │ │
│  │ Allowed Egress:                                                        │ │
│  │   - huggingface.co                                                     │ │
│  │   - cdn-lfs.huggingface.co                                             │ │
│  │   - (blocked: everything else)                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ SCAN 1: ModelScan (Protect AI)                                         │ │
│  │ ─────────────────────────────────────────────                          │ │
│  │                                                                        │ │
│  │ from modelscan.modelscan import ModelScan                              │ │
│  │ scanner = ModelScan()                                                  │ │
│  │ results = scanner.scan(quarantine_path)                                │ │
│  │                                                                        │ │
│  │ Detects:                                                               │ │
│  │   - Pickle code execution                                              │ │
│  │   - Unsafe deserialization                                             │ │
│  │   - Embedded executables                                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ SCAN 2: AST Code Analysis                                              │ │
│  │ ───────────────────────────────────────────                            │ │
│  │                                                                        │ │
│  │ from model_security_scanner import ModelSecurityScanner                │ │
│  │ scanner = ModelSecurityScanner(quarantine_path)                        │ │
│  │ scanner.scan()                                                         │ │
│  │                                                                        │ │
│  │ Detects (SUSPICIOUS_PATTERNS):                                         │ │
│  │   - socket.socket          (Network creation)                          │ │
│  │   - os.fork()              (Process forking)                           │ │
│  │   - subprocess             (Command execution)                         │ │
│  │   - exec()/eval()          (Dynamic code)                              │ │
│  │   - base64.b64decode       (Obfuscation)                               │ │
│  │   - __import__             (Dynamic imports)                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ SCAN 3: Dependency Audit                                               │ │
│  │ ──────────────────────────                                             │ │
│  │                                                                        │ │
│  │ • Parse requirements.txt from model repo                               │ │
│  │ • Run: pip-audit, safety check                                         │ │
│  │ • Verify against locked versions                                       │ │
│  │ • Run: bandit on any .py files                                         │ │
│  │                                                                        │ │
│  │ LOCKED_DEPENDENCIES = {                                                │ │
│  │   "transformers": "4.36.0",                                            │ │
│  │   "torch": "2.1.0",                                                    │ │
│  │   "safetensors": "0.4.0"                                               │ │
│  │ }                                                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ SCAN 4: Secret Detection (Presidio)                                   │ │
│  │ ──────────────────────────────────────────                             │ │
│  │                                                                        │ │
│  │ from presidio_analyzer import AnalyzerEngine                           │ │
│  │ analyzer = AnalyzerEngine()                                            │ │
│  │                                                                        │ │
│  │ # Scan any text files for embedded secrets                             │ │
│  │ results = analyzer.analyze(text, entities=[                            │ │
│  │   "CREDIT_CARD", "CRYPTO", "API_KEY", "PASSWORD"                       │ │
│  │ ])                                                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  Decision Gate:                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ if any_scan_failed:                                                    │ │
│  │     quarantine_model()                                                 │ │
│  │     alert_security_team()                                              │ │
│  │     return ScanResult.BLOCKED                                          │ │
│  │ else:                                                                  │ │
│  │     promote_to_verified()                                              │ │
│  │     return ScanResult.PASSED                                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 3: Sign & Generate MLBOM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 3: SIGN & GENERATE MLBOM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Uses ECDSA Model Signing for artifact integrity verification               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ SIGN MODEL ARTIFACTS                                                   │ │
│  │ ──────────────────────                                                 │ │
│  │                                                                        │ │
│  │ from cryptography.hazmat.primitives.asymmetric import ec               │ │
│  │                                                                        │ │
│  │ for artifact in model_artifacts:                                       │ │
│  │     # Compute hash                                                     │ │
│  │     sha256 = hashlib.sha256(artifact.read()).hexdigest()               │ │
│  │                                                                        │ │
│  │     # Sign with org ECDSA key                                          │ │
│  │     signature = private_key.sign(artifact, ec.ECDSA(SHA256()))         │ │
│  │                                                                        │ │
│  │     # Store signature alongside artifact                               │ │
│  │     save_signature(f"{artifact}.sig", signature)                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ GENERATE MLBOM (Machine Learning Bill of Materials)                    │ │
│  │ ───────────────────────────────────────────────────                    │ │
│  │                                                                        │ │
│  │ mlbom.json:                                                            │ │
│  │ {                                                                      │ │
│  │   "mlbom_version": "1.0",                                              │ │
│  │   "generated_at": "2026-01-26T10:00:00Z",                              │ │
│  │   "model": {                                                           │ │
│  │     "id": "meta-llama/Llama-3-8B",                                     │ │
│  │     "version": "main",                                                 │ │
│  │     "commit": "abc123def456"                                           │ │
│  │   },                                                                   │ │
│  │   "provenance": {                                                      │ │
│  │     "source": "huggingface.co",                                        │ │
│  │     "download_timestamp": "2026-01-26T09:55:00Z",                      │ │
│  │     "gpg_verified": true,                                              │ │
│  │     "scanner_versions": {                                              │ │
│  │       "modelscan": "0.5.0",                                            │ │
│  │       "pathfinder_scanner": "1.0.0"                                    │ │
│  │     }                                                                  │ │
│  │   },                                                                   │ │
│  │   "artifacts": [                                                       │ │
│  │     {                                                                  │ │
│  │       "file": "model.safetensors",                                     │ │
│  │       "sha256": "abc123...",                                           │ │
│  │       "signature": "base64:...",                                       │ │
│  │       "size_bytes": 16000000000                                        │ │
│  │     }                                                                  │ │
│  │   ],                                                                   │ │
│  │   "dependencies": {                                                    │ │
│  │     "transformers": {"version": "4.36.0", "locked": true},             │ │
│  │     "torch": {"version": "2.1.0", "locked": true},                     │ │
│  │     "habana-frameworks": {"version": "1.23.0", "locked": true}         │ │
│  │   },                                                                   │ │
│  │   "security": {                                                        │ │
│  │     "trust_remote_code": false,                                        │ │
│  │     "scan_passed": true,                                               │ │
│  │     "format": "safetensors"                                            │ │
│  │   }                                                                    │ │
│  │ }                                                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 4: Isolated Execution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 4: ISOLATED EXECUTION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CURRENT (Insecure):                                                        │
│  ───────────────────                                                        │
│  docker run -it --runtime=habana \                                          │
│    --cap-add=sys_nice \                                                     │
│    --net=host \               # ⚠️ Full network access                      │
│    --ipc=host \               # ⚠️ Shared memory with host                  │
│    -e HABANA_VISIBLE_DEVICES=all \                                          │
│    vault.habana.ai/gaudi-docker/...                                         │
│                                                                              │
│  PATHFINDER (Secure):                                                       │
│  ────────────────────                                                       │
│  docker run -it --runtime=habana \                                          │
│    --cap-drop=ALL \                                                         │
│    --cap-add=sys_nice \       # Only what's needed for Gaudi                │
│    --network=pathfinder-isolated \  # Custom network with egress rules      │
│    --ipc=private \            # Isolated IPC namespace                       │
│    --read-only \              # Read-only root filesystem                    │
│    --tmpfs /tmp:rw,noexec,nosuid \                                          │
│    --security-opt=no-new-privileges \                                       │
│    --security-opt seccomp=pathfinder-seccomp.json \                         │
│    --memory=64g \             # Memory limit                                 │
│    --cpus=16 \                # CPU limit                                    │
│    --pids-limit=1000 \        # Process limit                                │
│    -e HABANA_VISIBLE_DEVICES=all \                                          │
│    -v /verified-models/llama-3:/models:ro \  # Read-only model mount        │
│    -v /secrets/hf_token:/run/secrets/hf_token:ro \  # Secret mount          │
│    pathfinder/gaudi-secure:1.23.0                                           │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ RUNTIME SECURITY CONTROLS                                              │ │
│  │ ──────────────────────────                                             │ │
│  │                                                                        │ │
│  │ 1. Signature Verification at Load Time                                 │ │
│  │    ────────────────────────────────────                                │ │
│  │    Before any model.from_pretrained(), verify signature:               │ │
│  │                                                                        │ │
│  │    def secure_load(model_path):                                        │ │
│  │        if not verify_signature(model_path):                            │ │
│  │            raise SecurityError("Signature verification failed")       │ │
│  │        return AutoModel.from_pretrained(                               │ │
│  │            model_path,                                                 │ │
│  │            trust_remote_code=False  # ALWAYS                           │ │
│  │        )                                                               │ │
│  │                                                                        │ │
│  │ 2. Watchdog Process                                                    │ │
│  │    ─────────────────                                                   │ │
│  │    - Monitor for anomalies (CPU spike, memory leak, network attempts)  │ │
│  │    - Auto-terminate after timeout                                      │ │
│  │    - Kill on suspicious syscalls                                       │ │
│  │                                                                        │ │
│  │ 3. Output Sanitization                                                 │ │
│  │    ─────────────────────────────────────                               │ │
│  │    - Redact any secrets from logs                                      │ │
│  │    - Presidio scan on all outputs                                      │ │
│  │    - No raw error traces in reports                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ NETWORK EGRESS RULES (iptables/nftables)                               │ │
│  │ ─────────────────────────────────────────                              │ │
│  │                                                                        │ │
│  │ # Default deny all egress                                              │ │
│  │ iptables -P OUTPUT DROP                                                │ │
│  │                                                                        │ │
│  │ # Allow only Hugging Face for model fetch                              │ │
│  │ iptables -A OUTPUT -d huggingface.co -p tcp --dport 443 -j ACCEPT      │ │
│  │ iptables -A OUTPUT -d cdn-lfs.huggingface.co -p tcp --dport 443 -j ACCEPT │
│  │                                                                        │ │
│  │ # Allow DNS                                                            │ │
│  │ iptables -A OUTPUT -p udp --dport 53 -j ACCEPT                         │ │
│  │                                                                        │ │
│  │ # Log and drop everything else                                         │ │
│  │ iptables -A OUTPUT -j LOG --log-prefix "PATHFINDER_BLOCKED: "          │ │
│  │ iptables -A OUTPUT -j DROP                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Developer Experience (New Workflow)

### Simple Case: Pre-Approved Model

```bash
# Developer runs single command
$ pathfinder enable meta-llama/Llama-3-8B --gaudi-count=8

╭─────────────────────────────────────────────────────────────────╮
│              AI Model Pathfinder v1.0                           │
│              Intel Gaudi Enablement Tool                        │
╰─────────────────────────────────────────────────────────────────╯

[1/5] Preflight Validation
  ✓ Model in allowlist: meta-llama/Llama-3-8B
  ✓ Organization verified: meta-llama (GPG signed)
  ✓ Format check: safetensors only
  ✓ trust_remote_code: not required

[2/5] Secure Download
  ✓ Downloading to quarantine: /tmp/pathfinder/quarantine/llama-3-8b
  ✓ Checksum verified: sha256:abc123...
  ✓ Size: 16.1 GB

[3/5] Security Scan
  ✓ ModelScan: PASSED (0 issues)
  ✓ AST Scanner: PASSED (0 suspicious patterns)
  ✓ Dependency Audit: PASSED (all versions locked)

[4/5] Sign & Verify
  ✓ ECDSA signature generated
  ✓ MLBOM created: mlbom.json
  ✓ Model promoted to verified store

[5/5] Environment Setup
  ✓ Secure container image: pathfinder/gaudi:1.23.0-secure
  ✓ Resource allocation: 8x Gaudi, 64GB RAM
  ✓ Network policy: egress-restricted

╭─────────────────────────────────────────────────────────────────╮
│  ✅ ENABLEMENT COMPLETE                                         │
│                                                                 │
│  Model:     meta-llama/Llama-3-8B                               │
│  Status:    VERIFIED & SIGNED                                   │
│  MLBOM:     /verified/llama-3-8b/mlbom.json                     │
│                                                                 │
│  To run:    pathfinder run meta-llama/Llama-3-8B                │
│  Report:    /reports/llama-3-8b/enablement_report.html          │
╰─────────────────────────────────────────────────────────────────╯
```

### Complex Case: Model Requires trust_remote_code

```bash
$ pathfinder enable helpful-ai/custom-model --gaudi-count=1

╭─────────────────────────────────────────────────────────────────╮
│              AI Model Pathfinder v1.0                           │
╰─────────────────────────────────────────────────────────────────╯

[1/5] Preflight Validation
  ⚠️  Model NOT in allowlist
  ⚠️  Organization NOT verified: helpful-ai
  ✓ Format check: safetensors only
  🚨 trust_remote_code: REQUIRED

╭─────────────────────────────────────────────────────────────────╮
│  ⚠️  HIGH RISK MODEL DETECTED                                   │
│                                                                 │
│  This model requires trust_remote_code=True which allows        │
│  arbitrary Python code execution during model loading.          │
│                                                                 │
│  Custom code files detected:                                    │
│    - modeling_helpfulqa.py                                      │
│    - configuration_helpfulqa.py                                 │
│                                                                 │
│  This requires TWO-PERSON APPROVAL before proceeding.           │
╰─────────────────────────────────────────────────────────────────╯

[ACTION REQUIRED]
  1. Review code at: /tmp/pathfinder/quarantine/helpful-ai/
  2. Request approval: pathfinder request-approval helpful-ai/custom-model
  3. Approver runs: pathfinder approve helpful-ai/custom-model --approver=<id>
  
Alternatively, to proceed with explicit risk acceptance:
  pathfinder enable helpful-ai/custom-model --accept-remote-code-risk \
    --approver-1=alice@intel.com --approver-2=bob@intel.com

Aborting. Model quarantined at: /tmp/pathfinder/quarantine/
```

---

## 6. Output Artifacts

### plan.json (Pre-execution)

```json
{
  "pathfinder_version": "1.0.0",
  "generated_at": "2026-01-26T10:00:00Z",
  "model": {
    "id": "meta-llama/Llama-3-8B",
    "source": "huggingface.co",
    "commit": "abc123def456"
  },
  "preflight": {
    "allowlist_status": "approved",
    "org_verified": true,
    "format_check": "passed",
    "trust_remote_code_required": false
  },
  "security_scan": {
    "modelscan": {"status": "passed", "findings": []},
    "ast_scanner": {"status": "passed", "findings": []},
    "dependency_audit": {"status": "passed", "findings": []}
  },
  "resource_plan": {
    "gaudi_count": 8,
    "estimated_vram_gb": 16,
    "kv_cache_gb": 4
  },
  "execution_plan": {
    "container_image": "pathfinder/gaudi:1.23.0-secure",
    "network_policy": "egress-restricted",
    "timeout_minutes": 60
  }
}
```

### result.json (Post-execution)

```json
{
  "pathfinder_version": "1.0.0",
  "completed_at": "2026-01-26T11:00:00Z",
  "model": {
    "id": "meta-llama/Llama-3-8B",
    "mlbom_path": "/verified/llama-3-8b/mlbom.json"
  },
  "execution": {
    "status": "success",
    "duration_seconds": 3600,
    "container_id": "abc123"
  },
  "benchmarks": {
    "throughput_tokens_per_sec": 150,
    "latency_p50_ms": 45,
    "latency_p99_ms": 120
  },
  "security": {
    "signature_verified": true,
    "trust_remote_code_used": false,
    "network_violations": 0,
    "anomalies_detected": 0
  },
  "audit": {
    "approvers": [],
    "risk_acceptances": [],
    "log_path": "/logs/llama-3-8b/run-20260126.log"
  }
}
```

---

## 7. Security Components Summary

| Security Requirement | Component |
|---------------------|------------|
| AST code scanning | `model_security_scanner.py` |
| Publisher verification | `TRUSTED_PUBLISHERS` list |
| trust_remote_code detection | `check_custom_code_requirement()` |
| Rate limiting (API protection) | `Flask-Limiter` patterns |
| Path sandboxing | `BLOCKED_PATHS` enforcement |
| Injection detection | Pattern detection in secured agent |
| Secret protection | Presidio + redaction |
| PII detection | Microsoft Presidio integration |
| ModelScan integration | `ModelSecurityScanner` class |
| Hash verification | SHA256 registry pattern |
| ECDSA signing | Signing workflow |
| Signature verification | Verification at load time |
| Tamper detection | Signature validation |

---

## 8. Migration Path

### Phase 1: Parallel Operation (Week 1-4)
- Run Pathfinder alongside existing workflow
- Generate MLBOM and reports without blocking
- Build allowlist from successful runs

### Phase 2: Soft Enforcement (Week 5-8)
- Warn on non-compliant models
- Require approval for trust_remote_code
- Log all security events

### Phase 3: Full Enforcement (Week 9+)
- Block non-allowlisted models by default
- Enforce signature verification
- Complete isolation in secure containers

---

## Appendix: Command Reference

```bash
# Enable a model (full security pipeline)
pathfinder enable <model-id> [options]

# Run an already-enabled model
pathfinder run <model-id> [options]

# List verified models
pathfinder list --verified

# Request approval for high-risk model
pathfinder request-approval <model-id>

# Approve a pending request (requires approver role)
pathfinder approve <model-id> --approver=<email>

# View MLBOM for a model
pathfinder mlbom <model-id>

# Generate enablement report
pathfinder report <model-id>

# Scan a model without enabling
pathfinder scan <model-id>
```
