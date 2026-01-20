# Lab 01: HuggingFace Supply Chain Attack

## 🎯 Overview

This lab demonstrates how **`trust_remote_code=True`** in HuggingFace's `transformers` library can be exploited to execute arbitrary code - including a **reverse shell** that gives attackers full access to your machine.

**Attack scenario:** An attacker uploads a malicious model to HuggingFace Hub. When a developer loads it with `trust_remote_code=True`, the attacker gets shell access.

---

## 🔥 The Vulnerability

```python
from transformers import AutoModelForCausalLM

# This single line can compromise your machine!
model = AutoModelForCausalLM.from_pretrained(
    "helpful-ai/super-fast-qa-bert",
    trust_remote_code=True  # ← Downloads & executes .py files!
)
```

When `trust_remote_code=True`:
1. HuggingFace reads `config.json` with `auto_map` pointing to custom code
2. Downloads Python files (e.g., `modeling_helpfulqa.py`) to cache
3. **Imports and executes** the Python code during model instantiation
4. Malicious code in `__init__` runs with your privileges!

---

## 📁 Lab Structure

```
lab-01-supply-chain-attack/
├── 1_attacker_listener.py          # Attacker's reverse shell listener
├── 2_victim_loads_model.py         # Victim loads model (gets compromised)
├── 3_safe_model_loading.py         # Safe version with security scanner
├── model_security_scanner.py       # Reusable security scanner class
├── hub_cache/                      # Simulated ~/.cache/huggingface/hub/
│   └── models--helpful-ai--super-fast-qa-bert/
│       ├── config.json             # Points to custom code via auto_map
│       ├── modeling_helpfulqa.py   # Custom model code (contains backdoor)
│       ├── model.safetensors       # Model weights (~350MB, generated)
│       ├── tokenizer_config.json   # Tokenizer configuration
│       ├── special_tokens_map.json # Special tokens
│       ├── vocab.json              # Vocabulary (downloaded)
│       └── merges.txt              # BPE merges (downloaded)
├── Makefile                        # Easy setup and run commands
├── requirements.txt
├── .env.example                    # Configuration template
└── .env                            # Your configuration (gitignored)
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- Linux/macOS (uses `fork`, `pty`)
- Two terminal windows

### Setup

```bash
cd lab-01-supply-chain-attack
make setup
```

This will:
- Create virtual environment
- Install dependencies
- Generate model weights (~350MB)
- Download tokenizer files
- Create `.env` from template

### Configure

For **local testing** (same machine):
```bash
echo "ATTACKER_HOST=127.0.0.1
ATTACKER_PORT=4444" > .env
```

For **two-machine demo**:
```bash
# Edit .env with attacker's IP
ATTACKER_HOST=<attacker-ip>
ATTACKER_PORT=4444
```

---

## 🎬 Running the Demo

### Demo 1: The Attack

**Terminal 1 - Attacker:**
```bash
make run-attacker
# Or: python 1_attacker_listener.py
```

**Terminal 2 - Victim:**
```bash
make run-victim
# Or: python 2_victim_loads_model.py
```

**What happens:**
1. Victim runs standard HuggingFace model loading code
2. Model's `__init__` triggers reverse shell (disguised as "telemetry")
3. Attacker gets full bash shell on victim's machine
4. Victim sees normal chatbot - doesn't notice compromise!

**Attacker's view:**
```
🚨 SHELL CONNECTED!

$ whoami
victim
$ cat ~/.ssh/id_rsa
-----BEGIN OPENSSH PRIVATE KEY-----
...
```

### Demo 2: The Defense

**Terminal 2 - Safe User:**
```bash
make run-safe
# Or: python 3_safe_model_loading.py
```

**Output:**
```
🔍 WAIT! Let me inspect the downloaded files first...

[1/3] Checking if model requires custom code execution...
  ⚠️  Model requires trust_remote_code=True
  ⚠️  Will execute: ['modeling_helpfulqa.HelpfulQAForCausalLM']

[2/3] Inspecting downloaded files...
  Found 7 file(s) in cache:
     ⚠️  modeling_helpfulqa.py (EXECUTABLE CODE)
     - config.json
     ...

[3/3] Scanning downloaded code for red flags...
  🚨 DANGEROUS CODE DETECTED:
     - modeling_helpfulqa.py: Network socket creation
     - modeling_helpfulqa.py: Process forking
     - modeling_helpfulqa.py: PTY shell spawning

════════════════════════════════════════════════════════════
  ❌ NOT SAFE to load with trust_remote_code=True
════════════════════════════════════════════════════════════
```

---

## 🛡️ Defense Strategies

### First Line of Defense
**Avoid `trust_remote_code=True` entirely!**

```python
# ❌ DANGEROUS
model = AutoModelForCausalLM.from_pretrained("unknown/model", trust_remote_code=True)

# ✅ SAFE - Use only standard architectures
model = AutoModelForCausalLM.from_pretrained("gpt2")

# ✅ SAFE - Explicit trust_remote_code=False (default)
model = AutoModelForCausalLM.from_pretrained("model", trust_remote_code=False)
```

### If You Must Use Custom Code

```python
from model_security_scanner import ModelSecurityScanner

# Scan BEFORE loading
scanner = ModelSecurityScanner("/path/to/model/cache")
if scanner.scan():
    model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
else:
    print("Malicious code detected!", scanner.findings)
```

### Security Scanner (model_security_scanner.py)

The scanner implements 5 defense layers:

| Layer | Check | What It Detects |
|-------|-------|-----------------|
| 1 | Publisher Verification | Unknown/untrusted publishers |
| 2 | Dangerous Pattern Scan | socket, subprocess, os.system, eval, pickle |
| 3 | Entropy Analysis | Obfuscated/encoded payloads (base64, hex) |
| 4 | AST Import Analysis | Dangerous import chains |
| 5 | File Hash Registry | Track known-good vs modified files |

### Additional Protections

| Setting | Custom .py files | Pickle weights |
|---------|-----------------|----------------|
| `trust_remote_code=False` | ✅ Blocked | ⚠️ Still risky |
| `use_safetensors=True` | N/A | ✅ Safe |
| Both | ✅ Safe | ✅ Safe |

**Safest approach:**
```python
model = AutoModelForCausalLM.from_pretrained(
    "model",
    trust_remote_code=False,  # No custom code
    use_safetensors=True,     # No pickle deserialization
)
```

---

## 🔧 Makefile Commands

```bash
make help         # Show all commands
make setup        # Full setup (venv + deps + model + tokenizer)
make install      # Install dependencies only

make run-attacker # Start attacker listener
make run-victim   # Run victim script
make run-safe     # Run safe version with scanner

make clean        # Remove generated files
make reset        # Kill processes + clean pycache
```

---

## 🧹 Reset Lab

```bash
make reset
# Or: python reset.py
```

---

## ⚠️ Disclaimer

**FOR EDUCATIONAL PURPOSES ONLY.** 

This lab demonstrates security vulnerabilities to help defenders understand and mitigate risks. Do not use these techniques maliciously.
