# AI Security Labs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![MITRE ATLAS](https://img.shields.io/badge/MITRE-ATLAS-red.svg)](https://atlas.mitre.org/)

**Author:** GopeshK

> ⚠️ **DISCLAIMER:** This repository is for **EDUCATIONAL** and **RESEARCH** purposes only. Do not use for malicious activities.

---

## Overview

Hands-on security labs demonstrating common AI/ML attack vectors and defense mechanisms, aligned with the [MITRE ATLAS](https://atlas.mitre.org/) framework.

---

## Labs

| Lab | Topic | Description | Link |
|-----|-------|-------------|------|
| 01 | Supply Chain | Malicious Code Injection in ML Models | [📖 README](lab-01-malicious-code-injection/README.md) |
| 02 | Integrity | Model Signing & Tampering Detection | [📖 README](lab-02-model-signing/README.md) |
| 03 | Exfiltration | Model Stealing via Query Attack | [📖 README](lab-03-model-stealing/README.md) |
| 04 | Data Leakage | RAG Sensitive Data Extraction | [📖 README](lab-04-rag-data-extraction/README.md) |

---

## MITRE ATLAS Techniques Covered

| Technique | Name | Lab |
|-----------|------|-----|
| [AML.T0010](https://atlas.mitre.org/techniques/AML.T0010) | ML Supply Chain Compromise | Lab 01, Lab 02 |
| [AML.T0011](https://atlas.mitre.org/techniques/AML.T0011) | Backdoor ML Model | Lab 01, Lab 02 |
| [AML.T0044](https://atlas.mitre.org/techniques/AML.T0044) | Full ML Model Access | Lab 03 |
| [AML.T0024](https://atlas.mitre.org/techniques/AML.T0024) | Exfiltration via ML Inference API | Lab 03 |
| [AML.T0051](https://atlas.mitre.org/techniques/AML.T0051) | LLM Prompt Injection | Lab 04 |

---

## Quick Start

```bash
# Clone and navigate
git clone <repo-url>
cd ai-security/labs

# Install base dependencies
pip install -r requirements.txt

# Navigate to a lab
cd lab-01-malicious-code-injection
```

---

## Reset All Labs

```bash
python reset_all.py
```

---

## Contributing

Contributions welcome! Please read the lab READMEs and follow the existing code style.

---

## License

[MIT License](LICENSE)
