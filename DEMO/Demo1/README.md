# Demo1: Malicious Code Injection in ML Models

**Author:** GopeshK | [MIT License](../LICENSE)

> ⚠️ Educational purposes only. Do not use for malicious activities.

## Quick Start

```bash
# Step 1: Train a benign model
python 1_train_model.py

# Step 2: Use the model (works normally)
python 3_consume_model.py

# Step 3: Inject malicious code into model
python 2_inject_malicious_code.py

# Step 4: Use the model again (malicious payload executes!)
python 3_consume_model.py

# Reset
python reset.py
```

## What This Demonstrates

- **Attack:** Malicious Lambda layer injected into Keras model
- **Impact:** Code executes automatically during inference
- **MITRE ATLAS:** AML.T0010 (Supply Chain), AML.T0011 (Backdoor)

## Key Takeaway

> Always verify model integrity before loading untrusted models. Use model signing and hash verification.
