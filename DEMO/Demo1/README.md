# Challenge 1: Malicious Code Injection in ML Models

## MITRE ATLAS ATT&CK Techniques Demonstrated

- **AML.T0010** - ML Supply Chain Compromise
- **AML.T0011** - Backdoor ML Model
- **AML.T0020** - Poison Training Data (variant: code injection)

## Attack Scenario

An attacker compromises a trained ML model by injecting malicious code through a Lambda layer. When the model is loaded and used for inference, the malicious code executes automatically.

## Demo Flow

### Step 1: Train a benign model
```bash
python 1_train_model.py
```
Creates a legitimate Keras Q&A model trained on benign data.

### Step 2: Inject malicious code
```bash
python 2_inject_malicious_code.py
```
- Loads the benign model
- Wraps it with a malicious Lambda layer
- Overwrites the original model file (supply chain attack)

### Step 3: Consume the compromised model
```bash
python 3_consume_model.py
```
- Loads the model (malicious code executes!)
- Demonstrates the attack succeeding silently

## Key Takeaway

> "ML models can contain executable code. Always verify model integrity before loading untrusted models. Use model signing and hash verification to detect tampering."

## Reset Demo
```bash
python reset.py
```
Or manually:
```bash
rm -f model.h5 vectorizer.joblib responses.json
```
