# Challenge 4: Model Signing & Tampering Detection

## MITRE ATLAS ATT&CK Techniques Demonstrated

- **AML.T0010** - ML Supply Chain Compromise
- **AML.T0011** - Backdoor ML Model
- **Defense**: Cryptographic model signing (like Sigstore/cosign)

## Demo Flow

### Step 1: Train a benign model
```bash
python 1_train_model.py
```
Creates a legitimate Keras model for Q&A.

### Step 2: Sign the model (Trusted Publisher)
```bash
python 2_sign_model.py
```
- Generates ECDSA key pair (cosign.key, cosign.pub)
- Signs model with private key
- Saves signature (keras_model.h5.sig)

### Step 3: Attacker tampers with model
```bash
python 3_tamper_model.py
```
Simulates supply chain attack - injects backdoor layer.

### Step 4: Consumer verifies signature (DETECTS ATTACK!)
```bash
python 4_verify_and_consume.py
```
- Verifies signature before loading
- **FAILS** because model was tampered
- Refuses to load malicious model

## Key Takeaway

> "Cryptographic signing of ML models enables consumers to detect supply chain attacks. If the model is tampered with, the signature verification fails, preventing execution of potentially malicious code."

## Reset Demo
```bash
rm -f keras_model.h5 keras_model.h5.sig cosign.key cosign.pub vectorizer.joblib responses.json
```
