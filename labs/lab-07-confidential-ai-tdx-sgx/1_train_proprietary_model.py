#!/usr/bin/env python3
"""
Step 1: Train a Proprietary AI Model

This simulates a valuable proprietary model that attackers want to steal.
The model weights represent significant IP investment.

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import joblib
import hashlib

MODEL_PATH = "proprietary_model.h5"
VECTORIZER_PATH = "vectorizer.joblib"
METADATA_PATH = "model_metadata.json"

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           Training Proprietary AI Model                               ║
║     (Simulates valuable IP that needs protection)                     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def train_model():
    """Train a model that simulates proprietary ML."""
    print("[Step 1] Training proprietary classification model...")
    
    # Generate synthetic dataset (simulates proprietary training data)
    np.random.seed(42)
    X_train, y_train = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        n_classes=5,
        random_state=42
    )
    
    # Build model architecture (simulates proprietary design)
    model = Sequential([
        Input(shape=(50,)),
        Dense(256, activation='relu', name='proprietary_layer_1'),
        Dense(128, activation='relu', name='proprietary_layer_2'),
        Dense(64, activation='relu', name='proprietary_layer_3'),
        Dense(5, activation='softmax', name='output_layer')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    print("[*] Training model (this represents significant compute investment)...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    final_accuracy = history.history['val_accuracy'][-1]
    print(f"[*] Training complete. Validation accuracy: {final_accuracy:.2%}")
    
    return model, X_train

def calculate_model_hash(model_path):
    """Calculate SHA256 hash of model file."""
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def count_parameters(model):
    """Count trainable parameters."""
    return sum(np.prod(w.shape) for w in model.get_weights())

def save_model_and_metadata(model, X_train):
    """Save model and create metadata."""
    
    # Save model
    model.save(MODEL_PATH)
    print(f"[✓] Model saved: {MODEL_PATH}")
    
    # Calculate model value metrics
    param_count = count_parameters(model)
    model_hash = calculate_model_hash(MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH)
    
    # Create metadata
    metadata = {
        "model_name": "ProprietaryClassifier-v1",
        "version": "1.0.0",
        "parameters": int(param_count),
        "file_size_bytes": model_size,
        "sha256_hash": model_hash,
        "classification": "CONFIDENTIAL",
        "estimated_value_usd": 500000,  # Simulated IP value
        "training_compute_hours": 100,   # Simulated
        "data_samples_used": len(X_train),
        "protection_required": [
            "Model weights (IP)",
            "Architecture design",
            "Training hyperparameters",
            "Inference data"
        ],
        "threats": [
            "Memory extraction by hypervisor",
            "Cloud operator access",
            "Side-channel attacks",
            "Model stealing via inference"
        ]
    }
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[✓] Metadata saved: {METADATA_PATH}")
    
    # Print summary
    print("\n" + "="*60)
    print("PROPRIETARY MODEL SUMMARY")
    print("="*60)
    print(f"  Model Name:     {metadata['model_name']}")
    print(f"  Parameters:     {param_count:,}")
    print(f"  File Size:      {model_size / 1024:.2f} KB")
    print(f"  SHA256 Hash:    {model_hash[:32]}...")
    print(f"  Classification: {metadata['classification']}")
    print(f"  Est. Value:     ${metadata['estimated_value_usd']:,}")
    print("="*60)
    
    print("""
⚠️  THREAT MODEL:
    Without confidential computing, this model is vulnerable to:
    
    1. HYPERVISOR ATTACK
       Cloud operators can dump VM memory and extract weights
       
    2. MEMORY SCRAPING
       Privileged OS processes can read model from RAM
       
    3. COLD BOOT ATTACK
       Physical access allows DRAM extraction
       
    Run 3_memory_attack_demo.py to see the attack in action.
    """)

def main():
    print_banner()
    model, X_train = train_model()
    save_model_and_metadata(model, X_train)
    print("\n[✓] Step 1 complete. Run: python 2_run_inference.py")

if __name__ == "__main__":
    main()
