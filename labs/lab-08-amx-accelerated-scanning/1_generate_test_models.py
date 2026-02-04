#!/usr/bin/env python3
"""
Step 1: Generate Test Models for Security Scanning

Creates a variety of test models (benign and suspicious) to demonstrate
security scanning performance with Intel AMX acceleration.

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime

# Create test models directory
TEST_MODELS_DIR = Path("test_models")
TEST_MODELS_DIR.mkdir(exist_ok=True)

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         Generating Test Models for Security Scanning                  ║
║              (Benign + Suspicious patterns)                           ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def generate_keras_models(count=5):
    """Generate Keras models of varying sizes."""
    print(f"\n[1/4] Generating {count} Keras models...")
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    
    models_info = []
    
    for i in range(count):
        # Vary model size
        layers = [32, 64, 128, 256, 512][i % 5]
        
        model = Sequential([
            Input(shape=(50,)),
            Dense(layers, activation='relu'),
            Dense(layers // 2, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        model_path = TEST_MODELS_DIR / f"keras_model_{i+1}.h5"
        model.save(model_path)
        
        size_kb = model_path.stat().st_size / 1024
        param_count = model.count_params()
        
        models_info.append({
            "path": str(model_path),
            "format": "keras_h5",
            "size_kb": round(size_kb, 2),
            "parameters": param_count,
            "type": "benign"
        })
        
        print(f"    [✓] {model_path.name}: {size_kb:.1f} KB, {param_count:,} params")
    
    return models_info

def generate_sklearn_models(count=5):
    """Generate sklearn models (pickle format)."""
    print(f"\n[2/4] Generating {count} sklearn models (pickle)...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    import joblib
    
    models_info = []
    
    model_types = [
        ("random_forest", lambda: RandomForestClassifier(n_estimators=100)),
        ("logistic_regression", lambda: LogisticRegression()),
        ("svm", lambda: SVC()),
        ("random_forest_large", lambda: RandomForestClassifier(n_estimators=500)),
        ("logistic_multiclass", lambda: LogisticRegression(max_iter=200)),  # multi_class deprecated
    ]
    
    # Generate dummy training data
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 3, 100)
    
    for i, (name, model_fn) in enumerate(model_types[:count]):
        model = model_fn()
        model.fit(X, y)
        
        model_path = TEST_MODELS_DIR / f"sklearn_{name}.pkl"
        joblib.dump(model, model_path)
        
        size_kb = model_path.stat().st_size / 1024
        
        models_info.append({
            "path": str(model_path),
            "format": "pickle",
            "size_kb": round(size_kb, 2),
            "type": "benign"
        })
        
        print(f"    [✓] {model_path.name}: {size_kb:.1f} KB")
    
    return models_info

def generate_suspicious_models(count=3):
    """Generate models with suspicious patterns for testing detection."""
    print(f"\n[3/4] Generating {count} suspicious models (for testing detection)...")
    
    models_info = []
    
    # Suspicious pattern 1: Model with eval in metadata
    suspicious_1 = {
        "model_type": "custom",
        "weights": np.random.randn(100, 50).tolist(),
        "config": {
            "layers": ["dense", "dense"],
            # Suspicious: code that could be executed
            "init_code": "eval(config.get('payload', ''))",
        }
    }
    path_1 = TEST_MODELS_DIR / "suspicious_eval_model.pkl"
    with open(path_1, 'wb') as f:
        pickle.dump(suspicious_1, f)
    
    models_info.append({
        "path": str(path_1),
        "format": "pickle",
        "size_kb": round(path_1.stat().st_size / 1024, 2),
        "type": "suspicious",
        "pattern": "eval() in config"
    })
    print(f"    [✓] {path_1.name}: Contains eval() pattern")
    
    # Suspicious pattern 2: Model with subprocess reference
    suspicious_2 = {
        "model_type": "pipeline",
        "weights": np.random.randn(50, 25).tolist(),
        "preprocessing": {
            "type": "custom",
            # Suspicious: subprocess import
            "imports": ["subprocess", "os"],
            "code": "subprocess.run(['echo', 'test'])"
        }
    }
    path_2 = TEST_MODELS_DIR / "suspicious_subprocess_model.pkl"
    with open(path_2, 'wb') as f:
        pickle.dump(suspicious_2, f)
    
    models_info.append({
        "path": str(path_2),
        "format": "pickle",
        "size_kb": round(path_2.stat().st_size / 1024, 2),
        "type": "suspicious",
        "pattern": "subprocess reference"
    })
    print(f"    [✓] {path_2.name}: Contains subprocess pattern")
    
    # Suspicious pattern 3: Model with __reduce__ (pickle RCE vector)
    class SuspiciousClass:
        def __init__(self):
            self.data = np.random.randn(30, 30)
        
        def __reduce__(self):
            # This is the classic pickle RCE vector
            return (print, ("This could be malicious code execution!",))
    
    path_3 = TEST_MODELS_DIR / "suspicious_reduce_model.pkl"
    # Note: We won't actually create a working exploit, just the pattern
    suspicious_3 = {
        "model_type": "custom",
        "weights": np.random.randn(30, 30).tolist(),
        "__reduce__": "os.system('whoami')",  # Pattern, not actual code
    }
    with open(path_3, 'wb') as f:
        pickle.dump(suspicious_3, f)
    
    models_info.append({
        "path": str(path_3),
        "format": "pickle", 
        "size_kb": round(path_3.stat().st_size / 1024, 2),
        "type": "suspicious",
        "pattern": "__reduce__ pattern"
    })
    print(f"    [✓] {path_3.name}: Contains __reduce__ pattern")
    
    return models_info

def generate_large_models(count=2):
    """Generate larger models to test scanning performance."""
    print(f"\n[4/4] Generating {count} larger models (for performance testing)...")
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    
    models_info = []
    
    for i in range(count):
        # Create larger model
        layers = 1024 * (i + 1)
        
        model = Sequential([
            Input(shape=(256,)),
            Dense(layers, activation='relu'),
            Dense(layers, activation='relu'),
            Dense(layers // 2, activation='relu'),
            Dense(100, activation='softmax')
        ])
        
        model_path = TEST_MODELS_DIR / f"large_model_{i+1}.h5"
        model.save(model_path)
        
        size_mb = model_path.stat().st_size / (1024 * 1024)
        param_count = model.count_params()
        
        models_info.append({
            "path": str(model_path),
            "format": "keras_h5",
            "size_mb": round(size_mb, 2),
            "parameters": param_count,
            "type": "benign"
        })
        
        print(f"    [✓] {model_path.name}: {size_mb:.1f} MB, {param_count:,} params")
    
    return models_info

def compute_hashes(models_info):
    """Compute SHA256 hashes for all models."""
    print("\n[+] Computing model hashes...")
    
    for model in models_info:
        path = Path(model["path"])
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        model["sha256"] = sha256.hexdigest()
    
    return models_info

def save_manifest(all_models):
    """Save manifest of all generated models."""
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_models": len(all_models),
        "benign_count": sum(1 for m in all_models if m["type"] == "benign"),
        "suspicious_count": sum(1 for m in all_models if m["type"] == "suspicious"),
        "models": all_models
    }
    
    manifest_path = TEST_MODELS_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path

def main():
    print_banner()
    
    all_models = []
    
    # Generate different types of models
    all_models.extend(generate_keras_models(5))
    all_models.extend(generate_sklearn_models(5))
    all_models.extend(generate_suspicious_models(3))
    all_models.extend(generate_large_models(2))
    
    # Compute hashes
    all_models = compute_hashes(all_models)
    
    # Save manifest
    manifest_path = save_manifest(all_models)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Total models generated: {len(all_models)}")
    print(f"  Benign models: {sum(1 for m in all_models if m['type'] == 'benign')}")
    print(f"  Suspicious models: {sum(1 for m in all_models if m['type'] == 'suspicious')}")
    print(f"  Manifest saved: {manifest_path}")
    print("="*60)
    
    print("\n[✓] Step 1 complete. Run: python 2_sequential_scan.py")

if __name__ == "__main__":
    main()
