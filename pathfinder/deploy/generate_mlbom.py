#!/usr/bin/env python3
"""
Pathfinder MLBOM Generator
Generates Machine Learning Bill of Materials for verified models.
"""
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime


def sha256_file(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_mlbom(model_path: str, model_id: str, scan_passed: bool = True,
                   trust_remote_code: bool = False) -> dict:
    """Generate MLBOM for a model."""
    model_path = Path(model_path)
    
    # Collect file hashes
    files = []
    total_size = 0
    
    for f in sorted(model_path.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            files.append({
                "path": str(f.relative_to(model_path)),
                "size": size,
                "sha256": sha256_file(f) if size < 100_000_000 else "skipped_large_file"
            })
    
    # Detect format
    formats = set(f["path"].split(".")[-1] for f in files if "." in f["path"])
    primary_format = "safetensors" if "safetensors" in formats else "unknown"
    
    mlbom = {
        "mlbom_version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": {
            "id": model_id,
            "name": model_id.split("/")[-1],
            "local_path": str(model_path),
            "total_size_bytes": total_size,
            "file_count": len(files)
        },
        "security": {
            "scan_passed": scan_passed,
            "scan_timestamp": datetime.utcnow().isoformat() + "Z",
            "trust_remote_code": trust_remote_code,
            "primary_format": primary_format
        },
        "provenance": {
            "source": "huggingface.co",
            "downloaded_by": os.environ.get("USER", "unknown"),
            "host": os.uname().nodename
        },
        "files": files[:50]  # Limit to first 50 files
    }
    
    return mlbom


def main():
    parser = argparse.ArgumentParser(description="Generate MLBOM for model")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--model-id", required=True, help="Model ID (e.g., meta-llama/Llama-3.1-8B)")
    parser.add_argument("--scan-passed", action="store_true", default=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--output", "-o", help="Output file (default: model_path/mlbom.json)")
    
    args = parser.parse_args()
    
    mlbom = generate_mlbom(
        args.model_path, 
        args.model_id,
        args.scan_passed,
        args.trust_remote_code
    )
    
    output_path = args.output or str(Path(args.model_path) / "mlbom.json")
    Path(output_path).write_text(json.dumps(mlbom, indent=2))
    
    print(f"[SUCCESS] MLBOM generated: {output_path}")
    print(json.dumps(mlbom, indent=2))


if __name__ == "__main__":
    main()
